"""Byte floor: lower-bound time per kernel group for decode / MTP rounds / prefill, from config + tensor table (tp column)
+ measured GPU numbers (tools/bw_probe.py). Per rank, TP=2, plan P0 (vLLM split; tools/tp_plan.py) unless a variant says so.

Floor of a step = Σ_ops bytes/BW (memory-bound decode) + Σ collectives (latency + bytes/link) + kernel launches (graph).
Prefill floor = FLOPs/peak (compute-bound), memory and link checked as separate bounds.
Assumptions are flagged on screen: AR latency (not measured yet), uniform expert routing (upper bound on distinct
experts), MTP acceptance α per position (iid), GDN state read+written once per round (replay design).
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlog  # noqa: E402
from tp_plan import rank_bytes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--table", required=True)
ap.add_argument("--bw", required=True, help="bw_probe.py JSON")
ap.add_argument("--out", required=True, help="output prefix (.txt detail, .json numbers)")
ap.add_argument("--ctx", type=int, default=131072)
ap.add_argument("--ar-us", type=float, default=6.0, help="all-reduce latency per call, µs (estimate until measured)")
ap.add_argument("--link-gbs", type=float, default=12.5, help="CX7 one rail, GB/s")
ap.add_argument("--alpha", type=float, default=0.7,
                help="MTP per-position acceptance (the reference: 3.13 tok/round ≈ 0.7)")
ap.add_argument("--state", choices=["fp32", "bf16"], default="fp32", help="GDN recurrent state dtype (config: fp32)")
ap.add_argument("--kpl", type=int, default=10, help="kernels per layer in our fused design")
ap.add_argument("--new-tokens", type=int, default=8000, help="next turn: new tokens after a tool result")
ap.add_argument("--comm", help="comm.json from engine/tests/comm_check.py: measured all-reduce µs per M (replaces --ar-us)")
ap.add_argument("--dump", help="golden dump rank0 dir: measure distinct experts over consecutive tokens from router logits")
a = ap.parse_args()

cfg = json.load(open(a.config))["text_config"]
hw = json.load(open(a.bw))
BW = hw["read_best"] * 1e9
LINK = a.link_gbs * 1e9
NL, H = cfg["num_hidden_layers"], cfg["hidden_size"]
QSA = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
NE, TOPK = cfg["num_experts"], cfg["num_experts_per_tok"]
SB = 4 if a.state == "fp32" else 2

# ---- weights per rank from the table -------------------------------------------------------------------------
W = defaultdict(int)          # (layer, group) -> bytes on rank 0
P = defaultdict(float)        # (layer, group) -> params on rank 0 (for prefill FLOPs)
EXP = defaultdict(int)        # layer -> bytes of all routed experts on rank 0
EXPP = defaultdict(float)
lm_params = 0
GROUP = {"attn_hc": "hc", "mlp_hc": "hc", "hc_mixer": "mixer", "gdn": "attn", "qsa": "attn", "qsa_indexer": "attn",
         "router": "router", "shared_expert": "shared", "ple": "ple", "mtp": "mtp_fc", "lm_head": "lm_head"}
with open(a.table) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        b = row["block"]
        if b in ("visual", "embed", "ngram"):
            continue
        rb = rank_bytes(row, 0)
        shape = [int(x) for x in row["shape"].split("x")] if row["shape"] != "-" else [1]
        is_w = row["part"].endswith("weight") and row["dtype"] in ("BF16", "F8_E4M3", "U8", "F32")
        frac = 1.0 if row["tp"] == "rep" else 0.5
        params = math.prod(shape) * (2 if row["dtype"] == "U8" and "weight_scale" not in row["part"] else 1) * frac if is_w else 0
        lay = row["layer"]
        if b == "expert":
            EXP[lay] += rb
            EXPP[lay] += params
        else:
            g = GROUP.get(b, "other")
            W[(lay, g)] += rb
            P[(lay, g)] += params
            if b == "lm_head":
                lm_params = params

per_exp = {k: v / NE for k, v in EXP.items()}            # bytes per expert per rank
per_expp = {k: v / NE for k, v in EXPP.items()}
LM = W[("-1", "lm_head")]
LM_FP4 = LM / 2 * 0.5625                                  # NVFP4 copy of the head for drafting (draft-only, lossless)


def uniq_uniform(m):  # distinct experts per layer for m independent tokens under uniform routing (upper bound)
    return NE * (1 - (1 - TOPK / NE) ** m)


def measured_overlap(d):
    """Distinct experts per layer for windows of T consecutive tokens (B=1) and for B=4 windows from far-apart spots,
    from the router logits in the golden dump (decode steps + kept prefill rows, in step order, per request)."""
    import re
    import torch
    from safetensors.torch import load_file
    man = [json.loads(x) for x in open(os.path.join(d, "manifest.jsonl"))]
    seqs = defaultdict(list)  # req -> list of [48, top10 sets] rows in order
    pat = re.compile(r"\.layers\.(\d+)\.mlp\.gate#0\.out(\.0)?$")
    for m in man:
        if len(m["reqs"]) != 1:
            continue
        rid = next(iter(m["reqs"]))
        t = load_file(os.path.join(d, f"step{m['step']:04d}.safetensors"))
        lay = {int(pat.search(k).group(1)): v for k, v in t.items() if pat.search(k)}
        if len(lay) != NL:
            continue
        top = torch.stack([lay[i].float().topk(TOPK, dim=-1).indices for i in range(NL)], 1)  # [rows, 48, 10]
        seqs[rid].extend(top.tolist())
    seqs = [v for v in seqs.values() if len(v) >= 16]
    if not seqs:
        return None, 0
    u1, u4 = {}, {}
    for T in range(1, 14):
        acc, n = 0.0, 0
        for v in seqs:
            for s0 in range(0, len(v) - T + 1):
                acc += sum(len(set().union(*(set(v[s0 + j][l]) for j in range(T)))) for l in range(NL)) / NL
                n += 1
        u1[T] = acc / n if n else None
        acc, n = 0.0, 0
        for s0 in range(0, 64):  # 4 windows: two per sequence, far apart, rotating offsets
            ws = []
            for v in seqs[:2]:
                L = len(v)
                for o in (s0 % max(1, L - T), (s0 + L // 2) % max(1, L - T)):
                    ws.append(v[o:o + T])
            if len(ws) < 4:
                break
            acc += sum(len(set().union(*(set(r[l]) for w in ws[:4] for r in w))) for l in range(NL)) / NL
            n += 1
        u4[T] = acc / n if n else None
    return {"B1": u1, "B4": u4}, sum(len(v) for v in seqs)


OVL, OVL_N = measured_overlap(a.dump) if a.dump else (None, 0)


def uniq(m, B=1, T=None):
    if OVL and T is not None and B in (1, 4) and OVL[f"B{B}"].get(T):
        return OVL[f"B{B}"][T]
    return uniq_uniform(m)


COMM = {int(k): v["graph_best"] for k, v in json.load(open(a.comm)).items()} if a.comm else None


def ar(n, m, extra_bytes=0):
    if COMM:  # measured [m,2560] bf16 all-reduce, linear interpolation in m
        ks = sorted(COMM)
        lo = max([k for k in ks if k <= m] or [ks[0]])
        hi = min([k for k in ks if k >= m] or [ks[-1]])
        us = COMM[lo] if hi == lo else COMM[lo] + (COMM[hi] - COMM[lo]) * (m - lo) / (hi - lo)
        return n * (us * 1e-6 + extra_bytes / LINK)
    return n * (a.ar_us * 1e-6 + (5120 * m + extra_bytes) / LINK)


# ---- one target-model step: B sequences × T tokens at context L -----------------------------------------------
def step(B, T, L, hc_split=False):
    M = B * T
    ops = defaultdict(float)  # name -> bytes
    ops["ple(proj+conv)"] += W[("1", "ple")] + B * 2 * (9 + T) * 10240 * 2
    ops["embed+ngram rows"] += M * (H * 2 + 16 * 90)
    for i in range(NL):
        li = str(i)
        hc = W[(li, "hc")] / (2 if hc_split else 1)
        ops["hc"] += hc
        if i in QSA:
            ops["qsa proj"] += W[(li, "attn")]
            ops["indexer keys"] += B * (L // 4) * 128 * 2
            ops["sparse kv"] += B * min(L, 2051) * 1024
        else:
            ops["gdn proj"] += W[(li, "attn")]
            ops["gdn state"] += B * 2 * 24 * 128 * 128 * SB + B * 2 * (3 + T) * 5120 * 2
        ops["router"] += W[(li, "router")]
        ops["experts"] += uniq(M, B, T) * per_exp[li]
        ops["shared"] += W[(li, "shared")]
    ops["final mixer"] += W[("-1", "mixer")]
    ops["lm_head"] += LM + M * 124160 * 4
    n_ar = 2 + 2 * NL + (2 * (2 * NL + 1) if hc_split else 0)
    comm = ar(n_ar, M) + (a.ar_us * 1e-6 + M * 124160 * 2 / LINK)
    launch = (NL * a.kpl + 10) * hw["launch_graph_us"] * 1e-6
    mem = sum(ops.values()) / BW
    return {"ops": dict(ops), "mem": mem, "comm": comm, "launch": launch, "t": mem + comm + launch,
            "bytes": sum(ops.values())}


def draft_step(B, fp4_head):
    """MTP layer (index 48) one draft step for B sequences (1 token each)."""
    m = "mtp0"
    b = (W[(m, "hc")] + W[(m, "attn")] + W[(m, "router")] + W[(m, "shared")] + uniq(B, B, 1) * per_exp[m]
         + W[("mtp", "mtp_fc")] + W[("mtp", "mixer")] + (LM_FP4 if fp4_head else LM)
         + B * min(L_CTX, 2051) * 1024)                     # sparse kv of the MTP layer (indexer: once per round, below)
    comm = ar(3, B) + 3 * (a.ar_us * 1e-6) + B * 124160 * 2 / LINK
    launch = (a.kpl + 6) * hw["launch_graph_us"] * 1e-6
    return b / BW + comm + launch, b


L_CTX = a.ctx


def mtp_best(B, fp4_head, hc_split=False):
    best = None
    for k in range(1, 13):
        tau = (1 - a.alpha ** (k + 1)) / (1 - a.alpha)
        d, _ = draft_step(B, fp4_head)
        v = step(B, k + 1, L_CTX, hc_split)
        rt = k * d + B * (L_CTX // 4) * 256 / BW + v["t"]   # + MTP indexer keys, selected at draft step 0 only
        r = {"k": k, "tau": tau, "round_ms": rt * 1e3, "draft_ms": k * d * 1e3, "verify_ms": v["t"] * 1e3,
             "tok_s": B * tau / rt}
        if best is None or r["tok_s"] > best["tok_s"]:
            best = r
    return best


# ---- prefill: FLOPs per token per rank at position p ------------------------------------------------------------
dense_p = sum(v for (lay, g), v in P.items() if g in ("hc", "attn", "router", "shared", "ple", "mixer") and not lay.startswith("mtp"))
exp_p = sum(per_expp[str(i)] for i in range(NL)) * TOPK


def flops_tok(p):
    f = 2 * (dense_p + exp_p)
    f += len(QSA) * (12 * 4 * 256 * min(p, 2051) + 4 * 2 * 128 * (p // 4))   # sparse attn (rank: 12 heads) + indexer (rep)
    f += (NL - len(QSA)) * 3 * 4 * 24 * 128 * 128                            # chunked delta rule ≈ 3× recurrent
    return f


def prefill_rate(L, peak_tflops):
    n = 64
    tot = sum(flops_tok(int(L * (j + 0.5) / n)) for j in range(n)) / n       # average over positions 0..L
    t_comp = tot / (peak_tflops * 1e12)
    t_link = (2 + 2 * NL) * 5120 / LINK                                       # AR bytes per token (overlappable)
    return 1 / max(t_comp, t_link), tot


# ---- report ------------------------------------------------------------------------------------------------------
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
qlog.detail_file(a.out + ".txt")
res = {"assume": vars(a), "hw": hw, "overlap": OVL}
if OVL:
    qlog.ok("floor expert", f"measured on {OVL_N} dump tokens · distinct experts/layer · solo T=4 {OVL['B1'][4]:.1f} "
                           f"(uniform {uniq_uniform(4):.1f}) · 4 seqs×T=3 {OVL['B4'][3]:.1f} "
                           f"(uniform {uniq_uniform(12):.1f})")
else:
    qlog.warn("floor expert", "no dump — uniform routing assumed (upper bound)")
s1 = step(1, 1, L_CTX)
res["decode1"] = s1
ar_txt = f"AR NCCL {COMM[1]:.0f}–{COMM[max(COMM)]:.0f} µs (measured)" if COMM else f"AR {a.ar_us:g} µs (estimate)"
qlog.info("floor", f"bandwidth {hw['read_best']:.0f} GB/s (measured) · link {a.link_gbs} GB/s · {ar_txt} · "
                   f"GDN state {a.state} · α {a.alpha} (assumed) · {L_CTX // 1024}K")
qlog.info("floor 1 token", f"weights+state {s1['bytes'] / 1e9:.2f} GB/rank · memory {s1['mem'] * 1e3:.1f} + comm "
                         f"{s1['comm'] * 1e3:.2f} + launch {s1['launch'] * 1e3:.2f} = {s1['t'] * 1e3:.1f} ms → "
                         f"{1 / s1['t']:.0f} tok/s (no MTP)")
top = sorted(s1["ops"].items(), key=lambda x: -x[1])[:6]
qlog.info("floor top shares", " · ".join(f"{k} {v / 1e9:.2f}" for k, v in top) + " GB")
for B, goal, name in ((1, 110, "solo"), (4, 300, "4 at once")):
    r16, r4 = mtp_best(B, False), mtp_best(B, True)
    res[f"mtp_B{B}"] = {"bf16_head": r16, "fp4_draft_head": r4}
    for r, tag in ((r16, "draft head bf16"), (r4, "draft head NVFP4")):
        (qlog.ok if r["tok_s"] >= goal else qlog.warn)(
            f"floor {name}", f"{tag} · k={r['k']} · τ {r['tau']:.2f} · round {r['round_ms']:.1f} ms "
            f"(draft {r['draft_ms']:.1f} + verify {r['verify_ms']:.1f}) → {r['tok_s']:.0f} tok/s (target {goal})")
hs = step(1, 1, L_CTX, hc_split=True)
qlog.info("floor HC split", f"1 token {s1['t'] * 1e3:.1f} → {hs['t'] * 1e3:.1f} ms (half the HC weights, "
                            f"+{2 * (2 * NL + 1)} ARs included)")
for peak, tag in ((hw["bf16_tflops"], "bf16 A16·lossless"), (hw.get("fp8_tflops") or 0, "fp8 A8·numerics change")):
    if not peak:
        continue
    rate, ft = prefill_rate(L_CTX, peak)
    res[f"prefill_{tag[:4]}"] = {"tok_s": rate, "flops_tok_avg": ft}
    (qlog.ok if rate >= 5000 else qlog.warn)(
        "floor 128K read", f"{tag} · {ft / 1e9:.1f} GFLOP/token/rank · peak {peak:.0f} TFLOPS → {rate:,.0f} tok/s "
                           f"(target 5,000)")
    nt = a.new_tokens * flops_tok(L_CTX) / (peak * 1e12) + s1["t"]
    (qlog.ok if nt <= 1.56 else qlog.warn)(
        "floor next turn", f"{tag} · {a.new_tokens:,} new tokens @{L_CTX // 1024}K → {nt:.2f} s (target 1.56)")
for k in sorted(s1["ops"]):
    qlog.detail(f"decode1\t{k}\t{s1['ops'][k] / 1e6:.1f} MB\t{s1['ops'][k] / BW * 1e6:.0f} us")
json.dump(res, open(a.out + ".json", "w"), indent=1, default=float)
qlog.info("floor", f"details {a.out}.txt · {a.out}.json")
