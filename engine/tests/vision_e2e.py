"""Engine side of images / videos against the golden dump (run through tools/tp2.sh on C and D), without our vision
front end: the golden's own media embeddings and mRoPE positions go into the engine (set_mrope / set_media /
media_src), the prompt runs in chunks, then the golden output tokens one step at a time (teacher forcing).
Compared on rank 0 for every vision request of the dump:
  - logits of the last prompt row and of the dumped decode steps vs the golden's (top-1, max |Δlogit|, total
    variation), and log-probs of the golden top-20 at every output token (top-1 agreement, max |Δ|);
  - chunkings (--chunks) give bitwise equal logits (an image / video block split over steps, compressed indexer keys
    whose group starts in an earlier step);
  - the materialised state after every layer at the first output token vs the golden's (`layers.<i>#0.resid`): the
    smallest cosine over the 48 layers;
  - control: the same request with plain positions (no set_mrope) — shows what the 3D positions change;
  - noise floor: the reference vs itself (chart vs chart2 in tokens.json);
  - --ours: the request again with our own front end (server/vision.py from the dump's media files) and our vision
    tower (server/vision_tower.py, CUDA, rank 0) instead of the golden's embeddings — the answer-level difference of
    our tower (bf16 activations, one rounding) vs the reference's (MXFP8 activations).
One line per request on screen; details in /work/results/tests/vision_e2e/vision_e2e{rank}.log.
  tools/tp2.sh vision python3 engine/tests/vision_e2e.py [--cases chart,clip] [--chunks 2048,64]
"""
import argparse
import json
import os
import sys
import time

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "server"))
import qlog  # noqa: E402

import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--dump", default="/work/results/raw/dump/20261113-1910")
ap.add_argument("--chunks", default="2048,64", help="prompt chunkings; logits must be bitwise equal across them")
ap.add_argument("--cases", default="chart,two,rgba,turns,clip,vlong")
ap.add_argument("--control", default="chart,clip", help="cases also run with plain positions")
ap.add_argument("--yarn", type=float, default=4.0, help="YaRN factor the dump was made with")
ap.add_argument("--ours", action="store_true", help="also with our front end + vision tower (see above)")
a = ap.parse_args()
OUT = "/work/results/tests/vision_e2e"
os.makedirs(OUT, exist_ok=True)
qlog.detail_file(f"{OUT}/vision_e2e{a.rank}.log")
eng = ext.load("engine", ext.engine_sources(), log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
CH = [int(x) for x in a.chunks.split(",")]
eng.init("/models/Qwen3.8-Flash-Next-FP4-Local", "/work/results/tensor_table.tsv", a.rank, os.environ["Q38_HEAD_IP"],
         1, 16384, max(CH), yarn_factor=a.yarn, yarn_ramp=262144)

# requests of the dump in client order (tools/dump/client.py): 3 text, then the vision cases
NAMES = ["short", "long", "short2", "chart", "two", "rgba", "turns", "clip", "vlong", "chart2"]
VKEY = {"vlong": "long"}  # tokens.json["vision"]["cases"] name
RD = f"{a.dump}/rank{a.rank}"
reqs = [json.loads(x) for x in open(f"{RD}/requests.jsonl")]
man = [json.loads(x) for x in open(f"{RD}/manifest.jsonl")]
tok = json.load(open(f"{a.dump}/tokens.json"))
MEDIA = (248056, 248057)
LG = "language_model.logits_processor#0.out"


def steps_of(rid):
    return [m["step"] for m in man if rid in m["reqs"]]


def tensors(step, keys):
    with safe_open(f"{RD}/step{step:04d}.safetensors", "pt") as f:
        have = set(f.keys())
        return {k: f.get_tensor(k) if k in have else None for k in keys}


def tv(x, y):
    p, q = torch.softmax(x.float(), -1), torch.softmax(y.float(), -1)
    return 0.5 * (p - q).abs().sum().item()


def run(ids, outs, pos3, delta, rows, emb, chunk, mrope=True):
    """prompt in chunks, then the output tokens; logits rows (cpu fp32): last prompt row, then one per output token."""
    eng.reset(0)
    if mrope:
        eng.set_mrope(0, pos3, delta)
    eng.set_media(0, rows)
    if a.rank == 0:
        eng.media_src(0, emb)
    lgs = []
    p = 0
    while p < len(ids):
        n = min(chunk, len(ids) - p)
        lg = eng.step(ids[p:p + n], list(range(p, p + n)), [0], [0, n], False)
        p += n
    lgs.append(lg[-1].float().cpu())
    xs = []
    for k, t in enumerate(outs[:-1]):
        eng.set_taps(a.rank == 0 and k == 0)
        lg = eng.step([t], [p], [0], [0, 1], True)
        if a.rank == 0 and k == 0:
            xs = [eng.tap(f"X.{i}", 1, 10240).float().flatten().cpu() for i in range(48)]
        lgs.append(lg[0].float().cpu())
        p += 1
    eng.set_taps(False)
    torch.cuda.synchronize()
    return lgs, xs


v = tok["vision"]["cases"]
tower = None
if a.ours and a.rank == 0:
    import vision
    from vision_tower import Tower
    t0 = time.time()
    tower = Tower("/models/Qwen3.8-Flash-Next-FP4-Local", "cuda")
    qlog.info("vision_e2e", f"our tower loaded · {tower.nbytes() / 2**30:.2f} GiB · {time.time() - t0:.1f} s")


def our_emb(case):
    """our pixels → our tower (rank 0): bf16 [rows, 2560], tower ms, cosine vs the golden rows"""
    ms = []
    for f in case["files"]:
        data = open(f"{a.dump}/media/{f}", "rb").read()
        ms.append(vision.load_image(data) if f.endswith(".png") else vision.load_video(data))
    torch.cuda.synchronize()
    t0 = time.time()
    e = tower.embed(ms)
    torch.cuda.synchronize()
    return e.contiguous(), (time.time() - t0) * 1e3

a1, a2 = v["chart"], v["chart2"]
floor = max(abs(x - y) for x, y in zip(a1["logprobs"], a2["logprobs"]))
if a.rank == 0:
    qlog.info("vision_e2e", f"noise floor (reference chart vs chart2): max |Δlogprob| of the chosen token {floor:.3f}")
bad = 0
for name in a.cases.split(","):
    i = NAMES.index(name)
    r = reqs[i]
    case = v[VKEY.get(name, name)]
    ids, outs = r["prompt_ids"], case["output_ids"]
    assert ids == case["prompt_ids"], f"{name}: requests.jsonl and tokens.json prompts differ"
    st = steps_of(r["req"])
    pre = st[0]
    g = tensors(pre, ["language_model.model#0.kw.positions",
                      "Qwen4ExpForConditionalGeneration.fn.embed_input_ids#0.ret", LG])
    P = g["language_model.model#0.kw.positions"]
    assert P.shape == (3, len(ids)), (name, P.shape)
    pos3 = P.t().reshape(-1).tolist()
    delta = int(P.max()) + 1 - len(ids)
    rows = [j for j, t in enumerate(ids) if t in MEDIA]
    emb = None
    if a.rank == 0:
        emb = g["Qwen4ExpForConditionalGeneration.fn.embed_input_ids#0.ret"][rows].contiguous().cuda()
    t0 = time.time()
    res, xs = {}, None
    for c in CH:
        res[c], x = run(ids, outs, pos3, delta, rows, emb, c)
        xs = xs or x
    dt = time.time() - t0
    if a.rank != 0:
        if name in a.control.split(","):
            run(ids, outs, pos3, delta, rows, emb, CH[0], mrope=False)
        if a.ours:
            run(ids, outs, pos3, delta, rows, None, CH[0])
        continue
    base = res[CH[0]]
    same = all(torch.equal(x, y) for c in CH[1:] for x, y in zip(base, res[c]))
    # golden full logits: last prompt row (prefill step keeps the last rows) and the dumped decode steps
    cmp = []
    gl = g[LG]
    if gl is not None:
        cmp.append(("prompt", base[0], gl[-1].float()))
    for k, s in enumerate(st[1:], 1):
        x = tensors(s, [LG])[LG]
        if x is not None and k < len(base):
            cmp.append((f"dec{k}", base[k], x[-1].float()))
    for lab, x, y in cmp:
        qlog.detail(f"{name} {lab}: top1 {int(x.argmax())}/{int(y.argmax())} · max |Δlogit| "
                    f"{(x - y).abs().max().item():.3f} · TV {tv(x, y):.4f}")
    def resid_cos(xl):  # smallest cosine over the layers at the first output token (golden decode step 1)
        gr = tensors(st[1], [f"language_model.model.layers.{i}#0.resid" for i in range(48)])
        cs = [torch.nn.functional.cosine_similarity(xl[i], gr[f"language_model.model.layers.{i}#0.resid"].float()
                                                    .flatten(), 0).item() for i in range(48)]
        qlog.detail(f"{name} resid cos per layer: " + " ".join(f"{c:.5f}" for c in cs))
        return min(cs), cs.index(min(cs))
    cmin, cl = resid_cos(xs)
    # golden top-20 log-probs at every output token
    top1 = dl = 0
    for k, lp in enumerate(base[:len(outs)]):
        ls = torch.log_softmax(lp, -1)
        top1 += int(ls.argmax()) == outs[k]
        for t, gv in case["top_logprobs"][k].items():
            dl = max(dl, abs(ls[int(t)].item() - gv))
    ctrl = ""
    if name in a.control.split(","):
        c, cx = run(ids, outs, pos3, delta, rows, emb, CH[0], mrope=False)
        ct1 = sum(int(torch.log_softmax(lp, -1).argmax()) == outs[k] for k, lp in enumerate(c[:len(outs)]))
        ccmin, ccl = resid_cos(cx)
        ctrl = f" · plain positions: top-1 {ct1}/{len(outs)}, resid cos min {ccmin:.4f} (layer {ccl})"
    full = " · ".join(f"{lab} TV {tv(x, y):.4f}" for lab, x, y in cmp)
    if a.ours:
        oe, oms = our_emb(case)
        ecos = torch.nn.functional.cosine_similarity(oe.float(), emb.float(), -1)
        o, ox = run(ids, outs, pos3, delta, rows, oe, CH[0])
        ot1 = sum(int(torch.log_softmax(lp, -1).argmax()) == outs[k] for k, lp in enumerate(o[:len(outs)]))
        odl = max(abs(torch.log_softmax(o[k], -1)[outs[k]].item() - case["logprobs"][k]) for k in range(len(outs)))
        ocmin, ocl = resid_cos(ox)
        otv = " · ".join(f"{lab} TV {tv(o[k], y):.4f}" for k, (lab, _, y) in enumerate(cmp))
        ctrl += (f" · OUR TOWER ({oms:.0f} ms; emb cos min {ecos.min().item():.4f} mean {ecos.mean().item():.4f}): "
                 f"top-1 {ot1}/{len(outs)}, max |Δlogprob| chosen {odl:.3f}, {otv}, resid cos min {ocmin:.4f}")

    ok = same and top1 >= len(outs) - 2
    bad += not ok
    (qlog.ok if ok else qlog.warn)(
        f"vision {name}", f"prompt {len(ids):,} (media {len(rows):,}, delta {delta}) · top-1 {top1}/{len(outs)} · "
        f"max |Δlogprob| top-20 {dl:.3f} · {full} · resid cos min {cmin:.4f} (layer {cl}) · "
        f"chunks {'bitwise equal' if same else 'DIFFER'}{ctrl} · {dt:.1f} s")
if a.rank == 0:
    (qlog.ok if not bad else qlog.warn)("vision_e2e", f"{bad} request(s) off · details {OUT}/vision_e2e0.log")
sys.stdout.flush()
os._exit(0)
