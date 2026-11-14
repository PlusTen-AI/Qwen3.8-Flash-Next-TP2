#!/usr/bin/env python3
"""tensor_table.py — loader tensor table (TSV) for the Qwen3.8-Flash-Next FP4 checkpoint. Stdlib only.

Reads model.safetensors.index.json + every shard header (8-byte LE length + JSON) and writes one row per tensor,
sorted by (file, offset):
  name file offset nbytes dtype shape layer block part expert qfmt tp
  offset  absolute byte offset of the data in `file` (8 + header_len + data_offsets[0])
  shape   dims joined by 'x', scalar = '-'
  layer   decoder layer index; -1 = model-global; mtpK = mtp.layers.K, mtp = other mtp.*; vN = visual block N, v = other visual
  block   embed ngram ple hc_mixer attn_hc mlp_hc gdn qsa qsa_indexer router expert shared_expert lm_head mtp visual other
          (ngram = ple.ple_embedding.* n-gram tables; ple = the rest of the PLE module; mtp = mtp.fc_*/pre_fc_norm_*;
           mtp.layers.K.* uses the normal sub-block names with layer=mtpK)
  part    suffix after the block prefix (e.g. gate_proj.weight_scale_2)
  expert  routed-expert id or -1
  qfmt    module format from quantization_config.quantized_layers (nvfp4, nvfp4_csf, nvfp4_w4a16, mxfp8) for its
          weight/scale tensors, else from dtype (bf16, fp32, ...)
  tp      tools/tp_plan.py tp_rule(row) if that module exists (row = dict of the other columns as TSV strings), else '-'
Runs built-in consistency checks and writes a size summary next to the TSV (<out>.summary.txt).
"""
import argparse
import importlib.util
import json
import os
import re
import struct
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlog  # noqa: E402

COLS = "name file offset nbytes dtype shape layer block part expert qfmt tp".split()
ITEM = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "U16": 2, "I16": 2, "F16": 2,
        "BF16": 2, "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}
DTF = {"BF16": "bf16", "F16": "fp16", "F32": "fp32", "F64": "fp64", "F8_E4M3": "fp8_e4m3", "F8_E5M2": "fp8_e5m2",
       "U8": "u8", "I8": "i8", "U32": "u32", "I32": "int32", "I64": "int64", "BOOL": "bool"}
RE_QLEAF = re.compile(r"\.(weight|weight_scale|weight_scale_2|input_scale|weight_scale\.nvfp4_csf_\w+)$")
GIB = 1 << 30

# (regex on the name inside a layer, block, expert-group) — first match wins
LAYER_RULES = [
    (r"attn_hyper_connection\.(.+)", "attn_hc"),
    (r"mlp_hyper_connection\.(.+)", "mlp_hc"),
    (r"linear_attn\.(.+)", "gdn"),
    (r"self_attn\.indexer\.(.+)", "qsa_indexer"),
    (r"self_attn\.(.+)", "qsa"),
    (r"mlp\.experts\.(\d+)\.(.+)", "expert"),
    (r"mlp\.shared_expert\.(.+)", "shared_expert"),
    (r"mlp\.(shared_expert_gate\..+)", "shared_expert"),
    (r"mlp\.(gate\..+)", "router"),
    (r"ple\.ple_embedding\.(.+)", "ngram"),
    (r"ple\.(.+)", "ple"),
]
LAYER_RULES = [(re.compile(p), b) for p, b in LAYER_RULES]
GLOBAL_RULES = [(re.compile(p), b, L) for p, b, L in [
    (r"model\.language_model\.embed_tokens\.(.+)", "embed", "-1"),
    (r"model\.language_model\.hyper_connection_mixer\.(.+)", "hc_mixer", "-1"),
    (r"lm_head\.(.+)", "lm_head", "-1"),
    (r"mtp\.hyper_connection_mixer\.(.+)", "hc_mixer", "mtp"),
    (r"mtp\.(.+)", "mtp", "mtp"),
    (r"model\.visual\.(.+)", "visual", "v"),
]]
RE_LAYER = re.compile(r"(?:model\.language_model\.layers\.(\d+)|mtp\.layers\.(\d+)|model\.visual\.blocks\.(\d+))\.(.+)")


def classify(name):
    """-> (layer, block, part, expert)"""
    m = RE_LAYER.fullmatch(name)
    if m:
        dec, mtp, vis, rest = m.groups()
        if vis is not None:
            return f"v{vis}", "visual", rest, "-1"
        layer = dec if dec is not None else f"mtp{mtp}"
        for rx, block in LAYER_RULES:
            r = rx.fullmatch(rest)
            if r:
                return (layer, block, r.group(2), r.group(1)) if block == "expert" else (layer, block, r.group(1), "-1")
        return layer, "other", rest, "-1"
    for rx, block, layer in GLOBAL_RULES:
        r = rx.fullmatch(name)
        if r:
            return layer, block, r.group(1), "-1"
    return "-1", "other", name, "-1"


def qfmt_of(name, dtype, qmods):
    """Module format for weight/scale tensors of a quantized module, else dtype-derived."""
    parts = name.split(".")
    for k in range(len(parts) - 1, 0, -1):
        mod = ".".join(parts[:k])
        if mod in qmods:
            if RE_QLEAF.search(name):  # weight / scales of the module (biases etc. fall through to dtype)
                return qmods[mod], True
            break
    return DTF.get(dtype, dtype.lower()), False


def qmods_from(cfg):
    out = {}
    for mod, q in cfg.get("quantization_config", {}).get("quantized_layers", {}).items():
        a = q.get("quant_algo", "?").upper()
        f = {"NVFP4": "nvfp4", "W4A16_NVFP4": "nvfp4_w4a16", "MXFP8": "mxfp8"}.get(a, a.lower())
        enc = q.get("weight_scale_encoding")
        out[mod] = f + (f"_{enc}" if enc else "")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", help="checkpoint dir (model.safetensors.index.json, config.json)")
    ap.add_argument("--out", default="results/tensor_table.tsv", help="TSV path; summary goes to <out>.summary.txt")
    ap.add_argument("--expect-count", type=int, default=375487)
    ap.add_argument("--expect-total", type=int, default=102138950916)
    ap.add_argument("--gdn-layer", type=int, default=0)
    ap.add_argument("--qsa-layer", type=int, default=3)
    ap.add_argument("--expert-layer", type=int, default=0)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    summ_path = re.sub(r"\.tsv$", "", a.out) + ".summary.txt"
    summ = []  # lines for the summary file

    def say(fn, tag, msg):  # screen: warnings/failures only; everything goes to the summary file
        summ.append(f"[{tag}] {msg}")
        if fn in (qlog.warn, qlog.fail):
            fn(tag, msg)

    idx = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))
    wmap, total_meta = idx["weight_map"], int(idx.get("metadata", {}).get("total_size", -1))
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    qmods = qmods_from(cfg)
    spec = importlib.util.find_spec("tp_plan")
    tp_rule = importlib.import_module("tp_plan").tp_rule if spec else None
    summ.append(f"[table] {a.ckpt} · {len(wmap)} index entries · {len(qmods)} quantized modules · "
                f"tp_plan {'found' if tp_rule else 'absent (tp = -)'}")

    # ---- read shard headers ----
    files = sorted(set(wmap.values()))
    hdrs, fsize, dbase = {}, {}, {}
    for f in files:
        p = os.path.join(a.ckpt, f)
        if not os.path.isfile(p):
            qlog.fail("table", f"missing shard {f}")
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        hdrs[f], fsize[f], dbase[f] = h, os.path.getsize(p), 8 + n
    shard_dir = os.path.join(a.ckpt, "tensors")
    extra_files = sorted(set(os.path.join("tensors", x) for x in os.listdir(shard_dir) if x.endswith(".safetensors"))
                         - set(files)) if os.path.isdir(shard_dir) else []

    # ---- rows ----
    # A tensor the index maps to another file is overridden there (e.g. a sidecar file replacing some tensors while
    # the original shards are kept as they are): no row, but its bytes still count for the overlap / cover checks.
    rows, nfail, unq, shadow = [], [], [], []
    for f in files:
        for name, t in hdrs[f].items():
            a0, a1 = t["data_offsets"]
            if name in wmap and wmap[name] != f:
                shadow.append(dict(name=name, file=f, _a0=a0, _a1=a1))
                continue
            layer, block, part, expert = classify(name)
            qf, quant = qfmt_of(name, t["dtype"], qmods)
            if not quant and t["dtype"] in ("U8", "F8_E4M3", "U32", "F8_E8M0"):
                unq.append(name)
            r = dict(name=name, file=f, offset=dbase[f] + a0, nbytes=a1 - a0, dtype=t["dtype"],
                     shape="x".join(map(str, t["shape"])) or "-", layer=layer, block=block, part=part,
                     expert=expert, qfmt=qf, tp="-", _a0=a0, _a1=a1, _shape=t["shape"])
            if tp_rule:
                r["tp"] = str(tp_rule({c: str(r[c]) for c in COLS[:-1]}))
            rows.append(r)
    rows.sort(key=lambda r: (r["file"], r["offset"]))

    # ---- checks ----
    bad = []

    nchk = [0]

    def chk(tag, cond, good, fail):
        nchk[0] += 1
        if cond:
            say(qlog.ok, tag, good)
        else:
            bad.append(tag)
            say(qlog.warn, tag, fail)

    miss = [n for n, f in wmap.items() if n not in hdrs[f]]
    extra = sum(1 for f in files for n in hdrs[f] if n not in wmap)
    chk("index", not miss and not extra and not extra_files,
        f"{len(wmap)}/{len(wmap)} index entries found in their shard header · no extra header tensors or shards"
        + (f" · {len(shadow)} tensors overridden by another file ({', '.join(sorted({wmap[x['name']] for x in shadow}))})"
           if shadow else ""),
        f"{len(miss)} index entries missing from their shard (e.g. {miss[:1]}) · {extra} header tensors not in index"
        f" · unreferenced shards {extra_files}")
    ovl, gaps, gap_b, tail = 0, [], 0, []
    by_file = defaultdict(list)
    for r in rows + shadow:
        by_file[r["file"]].append(r)
    for f, rs in by_file.items():
        rs.sort(key=lambda r: r["_a0"])
        end = 0
        for r in rs:  # sorted by offset
            if r["_a0"] < end:
                ovl += 1
            elif r["_a0"] > end:
                gaps.append(f"{f} gap [{end}, {r['_a0']}) {r['_a0'] - end} B before {r['name']}")
                gap_b += r["_a0"] - end
            end = max(end, r["_a1"])
        if dbase[f] + end != fsize[f]:
            tail.append(f"{f} data end {dbase[f] + end} != file size {fsize[f]}")
    chk("overlap", ovl == 0, f"no overlapping byte ranges in {len(files)} files", f"{ovl} overlapping tensors")
    for r in rows:
        n = ITEM.get(r["dtype"], 0)
        for d in r["_shape"]:
            n *= d
        if n != r["nbytes"]:
            nfail.append(r["name"])
    chk("nbytes", not nfail, "nbytes == prod(shape)×itemsize for all tensors",
        f"{len(nfail)} tensors with nbytes != prod(shape)×itemsize (e.g. {nfail[:1]})")
    chk("cover", not gaps and not tail, "data regions fully covered, files end at last tensor (gaps 0)",
        f"{len(gaps)} gaps ({gap_b} B) · {len(tail)} files with tail"
        f" bytes — details in {summ_path}")
    tot = sum(r["nbytes"] for r in rows)
    chk("total", tot == total_meta == a.expect_total, f"total {tot} B == index total_size == {a.expect_total}",
        f"total {tot} B · index total_size {total_meta} · expected {a.expect_total}")
    chk("count", len(rows) == len(wmap) == a.expect_count, f"tensor count {len(rows)} == index == {a.expect_count}",
        f"tensor count {len(rows)} · index {len(wmap)} · expected {a.expect_count}")
    if unq:
        say(qlog.warn, "qfmt", f"{len(unq)} packed-dtype tensors outside quantized_layers (e.g. {unq[0]})")
    prefixes = set()
    for r in rows:
        p = r["name"].split(".")
        prefixes.update(".".join(p[:k]) for k in range(1, len(p)))
    unused = [m for m in qmods if m not in prefixes]
    if unused:  # config quirk, not a checkpoint defect
        say(qlog.warn, "qmods", f"{len(unused)} quantized_layers entries without tensors: {' '.join(unused[:3])}")
    else:
        say(qlog.ok, "qmods", f"all {len(qmods)} quantized_layers modules have tensors")

    # ---- write TSV ----
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fo:
        fo.write("\t".join(COLS) + "\n")
        for r in rows:
            fo.write("\t".join(str(r[c]) for c in COLS) + "\n")
    say(qlog.ok, "table", f"{len(rows)} rows → {a.out}")

    # ---- size summary ----
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        k = (f"mtp/{r['block']}" if r["layer"].startswith("mtp") and r["block"] != "mtp" else r["block"], r["qfmt"])
        agg[k][0] += 1
        agg[k][1] += r["nbytes"]
    summ.append("")
    summ.append(f"{'block':<18}{'qfmt':<14}{'count':>9}{'GiB':>10}")
    for (b, q), (n, s) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        summ.append(f"{b:<18}{q:<14}{n:>9}{s / GIB:>10.3f}")
    summ.append(f"{'total':<32}{len(rows):>9}{tot / GIB:>10.3f}")
    blk = defaultdict(lambda: [0, 0, defaultdict(float)])
    for (b, q), (n, s) in agg.items():
        blk[b][0] += n
        blk[b][1] += s
        blk[b][2][q] += s
    for b, (n, s, qs) in sorted(blk.items(), key=lambda kv: -kv[1][1]):
        mix = ", ".join(f"{q} {v / GIB:.3f}" for q, v in sorted(qs.items(), key=lambda kv: -kv[1]))
        summ.append(f"[size] {b:<18}{n:>8} t {s / GIB:>9.3f} GiB  ({mix})")
    top5 = " · ".join(f"{b} {s / GIB:.1f}" for b, (n, s, _) in sorted(blk.items(), key=lambda kv: -kv[1][1])[:5])

    def layer_bytes(L, tag, kind):
        sel = [r for r in rows if r["layer"] == str(L) and r["block"] != "expert"]
        per = defaultdict(int)
        for r in sel:
            per[r["block"]] += r["nbytes"]
        s = sum(per.values())
        line = f"{kind} layer {L} non-expert: {s / 2**20:.2f} MiB in {len(sel)} t (" + \
               ", ".join(f"{b} {v / 2**20:.2f}" for b, v in sorted(per.items(), key=lambda kv: -kv[1])) + ")"
        say(qlog.info, tag, line)

    lt = cfg.get("text_config", {}).get("layer_types", [])
    if lt and (lt[a.gdn_layer] != "linear_attention" or lt[a.qsa_layer] != "full_attention"):
        say(qlog.warn, "layer", f"layer_types: {a.gdn_layer}={lt[a.gdn_layer]} {a.qsa_layer}={lt[a.qsa_layer]}")
    summ.append("")
    layer_bytes(a.gdn_layer, "layer", "GDN")
    layer_bytes(a.qsa_layer, "layer", "QSA")
    ex = [r for r in rows if r["layer"] == str(a.expert_layer) and r["block"] == "expert"]
    per = defaultdict(int)
    for r in ex:
        per[re.sub(r"^(gate|up|down)_proj\.", "", r["part"])] += r["nbytes"]
    ne = len({r["expert"] for r in ex})
    s = sum(per.values())
    say(qlog.info, "layer", f"routed experts layer {a.expert_layer}: {s / 2**20:.2f} MiB, {ne} experts × "
                            f"{s / max(ne, 1) / 1024:.1f} KiB (" + ", ".join(
                                f"{p} {v / 2**20:.2f}" for p, v in sorted(per.items(), key=lambda kv: -kv[1])) + " MiB)")
    if gaps or tail:
        summ += ["", "coverage details:"] + gaps + tail
    with open(summ_path, "w") as fo:
        fo.write(f"# tensor_table summary · {a.ckpt} · {len(files)} shard files\n" + "\n".join(summ) + "\n")
    (qlog.ok if not bad else qlog.warn)("table", f"{len(rows):,} tensors · {tot / 1e9:.2f} GB · checks {nchk[0] - len(bad)}/{nchk[0]} · "
        f"tp {'yes' if tp_rule else 'no'} · largest {top5} GiB · summary {summ_path}")
    if bad:
        qlog.fail("table", f"{len(bad)} checks failed: {' '.join(bad)}")


if __name__ == "__main__":
    main()
