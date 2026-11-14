"""TP=2 placement per tensor — plan P0 = the reference's vLLM split (so per-rank kernel inputs match the golden dump).

Rules (value of the table's `tp` column):
  rep            both ranks hold the whole tensor
  col            dim 0 halved, rank r takes rows [r·N/2, (r+1)·N/2)
  row            dim 1 halved (input/K side; packed FP4 bytes and scale columns halve with it)
  seg:a,b,c      dim 0 is [a|b|c] segments, each halved (GDN qkv by heads: q 2048 | k 2048 | v 6144)
  csf:col/row    CSF scale tensors: decode to [N,K/16] E4M3 first, then split like the weight
  own:r          only rank r holds it (PLE n-gram table shards: 0–63 → rank 0, 64–127 → rank 1)
  skip           not loaded by the text engine (vision tower)
"""
import re

_SHARD = re.compile(r"ngram_embedding\.shard_(\d+)\.")


def tp_rule(row):
    b, p = row["block"], row["part"]
    if b == "visual":
        return "skip"
    if b in ("embed", "lm_head"):
        return "col"
    if b in ("attn_hc", "mlp_hc", "hc_mixer", "ple", "router", "qsa_indexer"):
        return "rep"
    if b == "ngram":
        m = _SHARD.search(row["name"])
        return f"own:{0 if int(m.group(1)) < 64 else 1}" if m else "rep"
    if b == "mtp":
        return "col" if p.startswith(("fc_embedding", "fc_hidden")) else "rep"
    if b == "gdn":
        if p.startswith(("in_proj_qkv.", "conv1d.")):
            return "seg:2048,2048,6144"
        if p.startswith(("in_proj_z.", "in_proj_a.", "in_proj_b.")) or p in ("A_log", "dt_bias"):
            return "col"
        if p.startswith("out_proj."):
            return "row"
        return "rep"
    if b == "qsa":
        if p.startswith(("q_proj.", "k_proj.", "v_proj.")):
            return "col"
        if p.startswith("o_proj."):
            return "row"
        return "rep"
    if b in ("expert", "shared_expert"):
        if p.startswith("shared_expert_gate") or p.endswith(("weight_scale_2", "input_scale")):
            return "rep"
        side = "row" if p.startswith("down_proj.") else "col"
        return f"csf:{side}" if "nvfp4_csf" in p else side
    return "rep"


def rank_bytes(row, rank=0):
    """Resident bytes of this tensor on `rank` after loading (CSF scales counted in their decoded E4M3 size)."""
    t, n = row["tp"], int(row["nbytes"])
    if t == "skip":
        return 0
    if t.startswith("own:"):
        return n if int(t[4:]) == rank else 0
    if t.startswith("csf:"):
        if row["part"].endswith("nvfp4_csf_exceptions"):
            return 0
        g, w = (int(x) for x in row["shape"].split("x"))  # fixed [N/16, 16+8C] → decoded N·C bytes
        return (16 * g) * ((w - 16) // 8) // 2
    if t == "rep":
        return n
    return n // 2
