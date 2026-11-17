# Changelog

One line per change. The newest heading is the engine's version (shown on the server's banner and at `/version`).

## 0.1.2 — 2026-11-17
- Logs no longer pile up: a run's saved logs (and its per-request timing lines) are removed after 7 days, and a running server's log holds at most 2 × 50 MB.

## 0.1.1 — 2026-11-16
- Faster start: weights load on several threads (long reads in 8 MiB pieces, a layer's experts in parallel) — ~100–150 s → ~40 s a rank, the same bytes on the GPU.
- Prefix cache shared between requests: the same text sent several times at once reuses one cached copy instead of cold reads.
- Cache reuse only where the MTP head's cache holds the prompt's next token: a continued turn gives the same output as a cold read.
- The worker runs the head's image (built once on the head, copied over), so both ranks run the very same image.

## 0.1.0 — 2026-11-14
- First release: two-node (TP2) C++/CUDA engine for Qwen3.8-Flash-Next with an OpenAI-compatible server.
