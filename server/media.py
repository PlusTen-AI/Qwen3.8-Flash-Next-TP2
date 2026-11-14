"""media.py — images and videos of chat requests on the head (rank 0), between the template and the scheduler.

MediaFront.prepare(items, ids) for the (kind, url) list chat.normalize_messages collected and the prompt's token ids
(each item as the template's single placeholder): fetch (data: URLs; http(s):// only with vision.fetch_urls), decode
and preprocess (vision.py — the reference's pixels), the request's limits, the vision tower (vision_tower.py), then the
prompt expansion and mRoPE positions (vision.expand). Returns the expanded ids, the prefix-reuse key (each media token
replaced by a value from its file's hash and index: the token ids are the same for every picture) and the scheduler's
media tuple (pos3, delta, rows, embeddings). Bad input → ValueError (400).

At most vision.max_concurrent requests are decoded and encoded at once (the others wait their turn); each item runs
through the tower alone (an item's embeddings do not depend on the others in the request). The tower is split over
both ranks (vision_tower.TP) and runs in the engine's call order: the worker hands an item to TowerJobs and waits; the
scheduler thread runs it in slices between engine calls through the command link (rank 1 replays them), so the tower's
collectives never overlap the engine's. Its peak working memory is taken at start (Tower.profile) before the KV pool. Embeddings are kept by file (sha256 of the bytes + kind) up to vision.cache_mb, least
recently used dropped first: a picture sent again (every turn of a conversation resends it) skips decoding and the tower.
"""
import asyncio
import base64
import binascii
import collections
import dataclasses
import hashlib
import threading
import time
import types
import urllib.request

import torch

import vision
from vision_tower import Tower


class TowerJobs:
    """Items waiting for the vision tower (rank 0). A media worker thread hands one in and waits (run); the scheduler
    thread takes the head, runs it in slices (Scheduler._tower: tower_begin / tower_blocks / tower_end on both ranks)
    and hands the embeddings back (finish)."""
    def __init__(self, depth):
        self.depth, self.q, self.lock, self.n = depth, collections.deque(), threading.Lock(), 0
        self.wake = lambda: None  # the scheduler's: wake its loop

    def run(self, m):
        with self.lock:
            self.n += 1
            j = types.SimpleNamespace(id=self.n, grids=[tuple(m.grid)], shape=tuple(m.pixels.shape), px=m.pixels,
                                      L=m.pixels.shape[0], k=-1, n=1, out=None, err=None, done=threading.Event())
            self.q.append(j)
        self.wake()
        j.done.wait()
        if j.err:
            raise RuntimeError(j.err)
        return j.out

    def head(self):
        with self.lock:
            return self.q[0] if self.q else None

    def finish(self, jid, out=None, err=None):
        with self.lock:
            j = self.q.popleft()
        assert j.id == jid, (j.id, jid)
        j.px, j.out, j.err = None, out, err
        j.done.set()

    def fail_all(self, msg):
        while self.head() is not None:
            self.finish(self.head().id, err=msg)


class MediaFront:
    def __init__(self, cfg, model_dir, tokenizer, tp=None):
        self.cfg, self.tok = cfg, tokenizer
        t0 = time.time()
        self.tower = Tower(model_dir, "cuda", tp=tp)
        self.jobs = TowerJobs(self.tower.depth)
        self.load_s = time.time() - t0
        # held beside the tower's working memory (rank 0): the embeddings cache and one request's embeddings (pieces +
        # their cat) — taken at start with the tower's peak, so the memory stays flat afterwards
        hold = float(cfg["cache_mb"]) * 2**20 + 2 * int(cfg["max_media_tokens"]) * self.tower.out_hidden * 2
        self.peak_gib, self.profile_s = self.tower.profile(hold_bytes=hold)
        self.sem = asyncio.Semaphore(max(1, int(cfg["max_concurrent"])))
        self.cache, self.cache_bytes = collections.OrderedDict(), 0  # (kind, sha256) → (Media without pixels, emb)
        self.lock = threading.Lock()  # the cache (max_concurrent > 1: several worker threads)
        self.stats = {"requests": 0, "items": 0, "tokens": 0, "tower_s": 0.0, "cache_hits": 0}

    # ---- input --------------------------------------------------------------------------------------------------
    def fetch(self, url):
        cap = int(float(self.cfg["max_item_mb"]) * 2**20)
        if url.startswith("data:"):
            head, sep, data = url.partition(",")
            if not sep or ";base64" not in head:
                raise ValueError("a media data: URL must be base64 (data:<type>;base64,...)")
            if len(data) > cap * 4 // 3 + 4:
                raise ValueError(f"a media file is larger than {self.cfg['max_item_mb']} MB")
            try:
                raw = base64.b64decode(data, validate=False)
            except (binascii.Error, ValueError):
                raise ValueError("invalid base64 in a media data: URL") from None
        elif url.startswith(("http://", "https://")):
            if not self.cfg["fetch_urls"]:
                raise ValueError("this server does not fetch media URLs (vision.fetch_urls); send the file as a "
                                 "data: URL")
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    raw = r.read(cap + 1)
            except OSError as e:
                raise ValueError(f"could not fetch {url[:100]}: {e}") from None
        else:
            raise ValueError("media URLs must be data: URLs" + ("" if not self.cfg["fetch_urls"] else " or http(s)://"))
        if len(raw) > cap:
            raise ValueError(f"a media file is larger than {self.cfg['max_item_mb']} MB")
        return raw

    def _work(self, items):
        """(worker thread) bytes → Media → embeddings [Σ tokens, 2560] (bf16, CUDA, finished)."""
        ms, embs, todo = [], [], []
        for kind, url in items:
            raw = self.fetch(url)
            key = (kind, hashlib.sha256(raw).hexdigest())
            with self.lock:
                hit = self.cache.get(key)
                if hit is not None:
                    self.cache.move_to_end(key)
                    self.stats["cache_hits"] += 1
            if hit is not None:
                ms.append(hit[0])
                embs.append(hit[1])
                continue
            ms.append(vision.load_image(raw) if kind == "image" else vision.load_video(raw))
            embs.append(None)
            todo.append((len(ms) - 1, key))
        n = sum(m.tokens for m in ms)
        if n > int(self.cfg["max_media_tokens"]):
            raise ValueError(f"the images and videos of this request take {n:,} tokens; this server allows "
                             f"{int(self.cfg['max_media_tokens']):,} (vision.max_media_tokens)")
        t0 = time.time()
        for i, key in todo:
            embs[i] = self.jobs.run(ms[i])  # through the scheduler, in the engine's call order
            ms[i] = dataclasses.replace(ms[i], pixels=None)  # what expand needs: kind, grid, timestamps
            self._keep(key, ms[i], embs[i])
        emb = torch.cat(embs) if embs else None  # a new tensor: the cache may drop its pieces later
        torch.cuda.current_stream().synchronize()  # done before the engine's stream reads it
        return ms, emb, time.time() - t0

    def _keep(self, key, m, e):
        cap = int(float(self.cfg["cache_mb"]) * 2**20)
        nb = e.numel() * e.element_size()
        if nb > cap:
            return
        with self.lock:
            if key in self.cache:
                return
            self.cache[key] = (m, e)
            self.cache_bytes += nb
            while self.cache_bytes > cap:
                _, (_, old) = self.cache.popitem(last=False)
                self.cache_bytes -= old.numel() * old.element_size()

    def _media_only(self, items):
        """(worker thread) bytes → Media (pixels not needed: from the cache when it has the file)."""
        ms = []
        for kind, url in items:
            raw = self.fetch(url)
            with self.lock:
                hit = self.cache.get((kind, hashlib.sha256(raw).hexdigest()))
            ms.append(hit[0] if hit is not None else
                      vision.load_image(raw) if kind == "image" else vision.load_video(raw))
        return ms

    async def expand_only(self, items, ids):
        """/tokenize: the expanded prompt ids, without running the tower."""
        if len(items) > int(self.cfg["max_items"]):
            raise ValueError(f"{len(items)} images / videos in one request; this server allows "
                             f"{int(self.cfg['max_items'])} (vision.max_items)")
        ms = await asyncio.to_thread(self._media_only, items)
        return vision.expand(ids, ms, self.tok)[0]

    async def prepare(self, items, ids):
        if len(items) > int(self.cfg["max_items"]):
            raise ValueError(f"{len(items)} images / videos in one request; this server allows "
                             f"{int(self.cfg['max_items'])} (vision.max_items)")
        async with self.sem:
            ms, emb, dt = await asyncio.to_thread(self._work, items)
        ids, pos3, delta, rows = vision.expand(ids, ms, self.tok)
        if emb.shape[0] != len(rows):
            raise ValueError(f"media rows {len(rows)} ≠ embeddings {emb.shape[0]}")
        key, j = list(ids), 0
        for m in ms:
            h = int(m.sha256[:15], 16)
            for i in range(m.tokens):
                key[rows[j]] = -1 - ((h + i) % (1 << 62))
                j += 1
        st = self.stats
        st["requests"] += 1
        st["items"] += len(ms)
        st["tokens"] += len(rows)
        st["tower_s"] += dt
        return ids, key, (pos3, delta, rows, emb), {"items": len(ms), "tokens": len(rows), "tower_ms": dt * 1e3}
