"""vision.py — image / video front end of Qwen3.8-Flash-Next (CPU only, no model weights).

Reproduces the reference (vLLM 0.32 + transformers 5.19 in the reference image) for one media item:
  bytes → RGB frames (PIL for images, OpenCV for videos, as vLLM's loaders) → smart resize (torchvision bicubic with
  antialias on uint8, as transformers' TorchvisionBackend) → uint8 patches [L, 1536] in 2×2-merge order, grid (t, h, w)
  in 16-px patches. vLLM runs the processor with do_rescale / do_normalize off (mm_device_do_normalize): the tower gets
  raw uint8 and normalises on the device (vision_tower.Tower).
and the prompt side: expand() turns each template placeholder into the reference's token run (image: N image_pad;
video: per temporal group "<t seconds>" + vision_start + N video_pad + vision_end) and builds the 3-row mRoPE
positions (vLLM Qwen3VLForConditionalGeneration.get_mrope_input_positions).

Checked against the golden dump by server/check_vision.py (pixels bit-exact, ids / positions / rows exact).
"""
import hashlib
import io
import math
from dataclasses import dataclass, field

import numpy as np
import torch

PATCH, MERGE, TEMPORAL = 16, 2, 2
FACTOR = PATCH * MERGE  # 32: resized sides are multiples of this
PATCH_DIM = 3 * TEMPORAL * PATCH * PATCH  # 1536
VISION_START, VISION_END, IMAGE_PAD, VIDEO_PAD = 248053, 248054, 248056, 248057

# reference defaults (preprocessor_config.json / video_preprocessor_config.json, vLLM envs / video loader)
IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS = 65536, 16777216
VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS = 4096, 25165824  # whole clip (t × h × w) budget
VIDEO_FPS, VIDEO_MIN_FRAMES, VIDEO_MAX_FRAMES, VIDEO_MAX_FPS = 2.0, 4, 768, 30.0
MAX_INPUT_PIXELS = 178_956_970  # VLLM_MAX_IMAGE_PIXELS: decoded image / video frame size before resizing

# Pillow formats vLLM accepts for untrusted media (vllm/multimodal/image.py)
ALLOWED_IMAGE_FORMATS = ("AVIF", "BLP", "BMP", "CUR", "DCX", "DDS", "DIB", "FITS", "FLI", "FTEX", "GBR", "GIF", "ICNS",
                         "ICO", "IM", "IMT", "IPTC", "JPEG", "JPEG2000", "MCIDAS", "MSP", "PCD", "PCX", "PIXAR", "PNG",
                         "PPM", "PSD", "QOI", "SGI", "SPIDER", "SUN", "TGA", "TIFF", "WEBP", "XBM", "XPM", "XVTHUMB")


@dataclass
class Media:
    kind: str                  # 'image' | 'video'
    pixels: torch.Tensor       # uint8 [L, 1536] on CPU, L = t·h·w, rows in merge order per temporal group
    grid: tuple                # (t, h, w) in patches (t = temporal groups of 2 frames; image: 1)
    timestamps: list = field(default_factory=list)  # video: per temporal group the text "<1.2 seconds>"
    sha256: str = ""           # of the input bytes
    frames: list = field(default_factory=list)      # video: source frame indices used (before padding to even)
    fps: float = 0.0           # video: source fps used for the timestamps (0 in the file → 24)

    @property
    def tokens(self):  # embedding rows of this item (merged 2×2)
        t, h, w = self.grid
        return t * (h // MERGE) * (w // MERGE)

    @property
    def group_tokens(self):  # embedding rows per temporal group
        _, h, w = self.grid
        return (h // MERGE) * (w // MERGE)


# ---- resize (transformers smart_resize + TorchvisionBackend.resize) ----------------------------------------------

def smart_resize_image(height, width, factor=FACTOR, min_pixels=IMAGE_MIN_PIXELS, max_pixels=IMAGE_MAX_PIXELS):
    """transformers qwen2_vl smart_resize (Python round = half-even, floor / ceil as there)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"image aspect ratio {max(height, width) / min(height, width):.0f} is over 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def smart_resize_video(num_frames, height, width, temporal_factor=TEMPORAL, factor=FACTOR,
                       min_pixels=VIDEO_MIN_PIXELS, max_pixels=VIDEO_MAX_PIXELS):
    """transformers qwen3_vl video smart_resize (one size for all frames; the budget counts the frames)."""
    if num_frames < temporal_factor:
        raise ValueError(f"video needs at least {temporal_factor} decodable frames, got {num_frames}")
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height, width = int(height * scale), int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"video aspect ratio {max(height, width) / min(height, width):.0f} is over 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _resize(frames, height, width):
    """uint8 [N, 3, H, W] → uint8 [N, 3, height, width]: tvF.resize(bicubic, antialias=True) on the CPU, as
    transformers' TorchvisionBackend.resize (torchvision resamples in float32, then clamps and rounds to uint8)."""
    from torchvision.transforms.v2 import functional as tvF
    return tvF.resize(frames, [height, width], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)


def patchify(frames):
    """uint8 [T, 3, H, W] (H, W multiples of 32) → uint8 [ceil(T/2)·(H/16)·(W/16), 1536].

    An odd T is padded with the last frame (an image is one frame, so it is doubled). Rows: per temporal group,
    block-row, block-col, then the 2×2 window (mh, mw); a row is ((c·2 + t)·16 + py)·16 + px."""
    t, c, h, w = frames.shape
    if t % TEMPORAL:
        frames = torch.cat([frames, frames[-1:].expand(TEMPORAL - t % TEMPORAL, -1, -1, -1)])
        t = frames.shape[0]
    gt, gh, gw = t // TEMPORAL, h // PATCH, w // PATCH
    x = frames.reshape(gt, TEMPORAL, c, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)  # gt, bh, bw, mh, mw, c, t, py, px
    return x.reshape(gt * gh * gw, c * TEMPORAL * PATCH * PATCH).contiguous(), (gt, gh, gw)


# ---- images --------------------------------------------------------------------------------------------------------

def _to_rgb(img):
    """vLLM ImageMediaIO._convert_image_mode(RGB): RGBA (and LA / PA / a tRNS chunk via RGBA) pasted on white with
    its alpha as the mask; any other mode → convert('RGB')."""
    from PIL import Image
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in getattr(img, "info", {}):
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        out = Image.new("RGB", img.size, (255, 255, 255))
        out.paste(img, mask=img.split()[3])
        return out
    return img.convert("RGB")


def decode_image(data, max_input_pixels=MAX_INPUT_PIXELS):
    """bytes → RGB PIL image as vLLM's ImageMediaIO.load_bytes (format allowlist, EXIF orientation, alpha on white)."""
    from PIL import Image, ImageOps
    try:
        img = Image.open(io.BytesIO(data), formats=ALLOWED_IMAGE_FORMATS)
        w, h = img.size
    except Exception as e:  # noqa: BLE001 — untrusted bytes: any decoder failure is a bad input
        raise ValueError(f"cannot decode the image ({type(e).__name__})") from None
    if max_input_pixels and w * h > max_input_pixels:
        raise ValueError(f"image {w}x{h} is over the limit of {max_input_pixels} pixels")
    try:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:  # noqa: BLE001 — vLLM ignores a broken EXIF block the same way
            pass
        img.load()
        return _to_rgb(img)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"cannot decode the image ({type(e).__name__})") from None


def load_image(data, *, min_pixels=IMAGE_MIN_PIXELS, max_pixels=IMAGE_MAX_PIXELS, max_input_pixels=MAX_INPUT_PIXELS):
    """Image bytes → Media (grid (1, h, w), pixels uint8 [h·w, 1536]). ValueError on bad / unsupported input."""
    if not data:
        raise ValueError("empty image")
    from torchvision.transforms.v2 import functional as tvF
    img = decode_image(data, max_input_pixels)
    x = tvF.pil_to_tensor(img)[None]  # uint8 [1, 3, H, W]
    height, width = x.shape[-2:]
    rh, rw = smart_resize_image(height, width, FACTOR, min_pixels, max_pixels)
    x = _resize(x, rh, rw)
    pixels, grid = patchify(x)
    return Media("image", pixels, grid, sha256=hashlib.sha256(data).hexdigest())


# ---- videos --------------------------------------------------------------------------------------------------------

def _cv2_stream_api(vr):
    """The first stream-buffered OpenCV backend with a usable plugin API (vLLM OpenCVVideoBackendMixin)."""
    for b in vr.getStreamBufferedBackends():
        if not vr.hasBackend(b):
            continue
        if not vr.isBackendBuiltIn(b):
            _, abi, api = vr.getStreamBufferedBackendPluginVersion(b)
            if abi < 1 or (abi == 1 and api < 2):
                continue
        return b
    return None


def decode_video(data, *, fps=VIDEO_FPS, min_frames=VIDEO_MIN_FRAMES, max_frames=VIDEO_MAX_FRAMES,
                 max_input_pixels=MAX_INPUT_PIXELS):
    """bytes → (RGB frames uint8 [T, H, W, 3], source frame indices, source fps): vLLM Qwen3VLVideoBackend with the
    OpenCV decoder — frame count from the container, corrected by an end seek; n = min(max(int(total / src_fps ·
    fps), min_frames), max_frames, total) frames at linspace(0, total-1, n).round(); frames that do not decode are
    dropped."""
    import cv2
    import cv2.videoio_registry as vr
    stream = io.BytesIO(data)  # OpenCV reads it through the capture's lifetime: keep it referenced until release()
    try:
        cap = cv2.VideoCapture(stream, _cv2_stream_api(vr), [])
    except cv2.error:
        raise ValueError("cannot open the video") from None
    try:
        if not cap.isOpened():
            raise ValueError("cannot open the video")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max_input_pixels and w * h > max_input_pixels:
            raise ValueError(f"video frame {w}x{h} is over the limit of {max_input_pixels} pixels")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if total > 0 and cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0):  # edit lists: the presentable frame count
            visible = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            at_end = not cap.grab()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if at_end and 0 < visible < total - 1:
                total = visible
        if total <= 0:
            raise ValueError("the video has no frames")
        if not src_fps or not math.isfinite(src_fps) or src_fps <= 0:
            src_fps = 24.0  # HF Qwen3VLVideoProcessor's assumption for an unknown rate
        n = int(total / src_fps * min(fps, VIDEO_MAX_FPS))
        n = min(max(n, min_frames), max_frames, total)
        want = np.linspace(0, total - 1, n).round().astype(int).tolist()
        want_set, frames, got = set(want), [], []
        for idx in range(max(want) + 1):
            if not cap.grab():
                continue
            if idx in want_set:
                ok, frame = cap.retrieve()
                if ok and frame is not None and frame.size > 0:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    got.append(idx)
    except cv2.error:
        raise ValueError("cannot decode the video") from None
    finally:
        cap.release()
    if not frames:
        raise ValueError("no decodable frames in the video")
    if len({f.shape for f in frames}) != 1:
        raise ValueError("video frames change size")
    return np.stack(frames), got, float(src_fps)


def timestamps(indices, src_fps, merge=TEMPORAL):
    """Per temporal group the mean time of its frames (vLLM _calculate_timestamps; indices padded to a multiple of
    `merge` with the last one), as the prompt text "<1.2 seconds>"."""
    idx = list(indices)
    if len(idx) % merge:
        idx += [idx[-1]] * (merge - len(idx) % merge)
    ts = [i / src_fps for i in idx]
    return [f"<{(ts[i] + ts[i + merge - 1]) / 2:.1f} seconds>" for i in range(0, len(ts), merge)]


def load_video(data, *, fps=VIDEO_FPS, min_frames=VIDEO_MIN_FRAMES, max_frames=VIDEO_MAX_FRAMES,
               min_pixels=VIDEO_MIN_PIXELS, max_pixels=VIDEO_MAX_PIXELS, max_input_pixels=MAX_INPUT_PIXELS):
    """Video bytes → Media (grid (ceil(T/2), h, w), pixels uint8, per-group timestamps). ValueError on bad input."""
    if not data:
        raise ValueError("empty video")
    frames, got, src_fps = decode_video(data, fps=fps, min_frames=min_frames, max_frames=max_frames,
                                        max_input_pixels=max_input_pixels)
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()  # uint8 [T, 3, H, W]
    height, width = x.shape[-2:]
    rh, rw = smart_resize_video(x.shape[0], height, width, TEMPORAL, FACTOR, min_pixels, max_pixels)
    x = _resize(x, rh, rw)
    pixels, grid = patchify(x)
    ts = timestamps(got, src_fps)
    assert len(ts) == grid[0]
    return Media("video", pixels, grid, ts, hashlib.sha256(data).hexdigest(), got, src_fps)


# ---- prompt expansion and mRoPE positions --------------------------------------------------------------------------

def _encode(tokenizer, text):
    enc = tokenizer.encode(text, add_special_tokens=False)
    return list(enc.ids if hasattr(enc, "ids") else enc)


def expand(ids, media, tokenizer):
    """Template ids → (ids, pos3, delta, rows).

    ids: the rendered prompt, each media item as the template renders it (vision_start, image_pad | video_pad,
    vision_end), items in order. Every image_pad / video_pad token is one item of `media` (same order and kind).
    Returns the expanded ids (vLLM's prompt), pos3 = flat [n·3] mRoPE positions (t, h, w per token), delta =
    max position + 1 − n (positions after the prompt are p + delta on all three rows), rows = positions of the
    image_pad / video_pad tokens in embedding-row order (item order, merge order inside an item)."""
    out, blocks, k = [], [], 0  # blocks: (offset of the first pad, merged h, merged w)
    for tok in ids:
        if tok not in (IMAGE_PAD, VIDEO_PAD):
            out.append(tok)
            continue
        if k >= len(media):
            raise ValueError("more media placeholders in the prompt than media items")
        m = media[k]
        k += 1
        want = IMAGE_PAD if m.kind == "image" else VIDEO_PAD
        if tok != want:
            raise ValueError(f"media item {k} is a {m.kind} but the prompt has a different placeholder")
        if not out or out[-1] != VISION_START:
            raise ValueError("media placeholder without vision_start before it")
        _, gh, gw = m.grid
        mh, mw = gh // MERGE, gw // MERGE
        if m.kind == "image":
            blocks.append((len(out), mh, mw))
            out += [IMAGE_PAD] * (mh * mw)
        else:
            if len(m.timestamps) != m.grid[0]:
                raise ValueError("video timestamps do not match its temporal groups")
            for ts in m.timestamps:  # "<t seconds>" vision_start pads vision_end, all inside the template's pair
                out += _encode(tokenizer, ts)
                out.append(VISION_START)
                blocks.append((len(out), mh, mw))
                out += [VIDEO_PAD] * (mh * mw)
                out.append(VISION_END)
    if k != len(media):
        raise ValueError("fewer media placeholders in the prompt than media items")

    n = len(out)
    pos = np.empty((n, 3), dtype=np.int64)
    st, nxt = 0, 0  # next token offset / next text position
    for off, mh, mw in blocks:
        tl = off - st
        pos[st:off] = (nxt + np.arange(tl))[:, None]
        s = nxt + tl
        r, c = np.divmod(np.arange(mh * mw), mw)
        pos[off:off + mh * mw, 0] = s
        pos[off:off + mh * mw, 1] = s + r
        pos[off:off + mh * mw, 2] = s + c
        nxt = s + max(mh, mw)
        st = off + mh * mw
    pos[st:] = (nxt + np.arange(n - st))[:, None]
    delta = int(pos.max()) + 1 - n if n else 0
    arr = np.asarray(out)
    rows = np.nonzero((arr == IMAGE_PAD) | (arr == VIDEO_PAD))[0].tolist()
    return out, pos.reshape(-1).tolist(), delta, rows
