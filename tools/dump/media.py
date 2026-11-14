"""Test media for the vision golden — drawn from code, nothing downloaded. Usage: media.py OUT_DIR

Runs in the reference image (Pillow 13, NumPy, OpenCV 5): the same versions give the same bytes, and the files' sha256
go into the dump record, so the golden and our engine always read identical files.
  chart.png  800×600 RGB   bar chart with a title number and value labels (text, flat colors, sharp edges)
  scene.png  1280×720 RGB  gradients, shapes and seeded noise (exercises resize filtering; 720 is not a multiple of 32)
  tiny.png   100×60 RGB    below the minimum pixel count (the processor scales it up)
  rgba.png   300×300 RGBA  transparent background with a shape (alpha handling)
  clip.mp4   4 s · 8 fps · 320×240 (mp4v)    a red square moving left to right, frame counter
  long.mp4   12 s · 30 fps · 480×270 (mp4v)  a ball on a circle, a clock hand, the second written in the corner
"""
import hashlib
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)


def font(size):
    return ImageFont.load_default(size=size)


def chart():
    im = Image.new("RGB", (800, 600), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((40, 24), "Q38 REPORT 4721", fill=(20, 20, 20), font=font(40))
    d.line((80, 540, 760, 540), fill=(0, 0, 0), width=3)
    d.line((80, 120, 80, 540), fill=(0, 0, 0), width=3)
    bars = [("A", 180, (220, 40, 40)), ("B", 390, (40, 160, 60)), ("C", 260, (40, 80, 220)), ("D", 120, (240, 150, 20))]
    for i, (name, v, col) in enumerate(bars):
        x = 130 + i * 160
        d.rectangle((x, 540 - v, x + 100, 540), fill=col)
        d.text((x + 30, 540 - v - 40), str(v), fill=(0, 0, 0), font=font(28))
        d.text((x + 40, 552), name, fill=(0, 0, 0), font=font(28))
    return im


def scene():
    h, w = 720, 1280
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    img = np.zeros((h, w, 3))
    img[..., 0] = 60 + 120 * y + 30 * x
    img[..., 1] = 120 + 80 * y
    img[..., 2] = 230 - 120 * y
    hill = 480 + 60 * np.sin(x * 7.0) + 30 * np.cos(x * 17.0)
    ground = np.stack(np.broadcast_arrays(40 + 60 * x, 140 + 40 * y, 50 + 0 * x), -1)
    img = np.where((np.arange(h)[:, None] > hill)[..., None], ground, img)
    img += np.random.default_rng(0).normal(0, 6, img.shape)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.ellipse((1000, 80, 1140, 220), fill=(255, 220, 60))
    d.rectangle((300, 380, 520, 560), fill=(170, 60, 50))
    d.polygon([(280, 380), (410, 280), (540, 380)], fill=(90, 40, 30))
    d.rectangle((390, 470, 440, 560), fill=(60, 40, 20))
    d.text((40, 640), "HOUSE 58", fill=(255, 255, 255), font=font(36))
    return im


def tiny():
    im = Image.new("RGB", (100, 60), (250, 250, 210))
    ImageDraw.Draw(im).text((38, 8), "7", fill=(200, 0, 0), font=font(40))
    return im


def rgba():
    im = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((40, 40, 260, 260), fill=(30, 90, 220, 255))
    d.rectangle((120, 120, 180, 180), fill=(255, 255, 255, 128))
    return im


def video(path, fps, n, w, h, draw):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        sys.exit(f"cannot open a video writer for {path}")
    for i in range(n):
        im = Image.new("RGB", (w, h), (128, 128, 128))
        draw(ImageDraw.Draw(im), i)
        vw.write(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
    vw.release()


def clip(d, i):
    x = 10 + i * 9
    d.rectangle((x, 90, x + 60, 150), fill=(230, 30, 30))
    d.text((8, 8), f"frame {i}", fill=(255, 255, 255), font=font(20))


def long(d, i):
    t = i / 30.0
    cx, cy = 240 + 90 * np.cos(t), 135 + 90 * np.sin(t)
    d.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=(40, 200, 40))
    a = t * 2 * np.pi / 12
    d.line((240, 135, 240 + 60 * np.sin(a), 135 - 60 * np.cos(a)), fill=(0, 0, 0), width=4)
    d.text((8, 8), f"t={int(t)}s", fill=(255, 255, 255), font=font(22))


for name, make in (("chart.png", chart), ("scene.png", scene), ("tiny.png", tiny), ("rgba.png", rgba)):
    make().save(os.path.join(OUT, name))
video(os.path.join(OUT, "clip.mp4"), 8, 32, 320, 240, clip)
video(os.path.join(OUT, "long.mp4"), 30, 360, 480, 270, long)
sums = {f: hashlib.sha256(open(os.path.join(OUT, f), "rb").read()).hexdigest()
        for f in sorted(os.listdir(OUT)) if f != "sha256.json"}
json.dump(sums, open(os.path.join(OUT, "sha256.json"), "w"), indent=1)
print(json.dumps({k: v[:12] for k, v in sums.items()}))
