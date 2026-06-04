"""Export the LoRA Royale crossfade as a shareable GIF or MP4.

The morph is the product: a face resolving epoch-by-epoch, ping-pong looped,
with an epoch ticker and a 'Fizgig · LoRA Royale' tag burned in. Every clip is
an advert for the tool, and the tool makes the clip for you.

Frames are built from the same Image.blend the crossfade slider uses, so the
exported sweep is exactly what you saw. No new dependencies — PIL writes the
GIF, OpenCV (already a Fizgig dep) writes the MP4.
"""

from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


# Speed preset -> (frames_per_transition, hold_frames, fps). fps is what sets
# playback speed for the fixed-length travel clips; the epoch morph also reacts
# to frames_per_transition/hold. All three vary so the control is felt.
SPEED_PRESETS = {
    "Slow":   (24, 12, 14),
    "Normal": (16, 8, 22),
    "Fast":   (10, 4, 32),
}


def _load_font(size: int):
    for name in ("seguisb.ttf", "segoeui.ttf", "arialbd.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _pill(draw: ImageDraw.ImageDraw, xy, text, font, anchor_right=False,
          bottom=0, fg=(255, 255, 255, 235), bg=(0, 0, 0, 140), pad=(10, 5)):
    """Draw a rounded translucent pill anchored to a bottom corner."""
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    px, py = pad
    w, h = tw + px * 2, th + py * 2
    x = xy[0] - w if anchor_right else xy[0]
    y = bottom - h
    rad = max(6, h // 3)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=rad, fill=bg)
    draw.text((x + px - l, y + py - t), text, font=font, fill=fg)


def _decorate(img: Image.Image, epoch_label=None, brand=True, badge=None) -> Image.Image:
    """Burn a corner badge (bottom-left, gold) + brand pill (bottom-right).

    `badge` is raw text; `epoch_label` is a convenience that renders as
    "EPOCH <n>". Pass at most one."""
    badge_text = badge if badge is not None else (f"EPOCH {epoch_label}" if epoch_label is not None else None)
    if badge_text is None and not brand:
        return img.convert("RGB")
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    W, H = base.size
    margin = max(8, H // 40)
    fs = max(13, H // 26)
    font = _load_font(fs)
    if badge_text is not None:
        _pill(d, (margin, 0), badge_text, font,
              anchor_right=False, bottom=H - margin, fg=(255, 210, 74, 245))
    if brand:
        _pill(d, (W - margin, 0), "Fizgig · LoRA Royale", font,
              anchor_right=True, bottom=H - margin, fg=(255, 255, 255, 235))
    return Image.alpha_composite(base, overlay).convert("RGB")


def _even(n: int) -> int:
    return n if n % 2 == 0 else n - 1


def build_frames(images: List[Tuple], speed: str = "Normal", pingpong: bool = True,
                 brand: bool = True, show_epoch: bool = True,
                 max_size: Optional[int] = 768) -> List[Image.Image]:
    """images: [(label, PIL)] in epoch order. Returns decorated RGB frames.

    Holds on each epoch, then blends to the next (the slider's morph). With
    pingpong, the return sweep replays the transitions in reverse for a
    seamless loop. Epoch ticker shows the dominant epoch in each frame.
    """
    if not images:
        return []
    fpt, hold, _fps = SPEED_PRESETS.get(speed, SPEED_PRESETS["Normal"])

    # Normalise size: all frames share the first image's box (even dims for MP4),
    # optionally downscaled so clips stay shareable.
    base_w, base_h = images[0][1].size
    if max_size and max(base_w, base_h) > max_size:
        scale = max_size / float(max(base_w, base_h))
        base_w, base_h = int(base_w * scale), int(base_h * scale)
    base_w, base_h = max(2, _even(base_w)), max(2, _even(base_h))

    def prep(im):
        return im if im.size == (base_w, base_h) else im.resize((base_w, base_h), Image.LANCZOS)

    norm = [(label, prep(im)) for label, im in images]

    def decorate(im, label):
        return _decorate(im, epoch_label=(label if show_epoch else None), brand=brand)

    frames: List[Image.Image] = []
    if len(norm) == 1:
        label, im = norm[0]
        frames.extend([decorate(im, label)] * max(hold, fpt))
        return frames

    for i in range(len(norm) - 1):
        (la, a), (lb, b) = norm[i], norm[i + 1]
        frames.extend([decorate(a, la)] * hold)              # hold on epoch i
        for k in range(1, fpt + 1):
            alpha = k / float(fpt + 1)
            label = la if alpha < 0.5 else lb                # dominant epoch
            frames.append(decorate(Image.blend(a, b, alpha), label))
    # hold on the final epoch
    lz, z = norm[-1]
    frames.extend([decorate(z, lz)] * hold)

    if pingpong and len(frames) > 2:
        frames = frames + frames[-2:0:-1]
    return frames


def frames_from_sequence(images: List[Image.Image], pingpong: bool = True,
                         brand: bool = True, label: Optional[str] = None,
                         labels: Optional[List[str]] = None,
                         max_size: Optional[int] = 768) -> List[Image.Image]:
    """Decorate an already-smooth sequence (e.g. a seed- or prompt-travel sweep)
    1:1 — no blending, since the frames are already continuous. Ping-pong for a
    seamless loop.

    Badge: pass `labels` (one per frame, e.g. the dominant prompt-travel word) for
    a ticking badge, or a single static `label` (e.g. "EPOCH 10"). `labels` wins.

    `images`: list of PIL frames in order.
    """
    if not images:
        return []
    base_w, base_h = images[0].size
    if max_size and max(base_w, base_h) > max_size:
        scale = max_size / float(max(base_w, base_h))
        base_w, base_h = int(base_w * scale), int(base_h * scale)
    base_w, base_h = max(2, _even(base_w)), max(2, _even(base_h))

    out = []
    for i, im in enumerate(images):
        if im.size != (base_w, base_h):
            im = im.resize((base_w, base_h), Image.LANCZOS)
        badge = labels[i] if (labels and i < len(labels)) else label
        out.append(_decorate(im, badge=badge, brand=brand))
    if pingpong and len(out) > 2:
        out = out + out[-2:0:-1]
    return out


def write_gif(frames: List[Image.Image], path: str, speed: str = "Normal", loop: int = 0):
    if not frames:
        raise ValueError("No frames to export.")
    _, _, fps = SPEED_PRESETS.get(speed, SPEED_PRESETS["Normal"])
    dur = int(round(1000.0 / fps))
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=dur, loop=loop, optimize=True, disposal=2)


def write_mp4(frames: List[Image.Image], path: str, speed: str = "Normal"):
    """Write an MP4 via OpenCV (mp4v). Raises a clear error if the codec/writer
    can't be opened so the caller can suggest GIF instead."""
    if not frames:
        raise ValueError("No frames to export.")
    import numpy as np
    import cv2
    _, _, fps = SPEED_PRESETS.get(speed, SPEED_PRESETS["Normal"])
    w, h = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer (codec unavailable). Try GIF instead.")
    try:
        for fr in frames:
            arr = np.array(fr.convert("RGB"))[:, :, ::-1]    # RGB -> BGR
            writer.write(np.ascontiguousarray(arr))
    finally:
        writer.release()
