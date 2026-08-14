"""Gizmo — the clip prep tool for Fizgig.

MiniMax H3 trains on video clips, and clips have to be exactly right: 24 fps, a frame count on
H3's 17n+5 grid, dimensions on a multiple of 32, audio at 32 kHz stereo. Fizgig refuses anything
off-spec rather than quietly fixing it, because a training app that silently transcodes footage
makes two identical-looking datasets train differently and nobody can tell why.

That leaves someone with an hour of source video and a spec sheet. Gizmo is the answer: open a
video, scrub to a moment, pick a length, save. Out comes a clip Fizgig accepts. Six sections from
one source is six marks and six clicks, and there is no project file to manage — mark, save,
repeat, close.

It is deliberately a separate app:

  * It keeps ffmpeg transcoding out of the training path, where it does not belong.
  * It opens in under a second, because it imports no torch and never touches Fizgig's GUI.

Two things it will NOT do, both on purpose:

  * It does not write a still beside each clip. That was the original design and it is wrong:
    Fizgig's latent cache keys on the filename stem with the extension stripped, so walk_03.mp4
    and walk_03.png collide on the same cache file and one silently overwrites the other.
  * It does not caption. Captioning happens in Fizgig afterwards, on the whole folder at once,
    which is also why muting is a filename suffix — there are no .txt files yet to keep in sync.

Run: venv/Scripts/pythonw.exe gizmo.pyw    (or the Launch Gizmo .bat, or from Fizgig's Image Prep)
"""

import os
import re
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

# --- the clip spec ---------------------------------------------------------------------------
# Mirrored from src/fizgig/minimax/clip.py rather than imported: that module pulls in
# fizgig.minimax.model, which pulls in torch, and a prep tool that takes ten seconds to open
# because it loaded CUDA would be a bad tool. tests/test_gizmo_spec.py asserts the two agree, so
# the copy cannot drift without something failing loudly.
FPS = 24
GRID_FRAMES = (5, 22, 39, 56, 73, 90, 107, 124)          # 17n+5, every length the VAE encodes
LATENT_FRAMES = {f: 5 * n + 2 for n, f in enumerate(GRID_FRAMES)}   # what each costs in the DiT

# What each length can be TRAINED at, by card: frames -> {free VRAM in GB: largest training
# megapixels}. None means that card cannot do it at all.
#
# Two ceilings, and the lower one wins.
#
# The TRAINING step comes from Fizgig's own swap planner (trainer.plan_vram) rather than from
# guesswork, by feeding a clip's token load in as the equivalent megapixel value — which is
# arithmetically what it is. Its activation term is linear in tokens while attention is
# quadratic, and it was anchored on stills of 256-1024 tokens, so at the long lengths it is an
# extrapolation and where it is wrong it will be optimistic. That is what sets the LENGTH limits
# here, and the long ones still want a real run behind them.
#
# The CACHING pass is measured, on a 5090 with ~30 GiB free, encoding in 17-frame groups:
#
#     0.25 MP  ~14.2 GiB      0.5 MP  ~23.2 GiB      1.0 MP  out of memory
#
# Flat in clip length — 124 frames costs the same as 22 — because the groups bound it. That is
# what sets the MEGAPIXEL limits: 1 MP is unreachable for clips of any length on a 32 GB card,
# and 0.25 MP at ~14 GiB is already marginal on a 16 GB one. Caching is fp32 today; the file is
# natively fp16, which measures 8.05 GiB and 0.09% different, so these could move.
#
# Frozen into a table because Gizmo must not import the trainer — that would pull torch in
# behind it, and a prep tool that takes ten seconds to open because it loaded CUDA is a bad tool.
CLIP_VRAM = {
    5: {16: 0.25, 24: 0.5, 32: 0.5},
    22: {16: 0.25, 24: 0.5, 32: 0.5},
    39: {16: 0.25, 24: 0.5, 32: 0.5},
    56: {16: 0.25, 24: 0.5, 32: 0.5},
    73: {16: None, 24: 0.25, 32: 0.5},
    90: {16: None, 24: 0.25, 32: 0.5},
    107: {16: None, 24: 0.25, 32: 0.25},
    124: {16: None, 24: 0.25, 32: 0.25},
}
SIZE_STEP = 32
AUDIO_SAMPLE_RATE = 32000
AUDIO_CHANNELS = 2
MUTE_SUFFIX = "_mute"

# --- audio / voice mode ------------------------------------------------------------------------
# A voice training item's duration is quantized by the same frame grid as a clip's: H3 packs
# round(frames/24*40) audio latents per grid slot, at 800 samples each. Gizmo exports segments at
# EXACTLY latents x 800 samples, so the trainer's strict duration check always passes with no
# tolerance spent. The 5-frame slot (0.2 s) is deliberately absent — fifty milliseconds of voice
# trains nothing worth a file.
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a")
AUDIO_HOP = 800                                # the audio VAE's samples-per-latent
AUDIO_GRID_FRAMES = (22, 39, 56, 73, 90, 107, 124)


def audio_latents_for(frames):
    return int(round(frames / FPS * 40))


def hop_exact_samples(frames):
    """The exact sample count a segment of `frames` exports at — and the exact count Fizgig's
    cache validation expects. 22 frames -> 29600 (0.925 s) ... 124 -> 165600 (5.175 s)."""
    return audio_latents_for(frames) * AUDIO_HOP


def is_audio_file(path):
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS


def voice_output_name(src_path, out_dir, claimed=()):
    """<source stem>_01.wav, skipping past files on disk and names the queue has spoken for.
    No mute variant: a muted voice file would be a training item with nothing in it."""
    stem = os.path.splitext(os.path.basename(src_path))[0]
    stem = re.sub(r"[^\w\-]+", "_", stem).strip("_") or "voice"
    for i in range(1, 1000):
        name = f"{stem}_{i:02d}.wav"
        if not os.path.exists(os.path.join(out_dir, name)) and name not in claimed:
            return os.path.join(out_dir, name)
    raise RuntimeError("1000 segments from one recording — give the output folder a clean start.")

# There is deliberately no size choice. Fizgig resizes clips down to the Target Megapixels on its
# Training tab — measured: a 1280x704 clip reaches the VAE as 672x384 at 0.25 MP and untouched at
# 1.0 — so cutting at native resolution keeps that decision open for every run afterwards, while
# cutting small forecloses it and can only be undone by re-cutting the set. Nothing is upscaled,
# so native is only ever the source's own detail.

# Crop shapes. H3 puts no constraint on aspect at all — its rotary position grid is normalised by
# sqrt(h*w), so it is aspect-agnostic by construction, and the only hard rule is that both sides
# are multiples of 32. These are here for CONSISTENCY, not legality: a dataset of thirty
# hand-drawn near-rectangles trains a model on thirty slightly different framings, and locking the
# drag to one shape costs nothing. Free stays the default.
CROP_SHAPES = (
    ("Free — drag any shape", None),
    ("Match the source", "source"),
    ("1:1 square", 1 / 1),
    ("16:9 wide", 16 / 9),
    ("9:16 vertical", 9 / 16),
    ("4:3", 4 / 3),
    ("3:4", 3 / 4),
    ("4:5 portrait", 4 / 5),
    ("21:9 ultrawide", 21 / 9),
)

# --- palette --------------------------------------------------------------------------------
# A local copy of Fizgig's, not an import: gizmo.py must never load lora_trainer_gui.py.
COLORS = {
    "bg_deep": "#1E2530", "bg_surface": "#252D38", "bg_hover": "#2A3542",
    "text_primary": "#F0F4F8", "text_secondary": "#8A9BAE", "text_explain": "#C3CDD9",
    "text_muted": "#5A6B7E",
    "accent": "#3B82F6", "accent_hover": "#60A5FA",
    "border": "#3A4555",
    "success": "#10B981", "warning": "#F59E0B", "error": "#EF4444",
}
FONT_FAMILY = "Segoe UI"

PREVIEW_W, PREVIEW_H = 512, 288
END_W, END_H = 320, 180

# Windows: every ffmpeg call would flash a console window, and under pythonw there is no console
# to inherit in the first place.
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW, **kw)


def find_ffmpeg():
    """imageio-ffmpeg's bundled binary first — it is already pinned in requirements.txt, so
    Gizmo adds no dependency at all — then whatever is on PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    import shutil
    return shutil.which("ffmpeg")


# --- reading a source video -------------------------------------------------------------------

def probe_source(ffmpeg, path):
    """Facts about the file the user opened. ffmpeg with no output prints its stream banner to
    stderr and exits non-zero; that is the documented way to read this without ffprobe, which
    imageio-ffmpeg does not bundle."""
    out = _run([ffmpeg, "-hide_banner", "-i", path], text=True).stderr
    info = {"fps": None, "width": None, "height": None, "duration": None,
            "has_audio": False, "sample_rate": None, "channels": None, "vcodec": None,
            "sar": 1.0, "rotation": 0, "display_width": None, "display_height": None}

    m = re.search(r"Stream #\d+:\d+.*?: Video: (\w+).*?, (\d{2,5})x(\d{2,5})", out, re.S)
    if m:
        info["vcodec"] = m.group(1)
        info["width"], info["height"] = int(m.group(2)), int(m.group(3))
    # Anamorphic sources store non-square pixels and carry the correction as a sample aspect
    # ratio — 1440x1080 with SAR 4:3 is a 1920x1080 picture. Ignore it and everything from such
    # a camera trains squashed, which looks like a bug in Gizmo rather than in the footage.
    m = re.search(r"SAR (\d+):(\d+)", out)
    if m and int(m.group(2)):
        info["sar"] = int(m.group(1)) / int(m.group(2))
    # Phones shot portrait for years by recording landscape frames and tagging them "rotate 90".
    # ffmpeg turns them upright on decode, so the frame that arrives is portrait while the banner
    # still reports the landscape dimensions it was STORED at. Take the banner at its word and
    # every such clip is scaled into a landscape box — squashed. Newer files store the pixels the
    # right way up, which is why only old ones misbehave.
    m = re.search(r"rotation of (-?\d+(?:\.\d+)?) degrees", out)
    if m:
        info["rotation"] = int(round(float(m.group(1)))) % 360
    m = re.search(r"(\d+(?:\.\d+)?)\s+fps", out)
    if m:
        info["fps"] = float(m.group(1))
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"Stream #\d+:\d+.*?: Audio: [^,]+, (\d+) Hz, (\w+)", out)
    if m:
        info["has_audio"] = True
        info["sample_rate"] = int(m.group(1))
        info["channels"] = {"mono": 1, "stereo": 2}.get(m.group(2))

    if not info["width"] or not info["fps"]:
        raise ValueError(f"{os.path.basename(path)} has no video stream Gizmo can read.")
    # The size the picture is MEANT to be seen at, after the sample aspect ratio and after any
    # rotation. Everything downstream — the preview, the target size, the crop, the export —
    # works from these, so ordinary upright square-pixel footage is unaffected and the awkward
    # cases come out the right shape instead of squashed.
    dw = max(2, int(round(info["width"] * info["sar"])))
    dh = info["height"]
    if info["rotation"] in (90, 270):
        dw, dh = dh, dw
    info["display_width"], info["display_height"] = dw, dh
    return info


def fit_within(src_w, src_h, box_w, box_h):
    """Largest size with the source's shape that fits inside the box. At least 2x2."""
    scale = min(box_w / src_w, box_h / src_h)
    return max(2, int(round(src_w * scale))), max(2, int(round(src_h * scale)))


def grab_frame(ffmpeg, path, seconds, box=(PREVIEW_W, PREVIEW_H), src_size=None):
    """One frame at `seconds`, scaled to fit `box`, as a PIL image.

    -ss BEFORE -i is the fast form, and has been frame-accurate since ffmpeg 2.1 (it seeks to the
    preceding keyframe and decodes forward). Scrubbing a long source with output-seek instead
    would decode everything up to the mark and take seconds per frame.

    The output size is COMPUTED and passed to ffmpeg, never inferred from how many bytes came
    back. Inferring it worked for 16:9 and quietly sheared everything else: raw RGB carries no
    dimensions, and picking the first width that divides the byte count lands on 512x216 for a
    4:3 source that is really 384x288 — same pixel count, wrong shape, a badly skewed picture.
    """
    if src_size is None:
        info = probe_source(ffmpeg, path)
        src_size = (info["display_width"], info["display_height"])
    w, h = fit_within(src_size[0], src_size[1], *box)
    p = _run([ffmpeg, "-hide_banner", "-loglevel", "error",
              "-ss", f"{max(0.0, seconds):.3f}", "-i", path,
              "-frames:v", "1", "-vf", f"scale={w}:{h},setsar=1",
              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    # Checked, not trusted: a mismatch means the filter did something other than what was asked,
    # and building an image from the wrong shape is precisely the bug this replaced.
    if not p.stdout or len(p.stdout) != w * h * 3:
        return None
    return Image.frombytes("RGB", (w, h), p.stdout)


# --- writing a clip ----------------------------------------------------------------------------

def snap(value):
    """Down to a multiple of 32, never to zero."""
    return max(SIZE_STEP, int(value) // SIZE_STEP * SIZE_STEP)


def target_size(src_w, src_h, megapixels):
    """Best /32 size at roughly `megapixels` that keeps the source's shape.

    Both sides have to be multiples of 32 for H3, and flooring each one independently is the
    obvious way to get there — but it BENDS THE PICTURE. 1920x1080 floored at 0.26 MP gives
    672x352, which is 1.909 against a true 1.778: a 7.4% horizontal stretch, and a face stretched
    7% trains as a face that is 7% too wide.

    So the pair is searched for instead of computed: every /32 height near the target, each with
    the /32 widths either side of the ideal, scored on aspect error first and area second. That
    gets most sizes under 2%, and build_export_command turns what is left into a crop rather than
    a stretch — losing a sliver of edge, which is honest, instead of reshaping everything in
    frame, which is not.

    Never larger than the source, on either axis. Upscaling would invent detail and then charge
    tokens for it, so a small source simply comes out at its own size.
    """
    aspect = src_w / src_h
    max_w, max_h = snap(src_w), snap(src_h)
    # Clamped to what the source actually has. Without this, asking a 480x360 clip for 1 MP
    # leaves every candidate hopelessly far from the target, the area term stops telling them
    # apart, and the search hands back 384x288 — throwing away a third of the pixels that were
    # there, to buy 2% of aspect. Asking for more than the source holds should mean "all of it".
    want_area = min(megapixels * 1_000_000, max_w * max_h)

    best = None
    for h in range(SIZE_STEP, max_h + SIZE_STEP, SIZE_STEP):
        ideal_w = h * aspect
        for w in {snap(ideal_w), snap(ideal_w) + SIZE_STEP}:
            if w < SIZE_STEP or w > max_w:
                continue
            err = abs((w / h) - aspect) / aspect
            # Both matter, so both are in one cost rather than one outranking the other. Aspect
            # alone would be a trap: 2560x1080 has exactly one /32 pair at its true 2.370, and
            # it is 2048x864 — so asking for a small clip would hand back a 1.8 MP one. Weighted
            # at 3x because the leftover aspect error is only ever a crop of that size, while
            # missing the requested megapixels costs detail that cannot come back.
            cost = abs(w * h - want_area) / want_area + 3.0 * err
            if best is None or cost < best[0]:
                best = (cost, (w, h))
    return best[1] if best else (max_w, max_h)


def tokens_for(width, height, frames):
    """What this clip costs the DiT: 32x32 pixel patches, times latent frames. Attention is
    quadratic in this number, which is the whole reason clips stop at 39 frames."""
    return (width // SIZE_STEP) * (height // SIZE_STEP) * LATENT_FRAMES[frames]


def build_export_command(ffmpeg, src, dst, start_s, frames, width, height,
                         keep_every=None, with_audio=True, crop=None, sar=1.0):
    """One pass: trim, retime, resize, and normalise the audio.

    Frame RATE is where a dataset goes quietly wrong, so it is worth being explicit about the two
    filters here:

      fps=24 preserves wall-clock time — it drops or duplicates so one real second stays one
      second. That keeps motion at true speed from any source (60 fps drops roughly every 2.5th
      frame, 25 fps drops 1 in 25) and it normalises the variable frame rate that most phone
      video has. It is the default and needs no decision from the user.

      select+setpts KEEPS every k-th source frame and re-times them to 24 fps, which plays back
      slower than life: 60 fps at every 2nd frame is 1.25x slow, 120 fps at every 2nd is 2.5x.
      This is the opposite of the intuition — it CREATES slow motion rather than avoiding it — so
      it is only ever offered as an explicit choice against the detected source rate.

    Audio is trimmed by atrim rather than left to -shortest, which decides on stream ends and can
    hand back a track a few samples long or short. Fizgig pads and trims the waveform to the frame
    count anyway, so this is belt and braces — but the belt is cheap.
    """
    if keep_every:
        vf = f"select='not(mod(n\\,{keep_every}))',setpts=N/({FPS}*TB)"
    else:
        vf = f"fps={FPS}"

    if crop:
        cx, cy, cw, ch = crop
        # A crop is expressed in DISPLAY pixels, because that is what the user drew on. For the
        # rare anamorphic source those are not the stored pixels, so the frame is normalised to
        # square pixels first and the crop lands where it was drawn.
        if abs(sar - 1.0) > 1e-3:
            vf += ",scale=iw*sar:ih:flags=lanczos,setsar=1"
        vf += f",crop={cw}:{ch}:{cx}:{cy}"
    # Cover, then crop — never stretch. force_original_aspect_ratio=increase scales the frame
    # until it covers the target box with its shape intact, and the crop takes the middle. The
    # target is chosen to sit within ~2% of the source aspect, so what the crop removes is a
    # sliver of edge. Plain scale=W:H instead would reshape everything in frame to make the
    # numbers fit, which is the one thing a training clip must not do.
    vf += (f",scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos"
           f",crop={width}:{height},setsar=1")

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{max(0.0, start_s):.3f}", "-i", src,
           "-vf", vf, "-frames:v", str(frames),
           "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p",
           "-r", str(FPS)]
    if with_audio:
        cmd += ["-af", f"aresample={AUDIO_SAMPLE_RATE},atrim=0:{frames / FPS:.6f},"
                       f"asetpts=PTS-STARTPTS",
                "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_SAMPLE_RATE),
                "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    return cmd + ["-movflags", "+faststart", dst]


def output_name(src_path, out_dir, muted, claimed=()):
    """<source stem>_01.mp4, or _01_mute.mp4, skipping past anything already there.

    `claimed` is names the queue has spoken for but not yet written — without it, two clips marked
    before either is exported would both be handed _01 and the second would overwrite the first.

    A source whose own name ends in _mute has that stripped first — otherwise every clip cut from
    it would read as muted to Fizgig regardless of what was chosen here.
    """
    stem = os.path.splitext(os.path.basename(src_path))[0]
    while stem.lower().endswith(MUTE_SUFFIX):
        stem = stem[:-len(MUTE_SUFFIX)]
    stem = re.sub(r"[^\w\-]+", "_", stem).strip("_") or "clip"
    for i in range(1, 1000):
        # An index is free only if NEITHER spelling is taken. Checking just the muted one would
        # hand out _01 and _01_mute as two different clips from the same source, which reads as
        # one clip saved twice.
        taken = any(os.path.exists(os.path.join(out_dir, f"{stem}_{i:02d}{suffix}.mp4"))
                    or f"{stem}_{i:02d}{suffix}.mp4" in claimed
                    for suffix in ("", MUTE_SUFFIX))
        if not taken:
            return os.path.join(out_dir, f"{stem}_{i:02d}{MUTE_SUFFIX if muted else ''}.mp4")
    raise RuntimeError("1000 clips from one source — give the output folder a clean start.")


def enable_file_drop(root, on_drop):
    """Let a video be dragged onto the window. Returns True if it took.

    Through the Windows shell API rather than by adding tkinterdnd2: a new dependency would mean
    everyone re-running the installer for a convenience, and Gizmo's install story is that it
    needs nothing Fizgig does not already have.

    Tk has no drag-and-drop, so the window procedure is subclassed to catch WM_DROPFILES. THE
    RULE THAT MAKES THIS SAFE: the procedure must not touch Tk. Not one call.

    That rule was learned the hard way. The first version marshalled the path with root.after,
    which is the ordinary way to get work back onto the GUI thread — and it killed the
    interpreter outright on the first real drag:

        Fatal Python error: PyEval_RestoreThread: the function must be called with the GIL held
        ... the GIL is released (the current Python thread state is NULL)

    Tk's mainloop sits inside Tcl with the GIL released. A drop arrives, Windows dispatches into
    the window procedure, and root.after re-enters Tcl from underneath the loop already running
    it. The ctypes callback was never the problem — one that only touches plain Python is fine,
    which is what this does: append the path to a list, and let an ordinary Tk timer, running
    where Tk expects to be running, pick it up.

    None of this reproduces with a synthesised drop sent from inside root.update(), because then
    Python is already executing and Tcl is not mid-loop. It has to be POSTED and left to arrive
    on an idle mainloop, exactly as Explorer's does.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        root.update_idletasks()
        # wm_frame is the REAL top-level window. winfo_id is Tk's child frame, which never
        # receives WM_DROPFILES however willing it looks.
        hwnd = int(root.wm_frame(), 16)
        user32, shell32 = ctypes.windll.user32, ctypes.windll.shell32
        WM_DROPFILES, GWLP_WNDPROC = 0x0233, -4

        # Pointer-sized by hand: WPARAM and the handle SetWindowLongPtrW returns do not fit
        # ctypes' default c_int, and the resulting OverflowError is raised inside the window
        # procedure where nothing can catch it.
        _64 = ctypes.sizeof(ctypes.c_void_p) == 8
        LONG_PTR = ctypes.c_longlong if _64 else ctypes.c_long
        UINT_PTR = ctypes.c_ulonglong if _64 else ctypes.c_ulong
        WNDPROC = ctypes.WINFUNCTYPE(LONG_PTR, ctypes.c_void_p, ctypes.c_uint,
                                     UINT_PTR, LONG_PTR)
        for fn, args, res in (
                (user32.CallWindowProcW,
                 [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, UINT_PTR, LONG_PTR], LONG_PTR),
                (user32.SetWindowLongPtrW,
                 [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p], ctypes.c_void_p),
                (user32.DefWindowProcW,
                 [ctypes.c_void_p, ctypes.c_uint, UINT_PTR, LONG_PTR], LONG_PTR),
                (shell32.DragQueryFileW,
                 [ctypes.c_void_p, ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_uint], ctypes.c_uint),
                (shell32.DragFinish, [ctypes.c_void_p], None)):
            fn.argtypes, fn.restype = args, res

        shell32.DragAcceptFiles(ctypes.c_void_p(hwnd), True)
        inbox, old = [], [None]

        def proc(h, msg, wparam, lparam):
            # Plain Python only. No Tk, no Tcl, nothing that can raise if it can be helped — this
            # runs on every message the window receives.
            try:
                if msg == WM_DROPFILES:
                    try:
                        buf = ctypes.create_unicode_buffer(1024)
                        if shell32.DragQueryFileW(ctypes.c_void_p(wparam), 0, buf, 1024):
                            inbox.append(buf.value)
                    finally:
                        shell32.DragFinish(ctypes.c_void_p(wparam))
                    return 0
                return user32.CallWindowProcW(old[0], h, msg, wparam, lparam)
            except Exception:
                try:
                    return user32.DefWindowProcW(h, msg, wparam, lparam)
                except Exception:
                    return 0

        cb = WNDPROC(proc)
        old[0] = user32.SetWindowLongPtrW(ctypes.c_void_p(hwnd), GWLP_WNDPROC,
                                          ctypes.cast(cb, ctypes.c_void_p))
        if not old[0]:
            return False
        root._drop_callback = cb          # held, or it is collected and the window proc dangles

        def drain():
            while inbox:
                on_drop(inbox.pop(0))
            root.after(120, drain)

        root.after(120, drain)
        return True
    except Exception:
        return False


def audio_playback_backend():
    """How this machine can play a WAV, or None if it cannot.

    Windows has winsound in the standard library, which is the whole reason listening costs no
    new dependency. Linux — which is what a RunPod pod is — usually has neither a player nor an
    audio device in the container, so the feature disables itself with a reason rather than
    failing at the moment someone presses the button.
    """
    if os.name == "nt":
        try:
            import winsound  # noqa: F401
            return "winsound"
        except Exception:
            return None
    import shutil
    for exe in ("paplay", "aplay", "afplay"):
        if shutil.which(exe):
            return exe
    return None


_PLAYBACK = None            # resolved once, on first use


def _play_wav(path):
    global _PLAYBACK
    if _PLAYBACK is None:
        _PLAYBACK = audio_playback_backend() or ""
    if not _PLAYBACK:
        raise RuntimeError("no way to play audio on this machine")
    if _PLAYBACK == "winsound":
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    else:
        subprocess.Popen([_PLAYBACK, path], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)


def _stop_wav():
    if _PLAYBACK == "winsound":
        import winsound
        winsound.PlaySound(None, winsound.SND_PURGE)


def set_live_volume(pct):
    """Listening volume that takes effect MID-PLAY, no restart.

    winsound has no volume of its own, but it plays through the process's wave-out session, and
    winmm's waveOutSetVolume moves that session live — so the slider works while a preview is
    playing instead of demanding a stop-and-start. Per-process: the system volume and every
    other app are untouched, and nothing here ever reaches a saved file.

    Returns True when the live path took effect; False means the caller should fall back to
    baking the gain into the preview file (the non-Windows external-player path)."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        v = max(0, min(0xFFFF, int(0xFFFF * float(pct) / 100.0)))
        return ctypes.windll.winmm.waveOutSetVolume(0, v | (v << 16)) == 0
    except Exception:
        return False


def count_frames(ffmpeg, path):
    """Decoded frame count. The container banner cannot be trusted for this (duration x fps
    rounds), and it is the one value Fizgig refuses a clip over, so it is checked for real."""
    out = _run([ffmpeg, "-hide_banner", "-i", path, "-map", "0:v:0",
                "-c", "copy", "-f", "null", "-"], text=True).stderr
    hits = re.findall(r"frame=\s*(\d+)", out)
    return int(hits[-1]) if hits else None


# --- the app -----------------------------------------------------------------------------------

class ToolTip:
    """Hover help. Same behaviour and look as Fizgig's, reimplemented rather than imported —
    gizmo.py must never load lora_trainer_gui.py.

    Delayed by half a second so sweeping the mouse across the frame-step buttons does not leave a
    trail of popups, and bound to <Button> as well as <Leave> because a tooltip left hanging over
    a button you just clicked hides the thing you clicked.
    """

    def __init__(self, widget, text, delay=500):
        self.widget, self.text, self.delay = widget, text, delay
        self.window = None
        self._after = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<Button>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except Exception:
                pass
            self._after = None

    def _show(self):
        if self.window or not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify=tk.LEFT, wraplength=380,
                 background=COLORS["bg_surface"], foreground=COLORS["text_primary"],
                 relief=tk.SOLID, borderwidth=1, font=(FONT_FAMILY, 9),
                 padx=8, pady=6).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self.window:
            self.window.destroy()
            self.window = None


class Gizmo:
    def __init__(self, root):
        self.root = root
        root.title("Gizmo — clip and voice prep for Fizgig")
        root.configure(bg=COLORS["bg_deep"])
        # Wide enough that the vertical scrollbar never eats into the content (which measures
        # ~985 px), tall enough for two cards at a time. The rest scrolls.
        root.geometry("1060x900")
        root.minsize(1010, 620)

        self.ffmpeg = find_ffmpeg()
        self.src = None
        self.info = None
        self.frame_pos = 0            # in SOURCE frames
        self.photo = None             # live references, or Tk garbage-collects the images
        self.end_photo = None
        self._scrub_job = None
        self._busy = False
        self._motion_keep = [None]    # parallel to the Motion dropdown; see _keep_every
        self.queue = []               # marked clips, exported in one go — see add_to_queue
        self.crop = None              # (x, y, w, h) in the source's DISPLAY pixels, or None
        self._crop_anchor = None
        self._radios = []             # every radio button, armed together once a video is open
        self._playing = False
        self._play_job = None
        self._video_sound = None      # (span_start_s, span_len_s, consumed_s) while paused
        self._video_offset = 0.0      # how far into the current span playback started
        self._video_t0 = 0.0

        # Voice / audio mode state — parallel world, deliberately separate variables: a video
        # open on one tab and a recording on the other must never share a playhead.
        self.audio_src = None
        self.audio_dur = 0.0          # seconds
        self.audio_from_video = False
        self.audio_pos = 0.0          # segment start, seconds
        self.audio_peaks = None       # [(lo, hi) 0..1] per column of the CURRENT view, or None
        self._audio_samples = None    # 8 kHz mono f32 array — peaks re-slice from this on zoom
        self._audio_peak_scale = 1.0  # one normalisation for the whole file, so zoom keeps scale
        self.audio_view = None        # (start_s, dur_s) window, None = the whole file
        self._audio_peaks_key = None  # cache key so refresh doesn't re-slice unchanged views
        self.audio_queue = []
        self._audio_playing = False
        self._audio_play_mode = None  # "file" (listen through) or "segment" (the export span)
        self._audio_play_from = 0.0   # where the current playback started, for the cursor
        self._audio_play_t0 = 0.0
        self._audio_play_rate = 1.0   # J/K/L shuttle speed
        self._audio_play_dir = 1      # 1 forward, -1 backward (J)
        self._audio_resume = None     # (mode, position) left by pause, for space to resume
        self._audio_play_job = None
        self._audio_cursor_job = None
        self._whisper_busy = False

        self._style()
        self._build()
        # Three ways in besides the Open button, none of them a dependency: drop a video on the
        # window, hand Gizmo a path on the command line — which is what dropping one onto the
        # launcher does, and what Explorer's "Open with" does — or paste a path with Ctrl+V.
        self.can_drop = enable_file_drop(self.root, self.open_path)
        self.root.bind("<Control-v>", lambda _e: self.paste_path())
        if not self.ffmpeg:
            messagebox.showerror(
                "Gizmo — no ffmpeg",
                "Gizmo needs ffmpeg, which normally arrives with Fizgig's install.\n\n"
                "Run install_fizgig.bat to restore it, or put ffmpeg on your PATH.")

    # -- chrome ---------------------------------------------------------------------------------
    def _style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("G.TCombobox", fieldbackground=COLORS["bg_hover"],
                    background=COLORS["bg_hover"], foreground=COLORS["text_primary"],
                    arrowcolor=COLORS["text_secondary"], bordercolor=COLORS["border"],
                    font=(FONT_FAMILY, 10))
        # configure() alone is not enough and the failure is invisible until you look: a READONLY
        # combobox — which all of these are — draws its text as a selection, so it takes
        # selectforeground on selectbackground, and clam's stock values for those are a light
        # system pair. The result is white text on a white field.
        s.map("G.TCombobox",
              fieldbackground=[("disabled", COLORS["bg_surface"]),
                               ("readonly", COLORS["bg_hover"]),
                               ("!disabled", COLORS["bg_hover"])],
              foreground=[("disabled", COLORS["text_muted"]),
                          ("readonly", COLORS["text_primary"]),
                          ("!disabled", COLORS["text_primary"])],
              selectbackground=[("disabled", COLORS["bg_surface"]),
                                ("readonly", COLORS["bg_hover"]),
                                ("!disabled", COLORS["bg_hover"])],
              selectforeground=[("disabled", COLORS["text_muted"]),
                                ("readonly", COLORS["text_primary"]),
                                ("!disabled", COLORS["text_primary"])],
              arrowcolor=[("disabled", COLORS["text_muted"])],
              bordercolor=[("focus", COLORS["accent"])])

        # The list that drops DOWN is a plain Tk listbox built by Tcl, which ttk styling never
        # reaches. Without this it opens white-on-white too, and only when clicked.
        for option, value in (("background", COLORS["bg_surface"]),
                              ("foreground", COLORS["text_primary"]),
                              ("selectBackground", COLORS["accent"]),
                              ("selectForeground", COLORS["text_primary"])):
            self.root.option_add(f"*TCombobox*Listbox.{option}", value)

        s.configure("G.Horizontal.TScale", background=COLORS["bg_surface"],
                    troughcolor=COLORS["bg_deep"])
        # The window's own scrollbar, which clam otherwise draws in stock light grey against
        # every dark card.
        s.configure("Vertical.TScrollbar", background=COLORS["bg_hover"],
                    troughcolor=COLORS["bg_deep"], bordercolor=COLORS["bg_deep"],
                    arrowcolor=COLORS["text_secondary"])
        s.map("Vertical.TScrollbar", background=[("active", COLORS["accent"])])
        s.configure("Horizontal.TScrollbar", background=COLORS["bg_hover"],
                    troughcolor=COLORS["bg_deep"], bordercolor=COLORS["bg_deep"],
                    arrowcolor=COLORS["text_secondary"])
        s.map("Horizontal.TScrollbar", background=[("active", COLORS["accent"])])

        # The mode tabs. Clam's stock notebook is a light strip on our dark ground; restyle it
        # to read as part of the app rather than a window inside a window.
        s.configure("G.TNotebook", background=COLORS["bg_deep"], borderwidth=0)
        s.configure("G.TNotebook.Tab", background=COLORS["bg_surface"],
                    foreground=COLORS["text_secondary"], font=(FONT_FAMILY, 11, "bold"),
                    padding=(18, 8), borderwidth=0)
        s.map("G.TNotebook.Tab",
              background=[("selected", COLORS["accent"]), ("active", COLORS["bg_hover"])],
              foreground=[("selected", COLORS["text_primary"])])

    def _card(self, parent, title, description=None):
        outer = tk.Frame(parent, bg=COLORS["bg_surface"], highlightthickness=1,
                         highlightbackground=COLORS["border"])
        outer.pack(fill=tk.X, padx=16, pady=(0, 12))
        inner = tk.Frame(outer, bg=COLORS["bg_surface"])
        inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)
        tk.Label(inner, text=title, font=(FONT_FAMILY, 12, "bold"),
                 bg=COLORS["bg_surface"], fg=COLORS["text_primary"]).pack(anchor="w")
        if description:
            tk.Label(inner, text=description, font=(FONT_FAMILY, 9), justify=tk.LEFT,
                     wraplength=900, bg=COLORS["bg_surface"],
                     fg=COLORS["text_explain"]).pack(anchor="w", pady=(2, 8))
        body = tk.Frame(inner, bg=COLORS["bg_surface"])
        body.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        return body

    def _button(self, parent, text, command, kind="normal", tip=None, pad=14):
        # takefocus=0 plus a refocus after every click: a button that keeps keyboard focus
        # eats the space bar (Tk's Button class binding fires before any root handler can),
        # and the space bar's job in this tool is the transport, not re-pressing Open.
        def _run_and_refocus():
            command()
            try:
                self.root.focus_set()
            except tk.TclError:
                pass
        b = tk.Button(parent, text=text, command=_run_and_refocus,
                      bg={"normal": COLORS["bg_hover"], "accent": COLORS["accent"]}[kind],
                      fg=COLORS["text_primary"], takefocus=0,
                      activebackground=COLORS["accent_hover"],
                      activeforeground=COLORS["text_primary"],
                      font=(FONT_FAMILY, 10, "bold" if kind == "accent" else "normal"),
                      relief=tk.FLAT, bd=0, padx=pad, pady=7, cursor="hand2")
        if tip:
            ToolTip(b, tip)
        return b

    def _build(self):
        # Scrollable, because the whole app is four stacked cards and a 1080p laptop with a
        # taskbar has about 950 usable pixels. Without this the Save button is the thing that
        # falls off the bottom.
        shell = tk.Frame(self.root, bg=COLORS["bg_deep"])
        shell.pack(fill=tk.BOTH, expand=True)
        self._scroll_canvas = canvas = tk.Canvas(shell, bg=COLORS["bg_deep"],
                                                 highlightthickness=0)
        bar = ttk.Scrollbar(shell, orient=tk.VERTICAL, command=canvas.yview)
        body = tk.Frame(canvas, bg=COLORS["bg_deep"])
        self._scroll_window = canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        body.bind("<Configure>",
                  lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(self._scroll_window, width=e.width))
        self.root.bind_all("<MouseWheel>",
                           lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        head = tk.Frame(body, bg=COLORS["bg_deep"])
        head.pack(fill=tk.X, padx=16, pady=(14, 10))
        tk.Label(head, text="Gizmo", font=(FONT_FAMILY, 22, "bold"),
                 bg=COLORS["bg_deep"], fg=COLORS["text_primary"]).pack(anchor="w")
        tk.Label(head, text="Cut training clips and voice segments Fizgig will accept — on the "
                            "grid, sized, captioned, with the sound sorted out.",
                 font=(FONT_FAMILY, 11), bg=COLORS["bg_deep"],
                 fg=COLORS["text_secondary"]).pack(anchor="w")

        # Two modes, one window: video clips and voice segments share the queue-then-export
        # rhythm but nothing else, so each gets its own tab rather than a mode toggle whose
        # state has to be remembered.
        self.notebook = ttk.Notebook(body, style="G.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)
        video_tab = tk.Frame(self.notebook, bg=COLORS["bg_deep"])
        audio_tab = tk.Frame(self.notebook, bg=COLORS["bg_deep"])
        self.notebook.add(video_tab, text="Video clips")
        self.notebook.add(audio_tab, text="Voice / audio")
        self._video_tab, self._audio_tab = video_tab, audio_tab

        # Settings come BEFORE finding the moment, because the last-frame preview is only
        # meaningful once the length and the motion are settled — scrub first and it shows the
        # end of whatever length happened to be left over from the previous clip.
        tk.Frame(video_tab, bg=COLORS["bg_deep"], height=10).pack()
        self._build_source_card(video_tab)
        self._build_settings_card(video_tab)
        self._build_scrub_card(video_tab)
        self._build_queue_card(video_tab)

        tk.Frame(audio_tab, bg=COLORS["bg_deep"], height=10).pack()
        self._build_audio_tab(audio_tab)

        self._bind_keys()
        self._set_enabled(False)
        self._refresh_cost()
        self._refresh_queue_box()

        # Nothing HOLDS keyboard focus in this tool. A combobox, listbox or slider that kept
        # it after a click would eat the transport keys (space/J/K/L, arrows) exactly the way
        # a focused button ate space — so each hands focus back to the window once used.
        # Class-wide: covers both tabs and anything added later.
        for cls, seq in (("TCombobox", "<<ComboboxSelected>>"),
                         ("Listbox", "<ButtonRelease-1>"),
                         ("TScale", "<ButtonRelease-1>")):
            self.root.bind_class(cls, seq, lambda _e: self.root.focus_set(), add="+")

        # The other half: a text field RELEASES focus when you click away from it. A canvas or
        # label never takes focus, so without this, typing a trigger word then clicking the
        # waveform left the entry focused — and space typed into it (well, was shielded into
        # doing nothing) instead of driving the transport.
        def _click_anywhere(e):
            try:
                cls = e.widget.winfo_class()
            except Exception:
                return
            if cls not in ("Entry", "TEntry", "Text", "Listbox", "TCombobox"):
                self.root.focus_set()
        self.root.bind_all("<Button-1>", _click_anywhere, add="+")
        self.root.focus_set()

    def _build_source_card(self, body):
        c = self._card(body, "1. Source video", "Any format, any frame rate, any size.")
        row = tk.Frame(c, bg=COLORS["bg_surface"])
        row.pack(fill=tk.X)
        self._button(row, "Open video…", self.open_video, "accent",
                     tip="Open the video you want to cut clips out of.\n\n"
                         "Anything ffmpeg can read: phone footage, a camera file, a download. "
                         "Gizmo never modifies it — it only reads.").pack(side=tk.LEFT)
        self.src_label = tk.Label(row, text="No video open", font=(FONT_FAMILY, 10),
                                  bg=COLORS["bg_surface"], fg=COLORS["text_muted"])
        self.src_label.pack(side=tk.LEFT, padx=12)
        self.drop_hint = tk.Label(row, text="…or just drag a video onto this window",
                                  font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"],
                                  fg=COLORS["text_muted"])
        self.drop_hint.pack(side=tk.RIGHT)
        ToolTip(self.drop_hint,
                "Drag a video from Explorer straight onto this window.\n\nGizmo also opens one "
                "dragged onto the Launch Gizmo shortcut, one sent with right-click and Open "
                "with, or a path pasted with Ctrl+V.")
        self.src_info = tk.Label(c, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                 bg=COLORS["bg_surface"], fg=COLORS["text_explain"])
        self.src_info.pack(anchor="w", pady=(8, 0))
        ToolTip(self.src_info,
                "What Gizmo read from the file. None of it has to be right — the whole point is "
                "that Gizmo converts it. It is here so you can see what you are starting from.")

    def _build_scrub_card(self, body):
        c = self._card(body, "3. Find the moment",
                       "The playhead is where the clip STARTS, and the last frame follows from "
                       "the length you set above. Drag it, step with the arrow keys (Shift for "
                       "ten at a time, Home for the beginning), or type an exact frame below.")
        shots = tk.Frame(c, bg=COLORS["bg_surface"])
        shots.pack()
        # Two frames, not one: choosing a clip means knowing what is in it, and the end frame is
        # the half you cannot see from the playhead. It costs one more ffmpeg seek per scrub,
        # which is why it is small and shares the same debounce.
        first = tk.Frame(shots, bg=COLORS["bg_surface"])
        first.pack(side=tk.LEFT)
        tk.Label(first, text="FIRST FRAME", font=(FONT_FAMILY, 8, "bold"),
                 bg=COLORS["bg_surface"], fg=COLORS["text_muted"]).pack(anchor="w")
        self.canvas = tk.Canvas(first, width=PREVIEW_W, height=PREVIEW_H, bg="#000000",
                                highlightthickness=1, highlightbackground=COLORS["border"])
        self.canvas.pack()
        ToolTip(self.canvas,
                "The first frame of the clip you are about to save.\n\n"
                "With Choose an area set above, drag here to pick the part of the frame to train "
                "on. Drag again to redraw it.")
        self.canvas.bind("<Button-1>", self._crop_press)
        self.canvas.bind("<B1-Motion>", self._crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self._crop_release)

        last = tk.Frame(shots, bg=COLORS["bg_surface"])
        last.pack(side=tk.LEFT, padx=(12, 0))
        tk.Label(last, text="LAST FRAME", font=(FONT_FAMILY, 8, "bold"),
                 bg=COLORS["bg_surface"], fg=COLORS["text_muted"]).pack(anchor="w")
        self.end_canvas = tk.Canvas(last, width=END_W, height=END_H, bg="#000000",
                                    highlightthickness=1, highlightbackground=COLORS["border"])
        self.end_canvas.pack()
        ToolTip(self.end_canvas,
                "Where the clip ENDS, at the length you have chosen.\n\n"
                "Worth a glance before saving: it is how you catch a clip that runs past the "
                "moment you wanted, or into a cut.")

        self.scale = ttk.Scale(c, from_=0, to=100, orient=tk.HORIZONTAL,
                               command=self._on_scrub, style="G.Horizontal.TScale")
        self.scale.pack(fill=tk.X, pady=(10, 6))
        ToolTip(self.scale, "Drag to move through the source video. The clip starts here.")

        nav = tk.Frame(c, bg=COLORS["bg_surface"])
        nav.pack()
        self._button(nav, "⏮ Start", self.go_to_start, pad=10,
                     tip="Jump to the beginning of the source video.  (Home)").pack(
            side=tk.LEFT, padx=3)
        for text, delta, tip in (
                ("◀◀ 10", -10, "Back ten frames.  (Shift + ←)"),
                ("◀ 1", -1, "Back one frame.  (←)"),
                ("1 ▶", 1, "Forward one frame.  (→)"),
                ("10 ▶▶", 10, "Forward ten frames.  (Shift + →)")):
            self._button(nav, text, lambda d=delta: self.step(d), pad=10,
                         tip=tip).pack(side=tk.LEFT, padx=3)
        # Typed as well as dragged. A slider cannot land on frame 1483 of 9000, and "start it
        # exactly where the last clip ended" is an ordinary thing to want.
        posrow = tk.Frame(c, bg=COLORS["bg_surface"])
        posrow.pack(pady=(10, 0))
        tk.Label(posrow, text="Start at frame", font=(FONT_FAMILY, 10),
                 bg=COLORS["bg_surface"], fg=COLORS["text_secondary"]).pack(side=tk.LEFT)
        self.pos_var = tk.StringVar(value="0")
        self.pos_entry = tk.Entry(posrow, textvariable=self.pos_var, width=8,
                                  justify="right", bg=COLORS["bg_hover"],
                                  fg=COLORS["text_primary"],
                                  insertbackground=COLORS["text_primary"], relief=tk.FLAT,
                                  font=("Consolas", 10))
        self.pos_entry.pack(side=tk.LEFT, padx=6, ipady=3)
        self._shield_from_hotkeys(self.pos_entry)
        self.pos_entry.bind("<Return>", lambda _e: self._go_to_typed())
        self.pos_entry.bind("<KP_Enter>", lambda _e: self._go_to_typed())
        self.pos_entry.bind("<FocusOut>", lambda _e: self._go_to_typed())
        POS_TIP = ("Type an exact frame and press Enter.\n\n"
                   "Add s for seconds instead — 12.5s. Out-of-range values are pulled back to "
                   "the nearest frame that exists rather than refused.")
        ToolTip(self.pos_entry, POS_TIP)
        self.pos_label = tk.Label(posrow, text="", font=(FONT_FAMILY, 10, "bold"),
                                  bg=COLORS["bg_surface"], fg=COLORS["text_primary"])
        self.pos_label.pack(side=tk.LEFT, padx=(6, 0))
        ToolTip(self.pos_label, POS_TIP)
        self.span_label = tk.Label(c, text="", font=(FONT_FAMILY, 9),
                                   bg=COLORS["bg_surface"], fg=COLORS["text_explain"])
        self.span_label.pack()
        ToolTip(self.span_label,
                "How much of the SOURCE this clip uses. Slow motion uses more of it than real "
                "time does for the same number of output frames.")

    def _build_settings_card(self, body):
        c = self._card(body, "2. Clip settings",
                       "Set these first — the length decides where the clip ends, so the preview "
                       "on the next card depends on them.")
        grid = tk.Frame(c, bg=COLORS["bg_surface"])
        grid.pack(fill=tk.X)
        grid.columnconfigure(1, weight=1)

        def label(r, text, tip):
            lbl = tk.Label(grid, text=text, font=(FONT_FAMILY, 10), bg=COLORS["bg_surface"],
                           fg=COLORS["text_secondary"])
            lbl.grid(row=r, column=0, sticky="w", pady=5, padx=(0, 12))
            ToolTip(lbl, tip)
            return lbl

        LEN_TIP = ("How many frames the clip is.\n\n"
                   "Not a free choice: H3's VAE encodes video in groups of 17, so the only "
                   "lengths that exist are 5, 22, 39, 56, 73, 90, 107 and 124.\n\n"
                   "22 is the shortest that shows real movement. Beyond that the cost climbs "
                   "hard — attention scales with the square of the token count — so the line "
                   "below says what each length can be trained at on 16, 24 and 32 GB. Longer is "
                   "offered rather than withheld: whether it is affordable depends on the card "
                   "you have, and you know which that is.")
        label(0, "Length:", LEN_TIP)
        self.len_var = tk.StringVar()
        self.len_box = ttk.Combobox(grid, textvariable=self.len_var, state="readonly",
                                    style="G.TCombobox", width=46,
                                    values=[self._length_label(f) for f in GRID_FRAMES])
        self.len_box.current(1)
        self.len_box.grid(row=0, column=1, sticky="w", pady=5)
        self.len_box.bind("<<ComboboxSelected>>", lambda _e: self._on_length_changed())
        ToolTip(self.len_box, LEN_TIP)
        self.len_note = tk.Label(grid, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                 wraplength=820, bg=COLORS["bg_surface"],
                                 fg=COLORS["text_explain"])
        self.len_note.grid(row=1, column=1, sticky="w")
        ToolTip(self.len_note, LEN_TIP)

        SIZE_TIP = ("The size the clip is saved at — whatever you cut, at its own resolution.\n\n"
                    "There is nothing to choose because there is no good reason to throw pixels "
                    "away here: Fizgig resizes clips down to the Target Megapixels on its "
                    "Training tab, so cutting large keeps that decision open, and cutting small "
                    "cannot be undone without re-cutting the set.\n\n"
                    "Both sides land on a multiple of 32, which is what H3 needs, and the shape "
                    "is kept — nothing is ever scaled up.")
        label(2, "Size:", SIZE_TIP)
        self.size_label = tk.Label(grid, text="", font=(FONT_FAMILY, 10),
                                   bg=COLORS["bg_surface"], fg=COLORS["text_primary"])
        self.size_label.grid(row=2, column=1, sticky="w", pady=5)
        ToolTip(self.size_label, SIZE_TIP)

        MOTION_TIP = ("Real time is what you want almost always: whatever the source runs at, one "
                      "real second stays one second.\n\n"
                      "The slow-motion options keep MORE of the original frames instead of "
                      "resampling, so the action plays back slower than life. Only worth it when "
                      "the slow-motion look is the thing you are training.")
        label(4, "Motion:", MOTION_TIP)
        self.motion_var = tk.StringVar()
        self.motion_box = ttk.Combobox(grid, textvariable=self.motion_var, state="readonly",
                                       style="G.TCombobox", width=46, values=[])
        self.motion_box.grid(row=4, column=1, sticky="w", pady=5)
        self.motion_box.bind("<<ComboboxSelected>>", lambda _e: self._on_motion_changed())
        ToolTip(self.motion_box, MOTION_TIP)

        CROP_TIP = ("Train on part of the frame instead of all of it.\n\n"
                    "A clip's cost is its pixels, so a wide shot spends most of it on background. "
                    "Crop to the subject and every token goes on what you actually want learned — "
                    "and one source can give several clips, one per subject.\n\n"
                    "The crop holds still for the whole clip, so check the LAST frame preview: "
                    "the rectangle is drawn on both, and a subject that walks out of it is "
                    "obvious there and nowhere else.")
        label(5, "Crop:", CROP_TIP)
        croprow = tk.Frame(grid, bg=COLORS["bg_surface"])
        croprow.grid(row=5, column=1, sticky="w", pady=5)
        self.crop_var = tk.BooleanVar(value=False)
        for text, val, tip in (
                ("Whole frame", False, "Use everything in shot."),
                ("Choose an area", True,
                 "Then drag a rectangle on the FIRST FRAME preview below. It snaps to a "
                 "multiple of 32, and drag again to redraw it.")):
            rb = tk.Radiobutton(croprow, text=text, variable=self.crop_var, value=val,
                                command=self._on_crop_mode_changed,
                                bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
                                selectcolor=COLORS["bg_deep"],
                                activebackground=COLORS["bg_surface"],
                                activeforeground=COLORS["text_primary"],
                                font=(FONT_FAMILY, 10), highlightthickness=0, bd=0,
                                cursor="hand2")
            rb.pack(side=tk.LEFT, padx=(0, 16))
            ToolTip(rb, tip)
            self._radios.append(rb)
        self.crop_clear_btn = self._button(croprow, "Reset", self.clear_crop, pad=10,
                                           tip="Go back to the whole frame.")
        self.crop_clear_btn.pack(side=tk.LEFT)

        SHAPE_TIP = ("Lock the crop to a shape while you drag it.\n\n"
                     "H3 does not care — any shape trains, as long as both sides land on a "
                     "multiple of 32, which Gizmo handles. This is for consistency: thirty "
                     "hand-drawn rectangles are thirty slightly different framings, and one "
                     "shape across the set is usually what you meant.\n\n"
                     "The grid is coarse, so the lock gets as close as multiples of 32 allow "
                     "and tells you what it actually managed.")
        tk.Label(croprow, text="Shape", font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"],
                 fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(14, 6))
        self.shape_var = tk.StringVar()
        self.shape_box = ttk.Combobox(croprow, textvariable=self.shape_var, state="readonly",
                                      style="G.TCombobox", width=22,
                                      values=[name for name, _ in CROP_SHAPES])
        self.shape_box.current(0)
        self.shape_box.pack(side=tk.LEFT)
        self.shape_box.bind("<<ComboboxSelected>>", lambda _e: self._on_shape_changed())
        ToolTip(self.shape_box, SHAPE_TIP)
        self.crop_note = tk.Label(grid, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                  wraplength=820, bg=COLORS["bg_surface"],
                                  fg=COLORS["text_explain"])
        self.crop_note.grid(row=6, column=1, sticky="w")

        SOUND_TIP = ("H3 generates sound as well as video, so a clip's audio can be training data "
                     "too.\n\n"
                     "Mute the ones where it should not be — a cough, the wrong speaker, music "
                     "over the top. A muted clip trains its video exactly the same; only the "
                     "sound is ignored.")
        label(7, "Sound:", SOUND_TIP)
        snd = tk.Frame(grid, bg=COLORS["bg_surface"])
        snd.grid(row=7, column=1, sticky="w", pady=5)
        # Sticky between saves on purpose: one source video routinely has sections worth training
        # on and sections with a cough, the wrong speaker, or music over the top, so this gets
        # toggled far more often than it gets reset.
        self.mute_var = tk.BooleanVar(value=False)
        for text, val, tip in (
                ("Train on this clip's sound", False,
                 "The clip's audio becomes a training target — this is how H3 learns a voice."),
                ("Mute — video only", True,
                 "Saves with _mute in the filename. Fizgig trains the video and ignores the "
                 "sound.\n\nThe audio stays in the file, so you can still play it back — and "
                 "change your mind later by renaming, without re-exporting.")):
            rb = tk.Radiobutton(snd, text=text, variable=self.mute_var, value=val,
                                command=self._refresh_planned_name,
                                bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
                                selectcolor=COLORS["bg_deep"],
                                activebackground=COLORS["bg_surface"],
                                activeforeground=COLORS["text_primary"], font=(FONT_FAMILY, 10),
                                highlightthickness=0, bd=0, cursor="hand2")
            rb.pack(side=tk.LEFT, padx=(0, 16))
            ToolTip(rb, tip)
            self._radios.append(rb)

        # Listen before deciding. Muting is a judgement about what the clip actually sounds like,
        # and there is no way to make that judgement from a picture of it.
        label(8, "Listen:", "Play the marked section so you can hear what you would be training "
                            "on before you choose.")
        listen = tk.Frame(grid, bg=COLORS["bg_surface"])
        listen.grid(row=8, column=1, sticky="w", pady=5)
        self.play_btn = self._button(
            listen, "▶  Play sound", self.toggle_play, pad=10,
            tip="Play the audio under the clip you have marked — exactly the section that would "
                "be saved.\n\nTakes a moment to extract the first time you press it.")
        self.play_btn.pack(side=tk.LEFT)
        tk.Label(listen, text="Volume", font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"],
                 fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(14, 6))
        self.volume_var = tk.DoubleVar(value=70)
        self.volume_scale = ttk.Scale(listen, from_=0, to=100, orient=tk.HORIZONTAL, length=150,
                                      variable=self.volume_var, style="G.Horizontal.TScale",
                                      command=lambda v: set_live_volume(v))
        self.volume_scale.pack(side=tk.LEFT)
        ToolTip(self.volume_scale,
                "Listening volume only — moves live while the sound plays. The saved clip's "
                "audio is never touched — a quiet source stays quiet, and H3 sees exactly what "
                "your camera recorded.")

        self.sound_note = tk.Label(grid, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                   wraplength=820, bg=COLORS["bg_surface"],
                                   fg=COLORS["text_explain"])
        self.sound_note.grid(row=9, column=1, sticky="w")

        OUT_TIP = ("Where the clips land.\n\n"
                   "Point it at your Fizgig training folder and the clips are ready to caption "
                   "the moment you are done here. Defaults to a fizgig_clips folder beside the "
                   "source video.")
        label(10, "Save to:", OUT_TIP)
        outrow = tk.Frame(grid, bg=COLORS["bg_surface"])
        outrow.grid(row=10, column=1, sticky="ew", pady=5)
        self.out_var = tk.StringVar(value="")
        self.out_var.trace_add("write", lambda *_a: self._refresh_planned_name())
        out_entry = tk.Entry(outrow, textvariable=self.out_var, bg=COLORS["bg_hover"],
                             fg=COLORS["text_primary"], insertbackground=COLORS["text_primary"],
                             relief=tk.FLAT, font=(FONT_FAMILY, 10))
        out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self._shield_from_hotkeys(out_entry)
        ToolTip(out_entry, OUT_TIP)
        self._button(outrow, "Browse…", self.pick_output,
                     tip="Choose the folder to save clips into.").pack(side=tk.LEFT, padx=(8, 0))

        self.cost_label = tk.Label(c, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                   wraplength=900, bg=COLORS["bg_surface"],
                                   fg=COLORS["text_explain"])
        self.cost_label.pack(anchor="w", pady=(10, 0))
        ToolTip(self.cost_label,
                "Training cost, so a length is never an accidental decision.\n\n"
                "Attention scales with the square of this number, so a clip is dramatically more "
                "expensive to train than a still — that is the whole reason clips are short.")

    def _build_queue_card(self, body):
        c = self._card(body, "4. Mark it, then move on",
                       "Marking is fast and encoding is slow, so they are separate. Add every "
                       "section you want from this video, then export the lot in one go — and "
                       "keep scrubbing while it runs.")
        row = tk.Frame(c, bg=COLORS["bg_surface"])
        row.pack(fill=tk.X)
        self.add_btn = self._button(
            row, "➕  Add to queue", self.add_to_queue, "accent",
            tip="Remember this clip — where it starts, how long, its size, its sound.  (Ctrl+S)"
                "\n\nNothing is written yet, so adding is instant and you can carry on marking.")
        self.add_btn.pack(side=tk.LEFT)
        self.export_btn = self._button(
            row, "Export queue", self.export_queue,
            tip="Encode everything in the queue, one after another.\n\nA second or two per clip. "
                "Anything that fails says why and the rest still finish.")
        self.export_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.status = tk.Label(row, text="", font=(FONT_FAMILY, 10), bg=COLORS["bg_surface"],
                               fg=COLORS["text_secondary"])
        self.status.pack(side=tk.LEFT, padx=12)
        self.open_btn = self._button(row, "📂 Open folder", self.open_output_folder,
                                     tip="Open the save folder in your file browser.")
        self.open_btn.pack(side=tk.RIGHT)

        # The filename BEFORE anything is written, so the mute decision is visible as a
        # consequence rather than as a radio button whose meaning has to be remembered.
        self.name_label = tk.Label(c, text="", font=("Consolas", 9), justify=tk.LEFT,
                                   bg=COLORS["bg_surface"], fg=COLORS["text_muted"])
        self.name_label.pack(anchor="w", pady=(8, 0))
        ToolTip(self.name_label,
                "What Add would queue right now. The number steps past both the files already in "
                "the folder and anything else waiting in the queue.")

        self.queue_box = tk.Listbox(c, height=7, bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
                                    font=("Consolas", 9), relief=tk.FLAT, highlightthickness=1,
                                    highlightbackground=COLORS["border"],
                                    selectbackground=COLORS["accent"])
        self.queue_box.pack(fill=tk.X, pady=(10, 0))
        self.queue_box.bind("<Double-Button-1>", lambda _e: self.recall_selected())
        ToolTip(self.queue_box,
                "The queue. ✓ has been written, · is still waiting.\n\n"
                "Double-click a waiting row to put the playhead and settings back where they "
                "were when you added it. Gizmo keeps no project file — when you have what you "
                "need, close it.")

        qrow = tk.Frame(c, bg=COLORS["bg_surface"])
        qrow.pack(fill=tk.X, pady=(6, 0))
        self._button(qrow, "Remove selected", self.remove_selected, pad=10,
                     tip="Drop the selected row from the queue. Already-exported clips stay on "
                         "disk — delete those in your file browser.").pack(side=tk.LEFT)
        self._button(qrow, "Clear queue", self.clear_queue, pad=10,
                     tip="Empty the waiting list. Nothing already exported is "
                         "touched.").pack(side=tk.LEFT, padx=(8, 0))

    # -- state ----------------------------------------------------------------------------------
    def _length_label(self, frames):
        return f"{frames} frames — {frames / FPS:.2f} s"

    def _frames(self):
        return GRID_FRAMES[self.len_box.current()] if self.len_box.current() >= 0 else 22

    def _size(self):
        """The export size: the crop at its own resolution, or the whole frame at its own.

        There is no size CHOICE any more. Fizgig rescales clips down to its Target Megapixels at
        training time, so cutting at native keeps that decision open for every run afterwards,
        while cutting small forecloses it and can only be undone by re-cutting the whole set.
        Nothing is ever scaled up, so this is only ever the source's own detail.
        """
        if not self.info:
            return (None, None)
        if self.crop:
            return (self.crop[2], self.crop[3])
        # The /32 pair nearest the source's own shape and area — the aspect search with the
        # target clamped to what the source holds, which is exactly "as big as it goes".
        return target_size(self.info["display_width"], self.info["display_height"], 99.0)

    def _keep_every(self):
        """None for real time, else keep every k-th source frame. Read from a list held beside
        the dropdown rather than parsed back out of its label — the label is prose and 'keep
        every frame' has no number in it to parse."""
        i = self.motion_box.current()
        return self._motion_keep[i] if 0 <= i < len(self._motion_keep) else None

    def _set_enabled(self, on):
        state = tk.NORMAL if on else tk.DISABLED
        for w in (self.add_btn, self.export_btn, self.scale, self.open_btn, self.pos_entry,
                  self.crop_clear_btn, *self._radios):
            w.configure(state=state)
        for box in (self.len_box, self.motion_box, self.shape_box):
            box.configure(state="readonly" if on else tk.DISABLED)
        # Listening needs a track to listen to AND a way to play it. Both reasons are spelled out
        # in _fill_sound_note rather than left as a greyed button with no explanation.
        can_play = bool(on and self.info and self.info["has_audio"]
                        and (audio_playback_backend() or ""))
        self.play_btn.configure(state=tk.NORMAL if can_play else tk.DISABLED)
        self.volume_scale.configure(state=tk.NORMAL if can_play else tk.DISABLED)

    # -- keyboard ---------------------------------------------------------------------------------
    def _bind_keys(self):
        """Frame-accurate marking with a mouse is miserable, and this is a tool you use thirty
        times in a row. Bound on the root so they work wherever focus happens to be — except
        inside the output-folder Entry, where a left arrow has to mean 'move the cursor'."""
        # Each key routes by the active tab: on Voice/audio the arrows nudge the playhead and
        # Ctrl+S marks a segment, on Video clips they keep their original meanings.
        for seq, fn, afn in (("<Left>", lambda: self.step(-1), lambda: self._audio_nudge(-0.05)),
                             ("<Right>", lambda: self.step(1), lambda: self._audio_nudge(0.05)),
                             ("<Shift-Left>", lambda: self.step(-10),
                              lambda: self._audio_nudge(-0.5)),
                             ("<Shift-Right>", lambda: self.step(10),
                              lambda: self._audio_nudge(0.5)),
                             ("<Home>", self.go_to_start, lambda: self._audio_seek(0.0)),
                             ("<Control-s>", self.add_to_queue, None)):
            def _route(_e, f=fn, af=afn):
                if self._audio_tab_active():
                    (af or self.audio_add_to_queue)()
                else:
                    f()
                return "break"
            self.root.bind(seq, _route)

        # Transport keys on both tabs: space = play/pause/resume, and the editing-suite J-K-L
        # shuttle. On Voice, J/L shuttle backward/forward (tap again for 1.5/2/3x); on Video —
        # which has no moving playhead to shuttle — J restarts the section's sound, L plays or
        # resumes it. The focus guard leaves a focused Button or Listbox its own space/keys;
        # text fields are already shielded by bindtags.
        def _video_l():
            if not self._playing:
                self._video_space()

        def _video_j():
            self._stop_audio()
            self._video_sound = None
            self._video_play_sound(0.0)

        for seq, afn, vfn in (("<space>", self._audio_space, self._video_space),
                              ("<j>", lambda: self._audio_shuttle(-1), _video_j),
                              ("<k>", self._audio_pause, self._video_pause),
                              ("<l>", lambda: self._audio_shuttle(1), _video_l)):
            def _troute(_e, af=afn, vf=vfn):
                # Text fields are shielded by bindtags and never reach here; this guard is the
                # belt-and-braces for a future unshielded one. Buttons no longer take focus at
                # all (see _button), so the transport keys always work.
                focus = self.root.focus_get()
                if focus is not None and focus.winfo_class() in ("Entry", "TEntry", "Text"):
                    return None
                (af if self._audio_tab_active() else vf)()
                return "break"
            self.root.bind(seq, _troute)

    def _audio_tab_active(self):
        try:
            return self.notebook.select() == str(self._audio_tab)
        except Exception:
            return False

    @staticmethod
    def _shield_from_hotkeys(widget):
        """Stop root-level hotkeys reaching a text field.

        Tk delivers a key along the widget's bindtags — widget, class, toplevel, all — so a hotkey
        bound on the root fires even while someone is typing, and ← would scrub the video instead
        of moving the cursor. Dropping the toplevel tag is the fix; the class bindings that make
        an Entry an Entry are untouched.

        This is done by bindtags rather than by inspecting the event, because event.widget reports
        the TOPLEVEL for a key that arrived at a child — so the obvious check silently never
        matches, which is exactly the kind of guard that looks right and does nothing.
        """
        widget.bindtags((str(widget), widget.winfo_class(), "all"))

    def go_to_start(self):
        if self.src:
            self.scale.set(0)

    # Length and motion both move the END of the clip, so both have to redraw the last-frame
    # preview as well as the cost line.
    def _on_length_changed(self):
        self._refresh_cost()
        self.show_frame()

    def _on_motion_changed(self):
        self._update_pos_labels()
        self.show_frame()

    # -- opening ---------------------------------------------------------------------------------
    def open_video(self):
        path = filedialog.askopenfilename(
            title="Open a source video",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.mts *.wmv"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            self.load_video(path)
        except Exception as exc:
            messagebox.showerror("Gizmo — cannot read that file", str(exc))

    def open_path(self, path):
        """Open a file that arrived from somewhere other than the Open button — a drop, the
        command line, or the clipboard. Refuses politely rather than by traceback."""
        path = str(path).strip().strip('"')
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Gizmo", f"Not a file:\n{path}")
            return
        if os.path.splitext(path)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            messagebox.showinfo(
                "Gizmo — that is a still",
                "Gizmo cuts clips out of video. Still images go straight into your training "
                "folder and are prepared on Fizgig's Image Prep tab.")
            return
        # An audio file goes to the Voice tab wherever you are. A VIDEO follows the tab you are
        # on: dropped while the Voice tab is up, it is an audio source (you asked for its sound
        # by being here) — anywhere else it opens as footage, as before.
        if is_audio_file(path) or self._audio_tab_active():
            self.notebook.select(self._audio_tab)
            try:
                self.load_audio(path)
            except Exception as exc:
                messagebox.showerror("Gizmo — cannot read that audio", str(exc))
            return
        try:
            self.load_video(path)
        except Exception as exc:
            messagebox.showerror("Gizmo — cannot read that file", str(exc))

    def paste_path(self):
        try:
            self.open_path(self.root.clipboard_get())
        except tk.TclError:
            pass                              # empty or non-text clipboard

    def load_video(self, path):
        """Everything opening a video does, with no dialog in the way — so it can be tested."""
        info = probe_source(self.ffmpeg, path)
        self._stop_audio()
        # A crop belongs to the frame it was drawn on, and a new video is a new frame.
        self.crop = None
        self.crop_var.set(False)
        self.src, self.info = path, info
        self.root.title(f"Gizmo — {os.path.basename(path)}")
        self.frame_pos = 0
        total = int((info["duration"] or 0) * info["fps"])
        self.scale.configure(from_=0, to=max(1, total - 1))
        self.scale.set(0)
        self.src_label.configure(text=os.path.basename(path), fg=COLORS["text_primary"])

        if info["has_audio"]:
            layout = {1: "mono", 2: "stereo"}.get(info["channels"], f"{info['channels']} ch")
            audio = f"{info['sample_rate']} Hz {layout}"
        else:
            audio = "no audio track"
        # The size as it will be SEEN. A rotated or anamorphic file is stored at something else,
        # and reporting the stored numbers here would contradict the picture right beside them.
        facts = (f"{info['display_width']}x{info['display_height']}  ·  {info['fps']:g} fps  ·  "
                 f"{info['duration'] or 0:.1f} s  ·  {audio}")
        if info["rotation"]:
            facts += f"  ·  rotated {info['rotation']}° (stored {info['width']}x{info['height']})"
        # A source slower than 24 fps cannot be resampled up honestly: ffmpeg duplicates frames
        # to reach 24, and the model learns that nothing moves between them. Worth saying out
        # loud rather than silently producing it — animation at 12 fps is the usual case, and it
        # is a legitimate thing to train on as long as the person knows what they are getting.
        if info["fps"] < FPS - 0.1:
            self.src_info.configure(
                text=facts + f"\n⚠ Under {FPS} fps: reaching {FPS} means duplicating frames, so "
                             f"some of what H3 learns about motion will be 'nothing happened'. "
                             f"Fine for animation you want that way; not what you want from "
                             f"under-cranked live footage.",
                fg=COLORS["warning"])
        else:
            self.src_info.configure(text=facts, fg=COLORS["text_explain"])

        if not self.out_var.get():
            self.out_var.set(os.path.join(os.path.dirname(path), "fizgig_clips"))
        self._fill_motion()
        self._fill_sound_note()
        self._set_enabled(True)
        self.show_frame()
        self._refresh_cost()
        self._refresh_planned_name()

    def _fill_motion(self):
        fps = self.info["fps"]
        opts = [f"Real time — resample {fps:g} fps to {FPS}"]
        self._motion_keep = [None]
        # Only offered where it would actually be visible: 60 fps gives 1.25x and 2.5x, 50 fps
        # gives 2.08x, and 30 fps gives nothing worth a dropdown entry.
        for k in (2, 1):
            factor = fps / k / FPS
            if factor > 1.05:
                every = "every frame" if k == 1 else f"every {k}nd frame" if k == 2 \
                    else f"every {k}th frame"
                opts.append(f"{factor:.2f}x slow motion — keep {every}")
                self._motion_keep.append(k)
        self.motion_box.configure(values=opts)
        self.motion_box.current(0)

    def _fill_sound_note(self):
        info = self.info
        if not info["has_audio"]:
            self.mute_var.set(True)
            self.sound_note.configure(
                text="This source has no audio track, so there is nothing to train on either way.",
                fg=COLORS["text_muted"])
            return
        note = ("Muting adds _mute to the filename and Fizgig trains that clip's video only. The "
                "audio stays in the file, so you can still play it back and change your mind by "
                "renaming.")
        if not (audio_playback_backend() or ""):
            # Almost always a pod: the container has no audio device and no player. Worth naming,
            # because a greyed Play button with no reason reads as a bug.
            note += ("\nPlayback is not available on this machine, so the Play button is off — "
                     "everything else works normally.")
        self.sound_note.configure(text=note, fg=COLORS["text_explain"])

    # -- scrubbing --------------------------------------------------------------------------------
    def _on_scrub(self, value):
        if not self.src:
            return
        self.frame_pos = int(float(value))
        self._update_pos_labels()
        # Debounced: dragging the slider fires this continuously and each frame is an ffmpeg call.
        if self._scrub_job:
            self.root.after_cancel(self._scrub_job)
        self._scrub_job = self.root.after(120, self.show_frame)

    def step(self, delta):
        if not self.src:
            return
        self.scale.set(max(0, self.frame_pos + delta))

    def _last_frame(self):
        """The last frame that exists in the source, as an index."""
        return max(0, int((self.info["duration"] or 0) * self.info["fps"]) - 1)

    def _go_to_typed(self):
        """Jump to a typed frame — or a typed time, with an s on the end.

        Clamped rather than refused: someone typing 9999 into a 600-frame video means 'the end',
        and an error dialog for that is a worse answer than going there. A value that is not a
        number at all just snaps back to where the playhead already is.
        """
        if not self.src:
            return
        raw = self.pos_var.get().strip().lower().replace(",", "")
        try:
            if raw.endswith("s"):
                frame = int(round(float(raw[:-1]) * self.info["fps"]))
            else:
                frame = int(round(float(raw)))
        except ValueError:
            self._sync_pos_entry()
            return
        frame = max(0, min(frame, self._last_frame()))
        if frame != self.frame_pos:
            self.scale.set(frame)
        self._sync_pos_entry()

    def _sync_pos_entry(self):
        """Keep the box showing the truth — it is an input AND a readout, and after a drag the
        two disagree unless this runs."""
        if self.pos_var.get() != str(self.frame_pos):
            self.pos_var.set(str(self.frame_pos))

    def _update_pos_labels(self):
        fps = self.info["fps"]
        t = self.frame_pos / fps
        self._sync_pos_entry()
        self.pos_label.configure(
            text=f"of {self._last_frame()}   ·   {t:6.2f} s")
        frames = self._frames()
        k = self._keep_every()
        # How much SOURCE this clip eats depends on the motion choice: real time takes the same
        # wall-clock span, slow motion takes k source frames per output frame.
        span = frames * k / fps if k else frames / FPS
        end, duration = t + span, self.info["duration"] or 0
        text = f"clip covers {t:.2f} s → {end:.2f} s of the source"
        # Caught here as well as after the export, because a run-out is easy to walk into by
        # scrubbing near the end and the export only discovers it after encoding.
        if duration and end > duration:
            self.span_label.configure(
                text=text + f" — but the source ends at {duration:.2f} s. Move earlier, or "
                            f"choose a shorter length.",
                fg=COLORS["warning"])
        else:
            self.span_label.configure(text=text, fg=COLORS["text_explain"])

    def show_frame(self):
        self._scrub_job = None
        if not self.src:
            return
        fps = self.info["fps"]
        start = self.frame_pos / fps
        k = self._keep_every()
        span = self._frames() * k / fps if k else self._frames() / FPS
        # The last frame of the clip, not the frame after it — hence the one-frame step back.
        threading.Thread(target=self._grab_worker,
                         args=(start, max(start, start + span - 1 / fps)),
                         daemon=True).start()

    def _grab_worker(self, start_s, end_s):
        size = (self.info["display_width"], self.info["display_height"])
        try:
            first = grab_frame(self.ffmpeg, self.src, start_s, src_size=size)
        except Exception:
            first = None
        try:
            last = grab_frame(self.ffmpeg, self.src, end_s, box=(END_W, END_H), src_size=size)
        except Exception:
            last = None
        self.root.after(0, self._show_images, first, last)

    def _show_images(self, first, last):
        self.photo = self._paint(self.canvas, first, PREVIEW_W, PREVIEW_H)
        self.end_photo = self._paint(self.end_canvas, last, END_W, END_H)
        self._draw_crop_overlay()

    def _paint(self, canvas, img, w, h):
        """Draw into a canvas and hand back the PhotoImage — which the caller MUST keep, or Tk
        garbage-collects it and the canvas goes blank."""
        canvas.delete("all")
        if img is None:
            canvas.create_text(w // 2, h // 2, text="(no frame here)",
                               fill=COLORS["text_muted"], font=(FONT_FAMILY, 10))
            canvas._shown = None
            return None
        photo = ImageTk.PhotoImage(img)
        canvas.create_image(w // 2, h // 2, image=photo)
        # Remembered so canvas coordinates can be turned back into source ones: the image is
        # centred, so it sits at an offset that depends on its shape.
        canvas._shown = (img.size[0], img.size[1],
                         (w - img.size[0]) // 2, (h - img.size[1]) // 2)
        return photo

    # -- cropping ---------------------------------------------------------------------------------
    def _canvas_to_source(self, canvas, cx, cy):
        """A point on a preview canvas -> a point in the source's display pixels."""
        shown = getattr(canvas, "_shown", None)
        if not shown or not self.info:
            return None
        iw, ih, ox, oy = shown
        sx = (cx - ox) / iw * self.info["display_width"]
        sy = (cy - oy) / ih * self.info["display_height"]
        return sx, sy

    def _source_to_canvas(self, canvas, sx, sy):
        shown = getattr(canvas, "_shown", None)
        if not shown or not self.info:
            return None
        iw, ih, ox, oy = shown
        return (ox + sx / self.info["display_width"] * iw,
                oy + sy / self.info["display_height"] * ih)

    def _crop_press(self, event):
        if not self.src or not self.crop_var.get():
            return
        pt = self._canvas_to_source(self.canvas, event.x, event.y)
        if pt:
            self._crop_anchor = pt

    def _crop_drag(self, event):
        if not getattr(self, "_crop_anchor", None):
            return
        pt = self._canvas_to_source(self.canvas, event.x, event.y)
        if not pt:
            return
        self.crop = self._snap_crop(self._crop_anchor, pt)
        self._draw_crop_overlay()

    def _crop_release(self, _event):
        if not getattr(self, "_crop_anchor", None):
            return
        self._crop_anchor = None
        if self.crop:
            # The crop decides the shape AND how many pixels there are to work with, so the size
            # menu is rebuilt from it — keeping whichever tier was chosen rather than the pixel
            # dimensions, which have just changed under it.
            self._refresh_cost()
            self.show_frame()
        self._describe_crop()

    def _crop_aspect(self):
        """The locked aspect, or None for free-form."""
        i = self.shape_box.current()
        value = CROP_SHAPES[i][1] if 0 <= i < len(CROP_SHAPES) else None
        if value == "source" and self.info:
            return self.info["display_width"] / self.info["display_height"]
        return value if isinstance(value, float) else None

    def _snap_crop(self, a, b):
        """Two dragged corners -> an (x, y, w, h) box on the /32 grid, inside the frame.

        Snapped because H3 needs multiples of 32 and because a crop that is 31 pixels off the
        grid would otherwise be quietly rounded later, moving the framing away from what was
        drawn. Anything smaller than 32 in either direction is not a crop, it is a slip.

        With a shape locked, the /32 pair nearest that ratio is searched for rather than computed:
        the grid is coarse enough that dividing and rounding lands surprisingly far off — 16:9 at
        a height of 352 rounds to 640x352, which is 1.82. Both sides move, so the result is the
        closest the grid can actually get.
        """
        dw, dh = self.info["display_width"], self.info["display_height"]
        max_w = int(dw // SIZE_STEP) * SIZE_STEP
        max_h = int(dh // SIZE_STEP) * SIZE_STEP
        x0, x1 = sorted((max(0.0, min(a[0], dw)), max(0.0, min(b[0], dw))))
        y0, y1 = sorted((max(0.0, min(a[1], dh)), max(0.0, min(b[1], dh))))
        x = int(x0 // SIZE_STEP) * SIZE_STEP
        y = int(y0 // SIZE_STEP) * SIZE_STEP
        w = max(SIZE_STEP, int(round((x1 - x) / SIZE_STEP)) * SIZE_STEP)
        h = max(SIZE_STEP, int(round((y1 - y) / SIZE_STEP)) * SIZE_STEP)

        aspect = self._crop_aspect()
        if aspect:
            # Fitted INSIDE what was dragged, so the rectangle never grows past the gesture, then
            # scored on how close the grid can get. Ties go to the larger box.
            span_w = min(w, max_w - x)
            span_h = min(h, max_h - y)
            best = None
            for cw in range(SIZE_STEP, span_w + SIZE_STEP, SIZE_STEP):
                for ch in {int(round(cw / aspect / SIZE_STEP)) * SIZE_STEP,
                           int(round(cw / aspect / SIZE_STEP)) * SIZE_STEP + SIZE_STEP}:
                    if ch < SIZE_STEP or ch > span_h:
                        continue
                    score = (round(abs((cw / ch) - aspect) / aspect, 4), -(cw * ch))
                    if best is None or score < best[0]:
                        best = (score, (cw, ch))
            if best is None:
                return None
            w, h = best[1]

        w = min(w, max_w - x)
        h = min(h, max_h - y)
        if w < SIZE_STEP or h < SIZE_STEP:
            return None
        return (x, y, w, h)

    def _draw_crop_overlay(self):
        """Outline the crop on BOTH previews. The last frame is the point: a crop holds still and
        a subject does not, so whether they are still inside at the end is only visible there."""
        for canvas in (self.canvas, self.end_canvas):
            canvas.delete("crop")
            if not self.crop or not getattr(canvas, "_shown", None):
                continue
            x, y, w, h = self.crop
            p0 = self._source_to_canvas(canvas, x, y)
            p1 = self._source_to_canvas(canvas, x + w, y + h)
            if not p0 or not p1:
                continue
            iw, ih, ox, oy = canvas._shown
            # Dim what is being thrown away rather than only ringing what is kept — the discarded
            # part is the thing worth seeing.
            for box in ((ox, oy, ox + iw, p0[1]), (ox, p1[1], ox + iw, oy + ih),
                        (ox, p0[1], p0[0], p1[1]), (p1[0], p0[1], ox + iw, p1[1])):
                canvas.create_rectangle(*box, fill="#000000", outline="", stipple="gray50",
                                        tags="crop")
            canvas.create_rectangle(*p0, *p1, outline=COLORS["accent"], width=2, tags="crop")

    def _on_shape_changed(self):
        """Re-fit an existing crop to the new shape, around the same centre — otherwise picking a
        shape does nothing visible until the next drag, which reads as the control being broken."""
        if not self.src:
            return
        if self.crop:
            x, y, w, h = self.crop
            cx, cy = x + w / 2, y + h / 2
            refit = self._snap_crop((cx - w / 2, cy - h / 2), (cx + w / 2, cy + h / 2))
            if refit:
                # Re-centre: the snap anchors at the top-left corner, so a shape change would
                # otherwise slide the framing down and right.
                nx, ny, nw, nh = refit
                nx = max(0, min(int((cx - nw / 2) // SIZE_STEP) * SIZE_STEP,
                                int(self.info["display_width"] // SIZE_STEP) * SIZE_STEP - nw))
                ny = max(0, min(int((cy - nh / 2) // SIZE_STEP) * SIZE_STEP,
                                int(self.info["display_height"] // SIZE_STEP) * SIZE_STEP - nh))
                self.crop = (nx, ny, nw, nh)
            self._refresh_cost()
            self._draw_crop_overlay()
            self.show_frame()
        self._describe_crop()

    def _on_crop_mode_changed(self):
        if not self.crop_var.get():
            self.clear_crop()
            return
        self._describe_crop()

    def clear_crop(self):
        self.crop = None
        self.crop_var.set(False)
        self._crop_anchor = None
        if self.info:
            self._refresh_cost()
        self._draw_crop_overlay()
        self._describe_crop()
        self._refresh_planned_name()

    def _describe_crop(self):
        if not self.crop_var.get():
            self.crop_note.configure(text="", fg=COLORS["text_explain"])
            return
        if not self.crop:
            self.crop_note.configure(
                text="Drag a rectangle on the FIRST FRAME preview below.",
                fg=COLORS["accent"])
            return
        x, y, w, h = self.crop
        share = (w * h) / (self.info["display_width"] * self.info["display_height"]) * 100
        text = f"Cropping to {w} x {h} at ({x}, {y}) — {share:.0f}% of the frame."
        aspect = self._crop_aspect()
        if aspect:
            # What the lock actually managed, not what was asked for. The /32 grid cannot hit
            # every ratio, and quietly reporting the requested one would be a small lie.
            got = w / h
            text += (f" Shape {got:.3f}"
                     + ("" if abs(got - aspect) / aspect < 0.005
                        else f" — the closest to {aspect:.3f} the 32-pixel grid allows"))
        tw, th = self._size()
        # Cropped below the output size means the clip is upscaled-in-effect: there is simply
        # less real detail behind the same number of pixels. Worth saying, not worth refusing.
        if tw and (w < tw or h < th):
            self.crop_note.configure(
                text=text + f" That is smaller than the {tw} x {th} output, so the clip will be "
                            f"softer than the source — crop wider, or pick a smaller size.",
                fg=COLORS["warning"])
        else:
            self.crop_note.configure(text=text, fg=COLORS["text_explain"])

    # -- listening --------------------------------------------------------------------------------
    def toggle_play(self):
        """Play the marked section's audio, or stop it if it is already playing.

        The volume slider is applied by ffmpeg while extracting, because the only playback that
        needs no new dependency is winsound, and winsound has no volume of its own. That also
        keeps the promise the tooltip makes: the gain exists in a temp file for listening and
        never reaches the saved clip.
        """
        if self._playing:
            self._stop_audio()
            self._video_sound = None
            return
        self._video_sound = None
        self._video_play_sound(0.0)

    def _video_play_sound(self, offset):
        """Play the marked section's audio from `offset` seconds into it — 0 for the button,
        a stored consumed-time for space's resume."""
        if not self.src or not self.info or not self.info["has_audio"]:
            return
        fps = self.info["fps"]
        start = self.frame_pos / fps
        k = self._keep_every()
        # Real time: the audio you would get. Slow motion: the source audio under the section,
        # which is what you are judging — the saved clip's audio comes from the same span.
        span = self._frames() * k / fps if k else self._frames() / FPS
        if offset >= span - 0.05:
            offset = 0.0                       # a stale resume past the end starts over
        # Live volume when the backend allows it — the slider then works MID-play; otherwise
        # the gain is baked into the preview file as before.
        gain = 1.0 if set_live_volume(self.volume_var.get()) \
            else max(0.0, float(self.volume_var.get())) / 100.0
        self._video_offset = offset
        self.play_btn.configure(text="■  Stop")
        self._playing = True
        threading.Thread(target=self._play_worker,
                         args=(start + offset, span - offset, gain), daemon=True).start()

    def _video_pause(self):
        """Freeze the marked section's sound where it is; space resumes from there."""
        if not self._playing:
            return
        import time as _time
        consumed = self._video_offset + (_time.monotonic() - self._video_t0)
        self._stop_audio()
        self._video_sound = consumed
        self.status.configure(text=f"paused — space resumes", fg=COLORS["text_secondary"])

    def _video_space(self):
        """Space on the video tab: play the marked section's sound / pause / resume."""
        if self._playing:
            self._video_pause()
        elif self._video_sound is not None:
            offset, self._video_sound = self._video_sound, None
            self._video_play_sound(offset)
        else:
            self._video_play_sound(0.0)

    def _play_worker(self, start, span, gain):
        err = None
        try:
            wav = os.path.join(tempfile.gettempdir(), "gizmo_preview.wav")
            p = _run([self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                      "-ss", f"{start:.3f}", "-t", f"{span:.3f}", "-i", self.src,
                      "-vn", "-af", f"volume={gain:.3f}", "-ac", "2", "-ar", "44100",
                      "-c:a", "pcm_s16le", wav])
            if p.returncode != 0 or not os.path.exists(wav):
                raise RuntimeError((p.stderr or b"").decode("utf-8", "replace")[-300:]
                                   or "could not extract the audio")
            _play_wav(wav)
        except Exception as exc:
            err = str(exc)
        self.root.after(0, self._play_done, span, err)

    def _play_done(self, span, err):
        if err:
            self._playing = False
            self.play_btn.configure(text="▶  Play sound")
            self.status.configure(text="Could not play the sound", fg=COLORS["error"])
            messagebox.showerror("Gizmo — cannot play that audio", err)
            return
        # winsound plays asynchronously and never says when it finished, so the button is reset
        # on a timer the length of the clip. Under a second either way — nobody is counting.
        # The clock for pause's consumed-time starts here, when the sound actually starts.
        import time as _time
        self._video_t0 = _time.monotonic()
        self._play_job = self.root.after(int(span * 1000) + 200, self._stop_audio)

    def _stop_audio(self):
        if self._play_job is not None:
            try:
                self.root.after_cancel(self._play_job)
            except Exception:
                pass
            self._play_job = None
        _stop_wav()
        self._playing = False
        if self.play_btn.winfo_exists():
            self.play_btn.configure(text="▶  Play sound")

    # -- voice / audio mode -----------------------------------------------------------------------
    #
    # The same rhythm as the video side — open one file, mark the pieces you want, export the lot
    # — with the axes swapped: there is nothing to crop or retime, only WHERE a segment starts
    # and WHICH grid length it runs. Exports are wav, 32 kHz stereo, at exactly latents x 800
    # samples, each with its caption .txt beside it — so a segment dropped into a training folder
    # passes Fizgig's strict checks with nothing to configure.

    AUDIO_WAVE_W, AUDIO_WAVE_H = 940, 150

    def _build_audio_tab(self, body):
        # 1. source
        c = self._card(body, "1. Source recording",
                       "A voice memo, a podcast, an interview — any wav, mp3, flac or m4a. Or a "
                       "VIDEO file, to use just its audio track: nothing is shown, you scrub the "
                       "sound itself.")
        row = tk.Frame(c, bg=COLORS["bg_surface"])
        row.pack(fill=tk.X)
        self._button(row, "Open audio…", self.open_audio, "accent",
                     tip="Open a recording — or a video whose SOUND you want.\n\nAny format or "
                         "sample rate; Gizmo converts on export. The original is never "
                         "modified.").pack(side=tk.LEFT)
        self.audio_src_label = tk.Label(row, text="No recording open", font=(FONT_FAMILY, 10),
                                        bg=COLORS["bg_surface"], fg=COLORS["text_muted"])
        self.audio_src_label.pack(side=tk.LEFT, padx=12)
        tk.Label(row, text="…or drag a wav/mp3 onto this window",
                 font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"],
                 fg=COLORS["text_muted"]).pack(side=tk.RIGHT)

        # 2. caption + length — before the waveform, same doctrine as the video tab: decide what
        # a segment IS before hunting for where it starts.
        c = self._card(body, "2. Caption and length",
                       "The caption teaches the voice, so describe only the sound — tone, pace, "
                       "accent, mood. No visual claims: there is no picture for them to be true "
                       "of. Transcribe appends the spoken words, which helps the voice bind.")
        trow = tk.Frame(c, bg=COLORS["bg_surface"])
        trow.pack(fill=tk.X)
        tk.Label(trow, text="Trigger word:", font=(FONT_FAMILY, 10),
                 bg=COLORS["bg_surface"], fg=COLORS["text_primary"]).pack(side=tk.LEFT)
        self.audio_trigger_var = tk.StringVar(value="")
        trig = tk.Entry(trow, textvariable=self.audio_trigger_var, width=24,
                        bg=COLORS["bg_hover"], fg=COLORS["text_primary"],
                        insertbackground=COLORS["text_primary"], relief=tk.FLAT)
        trig.pack(side=tk.LEFT, padx=(8, 0), ipady=4)
        self._shield_from_hotkeys(trig)
        ToolTip(trig, "Leads every caption, exactly like the Captions tab's Qwen tool: "
                      "\"<trigger>, <description> saying …\". Set it once per recording.")
        self.audio_whisper_btn = self._button(
            trow, "🎤 Transcribe", self.audio_transcribe,
            tip="Listen to the marked segment with Whisper (a small speech-recognition model, "
                "~150 MB, downloaded once) and append the words it hears to the caption as "
                "saying \"…\". The description half stays yours — no model can hear that a "
                "voice is warm.")
        self.audio_whisper_btn.pack(side=tk.RIGHT)
        self.audio_lang_var = tk.StringVar(value="Auto detect")
        _lang = ttk.Combobox(trow, textvariable=self.audio_lang_var, width=11,
                             state="readonly", style="G.TCombobox",
                             values=["Auto detect", "English", "French", "German", "Spanish",
                                     "Italian", "Portuguese", "Dutch", "Polish", "Welsh",
                                     "Russian", "Ukrainian", "Japanese", "Chinese", "Korean",
                                     "Arabic", "Hindi"])
        _lang.pack(side=tk.RIGHT, padx=(0, 8))
        ToolTip(_lang, "The language Whisper listens FOR. Auto detect usually works, but a "
                       "short segment can fool it — it once heard Welsh in English and looped. "
                       "Pick the language and the guess is skipped entirely.")
        self.audio_caption = tk.Text(c, height=3, wrap=tk.WORD, bg=COLORS["bg_hover"],
                                     fg=COLORS["text_primary"], relief=tk.FLAT,
                                     insertbackground=COLORS["text_primary"],
                                     font=(FONT_FAMILY, 10))
        self.audio_caption.pack(fill=tk.X, pady=(8, 0))
        self._shield_from_hotkeys(self.audio_caption)
        ToolTip(self.audio_caption,
                "This segment's caption — saved as a .txt beside the exported wav. It stays put "
                "between Adds, so describe the voice once and only touch it when something "
                "changes (a different emotion, a different speaker).")

        lrow = tk.Frame(c, bg=COLORS["bg_surface"])
        lrow.pack(fill=tk.X, pady=(10, 0))
        tk.Label(lrow, text="Segment length:", font=(FONT_FAMILY, 10),
                 bg=COLORS["bg_surface"], fg=COLORS["text_primary"]).pack(side=tk.LEFT)
        self.audio_frames_var = tk.IntVar(value=124)
        for f in AUDIO_GRID_FRAMES:
            secs = hop_exact_samples(f) / AUDIO_SAMPLE_RATE
            rb = tk.Radiobutton(
                lrow, text=f"{secs:.1f} s", variable=self.audio_frames_var, value=f,
                command=self._audio_refresh, bg=COLORS["bg_surface"],
                fg=COLORS["text_primary"], selectcolor=COLORS["bg_deep"],
                activebackground=COLORS["bg_surface"], activeforeground=COLORS["text_primary"],
                font=(FONT_FAMILY, 10))
            rb.pack(side=tk.LEFT, padx=(10, 0))
            ToolTip(rb, f"Exactly {hop_exact_samples(f) / AUDIO_SAMPLE_RATE:.3f} s "
                        f"({audio_latents_for(f)} audio latents) — one of the grid lengths H3 "
                        f"trains on, the same clock as video clips. Longer segments carry more "
                        f"of the voice per item; 5.2 s is the ceiling the model packs.")

        # 3. the waveform
        c = self._card(body, "3. Find the moment",
                       "Play from here and just listen — click anywhere on the waveform and "
                       "playback jumps there, which is how you scrub sound. The shaded span is "
                       "what would export: grab it to slide it, drag either END to snap it to "
                       "the next allowed length, click outside it to place it, Play segment to "
                       "check it. Space is play / pause / resume, J-K-L shuttles like an edit "
                       "suite (tap J or L again for 1.5/2/3x), ←/→ nudge 50 ms, Shift 500 ms, "
                       "Home for the start.")
        self.audio_canvas = tk.Canvas(c, width=self.AUDIO_WAVE_W, height=self.AUDIO_WAVE_H,
                                      bg=COLORS["bg_deep"], highlightthickness=1,
                                      highlightbackground=COLORS["border"], cursor="crosshair")
        self.audio_canvas.pack()
        self.audio_canvas.bind("<Button-1>", self._audio_press)
        self.audio_canvas.bind("<B1-Motion>", self._audio_drag)
        self.audio_canvas.bind("<ButtonRelease-1>", self._audio_release)
        self.audio_canvas.bind("<Motion>", self._audio_hover)
        # Shift+wheel pans a zoomed view sideways (plain wheel zooms).
        self.audio_canvas.bind(
            "<Shift-MouseWheel>",
            lambda e: (self._audio_pan(-0.15 if e.delta > 0 else 0.15), "break")[1])
        # The nav bar under the waveform: where the zoomed window sits in the whole recording,
        # and the handle to slide it. Full-width (and inert) when not zoomed.
        self.audio_scroll = ttk.Scrollbar(c, orient=tk.HORIZONTAL, command=self._audio_xview)
        self.audio_scroll.pack(fill=tk.X, pady=(2, 0))
        ToolTip(self.audio_scroll,
                "Where the zoomed view sits in the whole recording — drag to scroll along it. "
                "Shift+wheel over the waveform pans too; plain wheel zooms.")
        # Wheel over the waveform zooms it, anchored on the mouse — the sound you are pointing
        # at stays under the pointer, map-style. Returning "break" keeps the page from
        # scrolling; everywhere else the wheel still scrolls the window.
        def _wheel_zoom(e):
            v0, vd = self._audio_view_bounds()
            self._audio_zoom(4.0 if e.delta > 0 else 0.25,
                             focus_t=v0 + e.x / self.AUDIO_WAVE_W * vd)
            return "break"
        self.audio_canvas.bind("<MouseWheel>", _wheel_zoom)
        prow = tk.Frame(c, bg=COLORS["bg_surface"])
        prow.pack(fill=tk.X, pady=(8, 0))
        self.audio_listen_btn = self._button(
            prow, "▶  Play from here", lambda: self.audio_toggle_play("file"),
            tip="Listen through the recording from the playhead — click the waveform while it "
                "plays to jump around. This is how you FIND the moment; Play segment is how "
                "you check it.")
        self.audio_listen_btn.pack(side=tk.LEFT)
        self.audio_play_btn = self._button(
            prow, "▶  Play segment", lambda: self.audio_toggle_play("segment"),
            tip="Hear exactly what would export — the marked span, nothing more.  (volume is "
                "listening only, never saved)")
        self.audio_play_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.audio_volume_var = tk.DoubleVar(value=80)
        _vol = ttk.Scale(prow, from_=0, to=100, orient=tk.HORIZONTAL, length=140,
                         style="G.Horizontal.TScale", variable=self.audio_volume_var,
                         command=lambda v: set_live_volume(v))
        _vol.pack(side=tk.LEFT, padx=(12, 0))
        ToolTip(_vol, "Listening volume only — moves live while a preview plays. The exported "
                      "segment's audio is never touched.")
        # Fine placement: zoom to see the detail, nudge to land on it. Both live beside the
        # play buttons because they are all one activity.
        for txt, fn, tip in (
                ("🔍−", lambda: self._audio_zoom(0.25), "Zoom out 4x. Fit shows the whole file."),
                ("🔍+", lambda: self._audio_zoom(4.0),
                 "Zoom in 4x around the playhead — one pixel gets 16x finer per press, which "
                 "is what makes trimming a breath off a segment start possible."),
                ("Fit", lambda: self._audio_zoom(None), "Show the whole recording again."),
                ("−0.5s", lambda: self._audio_nudge(-0.5), "Nudge the segment half a second "
                                                           "earlier.  (Shift+←)"),
                ("−50ms", lambda: self._audio_nudge(-0.05), "Nudge the segment 50 ms earlier — "
                                                            "restarts playback if something is "
                                                            "playing.  (←)"),
                ("+50ms", lambda: self._audio_nudge(0.05), "Nudge the segment 50 ms later.  (→)"),
                ("+0.5s", lambda: self._audio_nudge(0.5), "Nudge the segment half a second "
                                                          "later.  (Shift+→)")):
            self._button(prow, txt, fn, pad=8, tip=tip).pack(side=tk.LEFT, padx=(6, 0))
        # Its own line UNDER the buttons — in the button row it fought a dozen controls for
        # width and lost, showing up truncated or not at all.
        self.audio_pos_label = tk.Label(c, text="", font=("Consolas", 10),
                                        bg=COLORS["bg_surface"], fg=COLORS["text_secondary"])
        self.audio_pos_label.pack(anchor=tk.W, pady=(6, 0))

        # 4. queue + export
        c = self._card(body, "4. Mark it, then move on",
                       "Add every segment you want from this recording, then export the lot — "
                       "each wav lands with its caption .txt beside it, ready for the training "
                       "folder.")
        orow = tk.Frame(c, bg=COLORS["bg_surface"])
        orow.pack(fill=tk.X)
        tk.Label(orow, text="Save to:", font=(FONT_FAMILY, 10),
                 bg=COLORS["bg_surface"], fg=COLORS["text_primary"]).pack(side=tk.LEFT)
        self.audio_out_var = tk.StringVar(value="")
        oe = tk.Entry(orow, textvariable=self.audio_out_var, bg=COLORS["bg_hover"],
                      fg=COLORS["text_primary"], insertbackground=COLORS["text_primary"],
                      relief=tk.FLAT)
        oe.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8), ipady=4)
        self._shield_from_hotkeys(oe)
        ToolTip(oe, "Where the segments and their caption .txt files land — point it at your "
                    "training folder to skip a copy step.")
        self._button(orow, "Browse…", self._audio_pick_output, pad=10,
                     tip="Pick the save folder.").pack(side=tk.LEFT)

        qrow = tk.Frame(c, bg=COLORS["bg_surface"])
        qrow.pack(fill=tk.X, pady=(10, 0))
        self.audio_add_btn = self._button(
            qrow, "➕  Add to queue", self.audio_add_to_queue, "accent",
            tip="Remember this segment — where it starts, its length, its caption.  (Ctrl+S)\n\n"
                "Nothing is written yet; keep marking.")
        self.audio_add_btn.pack(side=tk.LEFT)
        self.audio_export_btn = self._button(
            qrow, "Export queue", self.audio_export_queue,
            tip="Write every queued segment: 32 kHz stereo wav at the exact grid length, caption "
                ".txt beside it. Fizgig accepts them as-is.")
        self.audio_export_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.audio_status = tk.Label(qrow, text="", font=(FONT_FAMILY, 10),
                                     bg=COLORS["bg_surface"], fg=COLORS["text_secondary"])
        self.audio_status.pack(side=tk.LEFT, padx=12)
        self._button(qrow, "📂 Open folder", self._audio_open_folder,
                     tip="Open the save folder in your file browser.").pack(side=tk.RIGHT)

        self.audio_queue_box = tk.Listbox(
            c, height=6, bg=COLORS["bg_deep"], fg=COLORS["text_primary"], font=("Consolas", 9),
            relief=tk.FLAT, highlightthickness=1, highlightbackground=COLORS["border"],
            selectbackground=COLORS["accent"])
        self.audio_queue_box.pack(fill=tk.X, pady=(10, 0))
        self.audio_queue_box.bind("<Double-Button-1>", lambda _e: self._audio_recall())
        ToolTip(self.audio_queue_box,
                "The queue. ✓ has been written, · is still waiting.\n\nDouble-click a waiting "
                "row to put the playhead, length and caption back where they were.")
        brow = tk.Frame(c, bg=COLORS["bg_surface"])
        brow.pack(fill=tk.X, pady=(6, 0))
        self._button(brow, "Remove selected", self._audio_remove_selected, pad=10,
                     tip="Drop the selected row from the queue. Already-exported segments stay "
                         "on disk — delete those in your file browser.").pack(side=tk.LEFT)
        self._button(brow, "Clear queue", self._audio_clear_queue, pad=10,
                     tip="Empty the waiting list. Nothing already exported is "
                         "touched.").pack(side=tk.LEFT, padx=(8, 0))

        self._audio_set_enabled(False)

    # -- audio: opening -----------------------------------------------------------------------------
    def open_audio(self):
        path = filedialog.askopenfilename(
            title="Open a recording — or a video, for its audio track",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.m4a"),
                       ("Video files (audio track)", "*.mp4 *.mov *.mkv *.avi *.m4v *.webm"),
                       ("All files", "*.*")])
        if not path:
            return
        self.notebook.select(self._audio_tab)
        try:
            self.load_audio(path)
        except Exception as exc:
            messagebox.showerror("Gizmo — cannot read that audio", str(exc))

    def load_audio(self, path):
        """Open a recording (or a video's audio track): probe duration, decode peaks in the
        background, arm the tab."""
        self._audio_stop()
        # Duration by decoding to null — the one number the container cannot lie about, and it
        # covers every format the same way. A few hundred ms even for long files.
        p = _run([self.ffmpeg, "-hide_banner", "-i", path, "-vn",
                  "-f", "null", os.devnull])
        m = None
        for m in re.finditer(r"time=(\d+):(\d\d):(\d\d(?:\.\d+)?)",
                             (p.stderr or b"").decode("utf-8", "replace")):
            pass
        if m is None:
            raise RuntimeError(f"{os.path.basename(path)}: no decodable audio in this file"
                               + (" — it has a picture but no sound"
                                  if not is_audio_file(path) else ""))
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        min_dur = hop_exact_samples(AUDIO_GRID_FRAMES[0]) / AUDIO_SAMPLE_RATE
        if dur < min_dur:
            raise RuntimeError(f"{os.path.basename(path)}: {dur:.2f} s of audio — the shortest "
                               f"segment H3 trains on is {min_dur:.1f} s.")
        self.audio_src = path
        self.audio_dur = dur
        self.audio_from_video = not is_audio_file(path)
        self.audio_pos = 0.0
        self.audio_peaks = None
        self._audio_samples = None
        self._audio_peaks_key = None
        self.audio_view = None
        self.audio_src_label.configure(
            text=f"{os.path.basename(path)} — {dur:.1f} s"
                 + ("  (a video's audio track)" if self.audio_from_video else ""),
            fg=COLORS["text_primary"])
        if not self.audio_out_var.get():
            self.audio_out_var.set(os.path.join(os.path.dirname(path), "fizgig_voice"))
        self._audio_set_enabled(True)
        self._audio_refresh()
        self.audio_status.configure(text="reading the waveform…", fg=COLORS["text_secondary"])
        threading.Thread(target=self._audio_peaks_worker, args=(path,), daemon=True).start()

    _AUDIO_PEAK_RATE = 8000

    def _audio_peaks_worker(self, path):
        """Decode to 8 kHz mono f32 and KEEP the samples — zoom re-slices them into per-column
        (lo, hi) pairs on demand. array-module slices keep every reduction at C speed with no
        numpy dependency. The normalisation is computed once over the whole file, so zooming
        into a quiet stretch shows it quiet instead of rescaling it to fill the canvas."""
        import array
        try:
            p = _run([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path, "-vn",
                      "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
                      "-ar", str(self._AUDIO_PEAK_RATE), "-"])
            samples = array.array("f")
            samples.frombytes(p.stdout or b"")
            if not samples:
                raise RuntimeError("decoded no samples")
            scale = max(1e-6, max(samples), -min(samples))
        except Exception as exc:
            err = str(exc)
            self.root.after(0, lambda: (self.audio_status.configure(
                text=f"waveform failed: {err[:60]}", fg=COLORS["error"])))
            return
        def done():
            if self.audio_src == path:          # not superseded by another open
                self._audio_samples = samples
                self._audio_peak_scale = scale
                self._audio_peaks_key = None
                self.audio_status.configure(text="", fg=COLORS["text_secondary"])
                self._audio_refresh()
        self.root.after(0, done)

    def _audio_view_bounds(self):
        """(start_s, dur_s) of what the canvas shows — the whole file unless zoomed."""
        if self.audio_view is None:
            return 0.0, max(self.audio_dur, 1e-6)
        return self.audio_view

    def _audio_slice_peaks(self):
        """Per-column (lo, hi) for the current view, cached until the view or samples change."""
        if self._audio_samples is None:
            self.audio_peaks = None
            return
        v0, vd = self._audio_view_bounds()
        key = (round(v0, 4), round(vd, 4), len(self._audio_samples))
        if key == self._audio_peaks_key:
            return
        s = self._audio_samples
        r = self._AUDIO_PEAK_RATE
        lo_i = max(0, int(v0 * r))
        n = max(1, int(vd * r))
        n_cols = self.AUDIO_WAVE_W
        step = max(1, n // n_cols)
        peaks = []
        for i in range(n_cols):
            chunk = s[lo_i + i * step: lo_i + (i + 1) * step]
            peaks.append((min(chunk) / self._audio_peak_scale,
                          max(chunk) / self._audio_peak_scale) if chunk else (0.0, 0.0))
        self.audio_peaks = peaks
        self._audio_peaks_key = key

    # -- audio: the waveform ------------------------------------------------------------------------
    def _audio_seek(self, seconds):
        if not self.audio_src:
            return
        span = self._audio_span()
        self.audio_pos = max(0.0, min(float(seconds), self.audio_dur - span))
        self._audio_resume = None     # a moved playhead invalidates a paused position
        # Zoomed in and the playhead left the window: follow it, keeping a fifth of margin.
        # NEVER while a drag is in progress — the pan would shift the view under the cursor,
        # the same mouse position would map to a new time, and that feedback loop flings the
        # segment across the whole recording on the slightest move.
        if self.audio_view is not None and not getattr(self, "_audio_dragging", False):
            v0, vd = self.audio_view
            if not (v0 <= self.audio_pos <= v0 + vd * 0.95):
                v0 = max(0.0, min(self.audio_pos - vd * 0.2, self.audio_dur - vd))
                self.audio_view = (v0, vd)
        self._audio_refresh()
        self._audio_arm_rescrub()

    def _audio_arm_rescrub(self):
        """Adjusting anything WHILE listening restarts playback from the new state, in
        whichever mode is running — a nudge or an edge-resize during Play segment should not
        demand another button press. Debounced so a drag or a held arrow relaunches once."""
        if self._audio_playing:
            if self._audio_cursor_job is not None:
                self.root.after_cancel(self._audio_cursor_job)
                self._audio_cursor_job = None
            if getattr(self, "_audio_rescrub_job", None) is not None:
                self.root.after_cancel(self._audio_rescrub_job)
            self._audio_rescrub_job = self.root.after(180, self._audio_rescrub)

    def _audio_rescrub(self):
        self._audio_rescrub_job = None
        if self._audio_playing and self._audio_play_mode:
            mode = self._audio_play_mode
            self._audio_stop()
            self.audio_toggle_play(mode)

    def _audio_nudge(self, delta):
        self._audio_seek(self.audio_pos + delta)

    _AUDIO_EDGE_PX = 7                        # how close to a span edge counts as grabbing it

    def _audio_edge_at(self, x):
        """"left"/"right" when x is on a span edge, else None. Right wins a tie on a tiny
        span — the more common resize."""
        if not self.audio_src:
            return None
        v0, vd = self._audio_view_bounds()
        x0 = (self.audio_pos - v0) / vd * self.AUDIO_WAVE_W
        x1 = (self.audio_pos + self._audio_span() - v0) / vd * self.AUDIO_WAVE_W
        if abs(x - x1) <= self._AUDIO_EDGE_PX:
            return "right"
        if abs(x - x0) <= self._AUDIO_EDGE_PX:
            return "left"
        return None

    def _nearest_grid_frames(self, len_s, max_len_s):
        """The allowed length closest to len_s that still fits in max_len_s."""
        fits = [f for f in AUDIO_GRID_FRAMES
                if hop_exact_samples(f) / AUDIO_SAMPLE_RATE <= max_len_s + 0.01]
        if not fits:
            fits = [AUDIO_GRID_FRAMES[0]]
        return min(fits, key=lambda f: abs(hop_exact_samples(f) / AUDIO_SAMPLE_RATE - len_s))

    def _audio_press(self, event):
        """Button down, three grips: an EDGE of the span resizes it (snapping to the allowed
        lengths), inside it DRAGS it (no jump — it stays under your hand), outside it places
        it at the click."""
        if not self.audio_src:
            return
        self._audio_dragging = True
        self._audio_edge = self._audio_edge_at(event.x)
        v0, vd = self._audio_view_bounds()
        t = v0 + event.x / self.AUDIO_WAVE_W * vd
        if self._audio_edge == "left":
            self._audio_edge_end = self.audio_pos + self._audio_span()   # the anchor
        elif self._audio_edge is None:
            if self.audio_pos <= t <= self.audio_pos + self._audio_span():
                self._audio_grab = t - self.audio_pos
            else:
                self._audio_grab = 0.0
                self._audio_seek(t)

    def _audio_drag(self, event):
        if not self.audio_src:
            return
        v0, vd = self._audio_view_bounds()
        # The cursor clamped to the canvas, THEN mapped: dragging past the edge parks the
        # segment at the view's boundary instead of extrapolating off into the rest of the
        # file. Panning further is a release away.
        x = max(0, min(event.x, self.AUDIO_WAVE_W))
        t = v0 + x / self.AUDIO_WAVE_W * vd
        edge = getattr(self, "_audio_edge", None)
        if edge == "right":
            frames = self._nearest_grid_frames(t - self.audio_pos,
                                               self.audio_dur - self.audio_pos)
            if frames != self.audio_frames_var.get():
                self.audio_frames_var.set(frames)
                self._audio_refresh()
                self._audio_arm_rescrub()
        elif edge == "left":
            end = self._audio_edge_end
            frames = self._nearest_grid_frames(end - t, end)
            span = hop_exact_samples(frames) / AUDIO_SAMPLE_RATE
            if frames != self.audio_frames_var.get() or abs(end - span - self.audio_pos) > 1e-6:
                self.audio_frames_var.set(frames)
                self._audio_seek(end - span)     # seek arms the rescrub itself
        else:
            self._audio_seek(t - getattr(self, "_audio_grab", 0.0))

    def _audio_release(self, _event):
        self._audio_dragging = False
        self._audio_edge = None
        self._audio_seek(self.audio_pos)      # one follow-check now the drag is over

    def _audio_pan(self, frac):
        """Slide a zoomed view sideways by `frac` of its own width. No-op unzoomed."""
        if self.audio_view is None:
            return
        v0, vd = self.audio_view
        self.audio_view = (max(0.0, min(v0 + frac * vd, self.audio_dur - vd)), vd)
        self._audio_refresh()

    def _audio_xview(self, *args):
        """The nav bar's command: moveto from a drag, scroll from its arrows/trough."""
        if self.audio_view is None or not self.audio_dur:
            return
        v0, vd = self.audio_view
        if args and args[0] == "moveto":
            n0 = float(args[1]) * self.audio_dur
        elif args and args[0] == "scroll":
            n0 = v0 + int(args[1]) * vd * (0.1 if args[2] == "units" else 0.9)
        else:
            return
        self.audio_view = (max(0.0, min(n0, self.audio_dur - vd)), vd)
        self._audio_refresh()

    def _audio_hover(self, event):
        """A resize cursor over the span's edges, so the grips announce themselves."""
        if getattr(self, "_audio_dragging", False):
            return
        want = "sb_h_double_arrow" if self._audio_edge_at(event.x) else "crosshair"
        if self.audio_canvas.cget("cursor") != want:
            self.audio_canvas.configure(cursor=want)

    def _audio_zoom(self, factor, focus_t=None):
        """Zoom the waveform. factor >1 in, <1 out, None = fit all. The anchor — what stays
        put on screen — is `focus_t` when given (the wheel passes the time under the mouse, so
        the sound you are pointing at stays under the pointer), else the playhead (buttons)."""
        if not self.audio_src:
            return
        if factor is None:
            self.audio_view = None
        else:
            v0, vd = self._audio_view_bounds()
            anchor = self.audio_pos if focus_t is None else max(0.0, min(focus_t, self.audio_dur))
            nd = max(0.5, min(self.audio_dur, vd / factor))
            if nd >= self.audio_dur:
                self.audio_view = None
            else:
                # The anchor keeps its screen position: same fraction of the window.
                frac = (anchor - v0) / vd if vd else 0.5
                frac = max(0.0, min(1.0, frac))
                n0 = max(0.0, min(anchor - frac * nd, self.audio_dur - nd))
                self.audio_view = (n0, nd)
        self._audio_refresh()

    def _audio_span(self):
        return hop_exact_samples(self.audio_frames_var.get()) / AUDIO_SAMPLE_RATE

    def _audio_refresh(self):
        """Redraw the waveform (the current VIEW of it), the segment span and the readout."""
        c = self.audio_canvas
        c.delete("all")
        W, H = self.AUDIO_WAVE_W, self.AUDIO_WAVE_H
        mid = H / 2
        v0, vd = self._audio_view_bounds()

        def to_x(t):
            return (t - v0) / vd * W

        if self.audio_src and self.audio_dur:
            span = self._audio_span()
            self.audio_pos = max(0.0, min(self.audio_pos, max(0.0, self.audio_dur - span)))
            x0 = max(0, to_x(self.audio_pos))
            x1 = min(W, to_x(self.audio_pos + span))
            if x1 > 0 and x0 < W:
                c.create_rectangle(x0, 0, x1, H, fill=COLORS["bg_hover"], outline="")
        self._audio_slice_peaks()
        if self.audio_peaks:
            for x, (lo, hi) in enumerate(self.audio_peaks):
                c.create_line(x, mid - hi * (mid - 4), x, mid - lo * (mid - 4),
                              fill=COLORS["accent"])
        else:
            c.create_line(0, mid, W, mid, fill=COLORS["border"])
            if self.audio_src:
                c.create_text(W / 2, mid - 14, text="reading the waveform…",
                              fill=COLORS["text_muted"], font=(FONT_FAMILY, 10))
        if self.audio_src and self.audio_dur:
            span = self._audio_span()
            x0, x1 = to_x(self.audio_pos), to_x(self.audio_pos + span)
            if x1 > 0 and x0 < W:
                c.create_rectangle(max(0, x0), 0, min(W, x1), H,
                                   outline=COLORS["accent"], width=2)
            # The view's own clock, so a zoomed window says where in the file it is.
            c.create_text(4, H - 4, anchor="sw", text=f"{v0:.2f} s",
                          fill=COLORS["text_muted"], font=(FONT_FAMILY, 8))
            c.create_text(W - 4, H - 4, anchor="se", text=f"{v0 + vd:.2f} s",
                          fill=COLORS["text_muted"], font=(FONT_FAMILY, 8))
            zoomed = f"   ·   view {vd:.1f} s of {self.audio_dur:.1f} s" \
                if self.audio_view is not None else ""
            self.audio_pos_label.configure(
                text=f"{self.audio_pos:6.2f} s  →  {self.audio_pos + span:6.2f} s   "
                     f"of {self.audio_dur:.1f} s{zoomed}")
        else:
            self.audio_pos_label.configure(text="")
        # The nav bar mirrors the view: thumb position and size are the window's place in the
        # whole recording. Unzoomed it fills the trough and there is nothing to drag.
        if hasattr(self, "audio_scroll"):
            if self.audio_dur:
                self.audio_scroll.set(v0 / max(self.audio_dur, 1e-9),
                                      (v0 + vd) / max(self.audio_dur, 1e-9))
            else:
                self.audio_scroll.set(0.0, 1.0)
        self._audio_refresh_planned_name()

    # -- audio: playback ----------------------------------------------------------------------------
    def audio_toggle_play(self, mode="segment"):
        """The two play BUTTONS: "file" from the playhead onward (finding the moment),
        "segment" exactly the export span (checking it). While playing they read Pause and DO
        pause — space resumes from there; pressing a play button starts its mode fresh."""
        if self._audio_playing:
            self._audio_pause()
            return
        self._audio_start(mode)

    def _audio_start(self, mode, start=None, rate=1.0, direction=1):
        """Start playback: mode x position x speed x direction — the one engine every play
        control drives. `start` None means the playhead (which for "segment" IS the segment
        start); J/K/L and pause/resume pass explicit positions and rates."""
        if not self.audio_src:
            return
        if self._audio_playing:
            self._audio_stop()
        if start is None:
            start = self.audio_pos
        start = max(0.0, min(float(start), self.audio_dur))
        if mode == "segment":
            seg_end = self.audio_pos + self._audio_span()
            # Resuming inside the span plays the REMAINDER; a stale resume point past the end
            # starts the segment over — the next tap should always produce sound.
            if not (self.audio_pos <= start < seg_end - 0.05):
                start = self.audio_pos
            span = seg_end - start
            self.audio_play_btn.configure(text="⏸  Pause")
        elif direction < 0:
            span = min(start, 600.0)            # backward: what lies behind the point
            self.audio_listen_btn.configure(text="⏸  Pause")
        else:
            # To the end of the recording, capped at ten minutes of extraction — nobody scrubs
            # longer than that in one sitting, and the temp wav stays sane.
            span = min(max(0.1, self.audio_dur - start), 600.0)
            self.audio_listen_btn.configure(text="⏸  Pause")
        if span <= 0.02:
            return
        # Live volume when the backend allows it — the slider then works MID-play. Only when it
        # doesn't (non-Windows external player) is the gain baked into the preview file.
        gain = 1.0 if set_live_volume(self.audio_volume_var.get()) \
            else max(0.0, float(self.audio_volume_var.get())) / 100.0
        self._audio_playing = True
        self._audio_play_mode = mode
        self._audio_play_rate = float(rate)
        self._audio_play_dir = 1 if direction >= 0 else -1
        self._audio_play_from = start
        self._audio_resume = None
        if rate != 1.0 or direction < 0:
            self.audio_status.configure(
                text=f"{'◀' if direction < 0 else '▶'} {rate:g}x", fg=COLORS["text_secondary"])
        threading.Thread(target=self._audio_play_worker,
                         args=(start, span, gain, float(rate), self._audio_play_dir),
                         daemon=True).start()

    @staticmethod
    def _atempo_chain(rate):
        """ffmpeg's atempo tops out at 2.0 per stage, so 3x is a chain. Returns filter parts."""
        parts = []
        while rate > 2.0:
            parts.append("atempo=2.0")
            rate /= 2.0
        if abs(rate - 1.0) > 1e-6:
            parts.append(f"atempo={rate:g}")
        return parts

    def _audio_play_worker(self, start, span, gain, rate=1.0, direction=1):
        err = None
        try:
            wav = os.path.join(tempfile.gettempdir(), "gizmo_voice_preview.wav")
            # Backward (J) extracts what lies BEHIND the point and reverses it; speed is an
            # atempo chain either way. All of it happens in the extraction, because winsound
            # can only ever play a file forward at 1x.
            ss = start - span if direction < 0 else start
            filters = [f"volume={gain:.3f}"]
            if direction < 0:
                filters.append("areverse")
            filters += self._atempo_chain(rate)
            p = _run([self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                      "-ss", f"{ss:.3f}", "-t", f"{span:.3f}", "-i", self.audio_src,
                      "-vn", "-af", ",".join(filters), "-ac", "2", "-ar", "44100",
                      "-c:a", "pcm_s16le", wav])
            if p.returncode != 0 or not os.path.exists(wav):
                raise RuntimeError((p.stderr or b"").decode("utf-8", "replace")[-300:]
                                   or "could not extract the audio")
            _play_wav(wav)
        except Exception as exc:
            err = str(exc)
        self.root.after(0, self._audio_play_done, span / max(rate, 1e-6), err)

    def _audio_play_done(self, span, err):
        if err:
            self._audio_stop()
            messagebox.showerror("Gizmo — cannot play that audio", err)
            return
        # winsound only starts once the extraction lands, so the cursor clock starts HERE, not
        # when the button was pressed — an early t0 would draw the cursor ahead of the sound.
        import time as _time
        self._audio_play_t0 = _time.monotonic()
        self._audio_play_job = self.root.after(int(span * 1000) + 200, self._audio_stop)
        self._audio_cursor_tick()

    def _audio_cursor_tick(self):
        """The moving playback line — wall-clock driven, since winsound reports nothing. Close
        enough for scrubbing by ear, which is what it exists for."""
        if not self._audio_playing:
            return
        cur = self._audio_playback_pos()
        c = self.audio_canvas
        c.delete("playcursor")
        if self.audio_dur:
            v0, vd = self._audio_view_bounds()
            # Listening past the edge of a zoomed view: pan the window along with the sound
            # (either direction), so scrub-by-ear works zoomed in too.
            if self.audio_view is not None and 0 < cur < self.audio_dur:
                if cur > v0 + vd:
                    self.audio_view = (min(cur - vd * 0.1, self.audio_dur - vd), vd)
                    self._audio_refresh()
                elif cur < v0:
                    self.audio_view = (max(0.0, cur - vd * 0.9), vd)
                    self._audio_refresh()
                v0, vd = self._audio_view_bounds()
            x = (cur - v0) / vd * self.AUDIO_WAVE_W
            if 0 <= x <= self.AUDIO_WAVE_W:
                c.create_line(x, 0, x, self.AUDIO_WAVE_H, fill=COLORS["success"], width=2,
                              tags="playcursor")
        self._audio_cursor_job = self.root.after(50, self._audio_cursor_tick)

    def _audio_playback_pos(self):
        """Where the sound IS right now — direction- and speed-aware wall-clock estimate."""
        import time as _time
        elapsed = _time.monotonic() - self._audio_play_t0
        cur = self._audio_play_from + self._audio_play_dir * self._audio_play_rate * elapsed
        return max(0.0, min(cur, self.audio_dur))

    def _audio_stop(self):
        for attr in ("_audio_play_job", "_audio_cursor_job"):
            job = getattr(self, attr, None)
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except Exception:
                    pass
                setattr(self, attr, None)
        _stop_wav()
        self._audio_playing = False
        self._audio_play_mode = None
        self._audio_play_rate = 1.0
        self._audio_play_dir = 1
        try:
            if self.audio_play_btn.winfo_exists():
                self.audio_play_btn.configure(text="▶  Play segment")
                self.audio_listen_btn.configure(text="▶  Play from here")
                self.audio_canvas.delete("playcursor")
        except (AttributeError, tk.TclError):
            pass                              # closing before the tab was built

    # -- audio: pause / resume and the J-K-L shuttle -----------------------------------------------
    def _audio_pause(self):
        """Freeze where the sound is; space (or K) again resumes from exactly there."""
        if not self._audio_playing:
            return
        mode, cur = self._audio_play_mode, self._audio_playback_pos()
        self._audio_stop()
        self._audio_resume = (mode, cur)
        self.audio_status.configure(text=f"paused at {cur:.2f} s — space resumes",
                                    fg=COLORS["text_secondary"])

    def _audio_space(self):
        """Space: play / pause / resume — the one key that always does the obvious thing.
        Idle it plays the segment from its start; playing it pauses; paused it resumes."""
        if self._audio_playing:
            self._audio_pause()
        elif self._audio_resume is not None:
            mode, cur = self._audio_resume
            self._audio_start(mode, start=cur)
        else:
            self._audio_start("segment")

    _SHUTTLE_RATES = (1.0, 1.5, 2.0, 3.0)

    def _audio_shuttle(self, direction):
        """The editing-suite J/L shuttle: tap to play that way, tap again to go faster
        (1 -> 1.5 -> 2 -> 3x), from wherever the sound currently is."""
        if self._audio_playing and self._audio_play_dir == direction \
                and self._audio_play_mode == "file":
            rates = self._SHUTTLE_RATES
            i = min(len(rates) - 1,
                    next((n for n, r in enumerate(rates)
                          if r >= self._audio_play_rate - 1e-6), 0) + 1)
            self._audio_start("file", start=self._audio_playback_pos(),
                              rate=rates[i], direction=direction)
            return
        start = (self._audio_playback_pos() if self._audio_playing
                 else (self._audio_resume[1] if self._audio_resume else self.audio_pos))
        self._audio_start("file", start=start, rate=1.0, direction=direction)

    # -- audio: whisper -----------------------------------------------------------------------------
    def audio_transcribe(self):
        """Whisper the marked segment and append `saying "…"` to the caption.

        transformers is already in Fizgig's venv (it runs the EN→ZH translator the same way), so
        no new dependency — just a one-time ~150 MB model download on first use."""
        if self._whisper_busy or not self.audio_src:
            return
        self._whisper_busy = True
        self.audio_whisper_btn.configure(state=tk.DISABLED, text="⏳ Transcribing…")
        self._whisper_dots = 0
        self._whisper_progress_tick()
        lang = self.audio_lang_var.get()
        threading.Thread(target=self._whisper_worker,
                         args=(self.audio_pos, self._audio_span(),
                               None if lang == "Auto detect" else lang.lower()),
                         daemon=True).start()

    def _whisper_progress_tick(self):
        """A visibly alive status line while Whisper works — the first run downloads ~150 MB
        and a silent, disabled button for that long reads as a hang."""
        if not self._whisper_busy:
            return
        self._whisper_dots = (self._whisper_dots + 1) % 4
        base = ("listening — first use downloads ~150 MB, please wait"
                if not hasattr(self, "_whisper_pipe") else "listening")
        self.audio_status.configure(text=base + "." * self._whisper_dots,
                                    fg=COLORS["text_secondary"])
        self.root.after(400, self._whisper_progress_tick)

    @staticmethod
    def _whisper_degenerate(text, span):
        """Whisper's two classic failures, caught by arithmetic rather than trust.

        A wrong language guess feeds its repetition loop, and the result is paragraphs of the
        same phrase — from 4.5 seconds of audio, which cannot physically hold more than ~30
        characters a second of speech. A loop also has almost no unique words. Either signal
        means the transcript is hallucinated, not heard."""
        words = text.split()
        if len(text) > max(80, span * 30):
            return True
        if len(words) >= 10 and len(set(w.lower() for w in words)) / len(words) < 0.4:
            return True
        return False

    def _whisper_worker(self, start, span, lang=None):
        """lang None = auto-detect, with an English-forced retry if the output degenerates
        (a wrong guess is the usual cause of the loop). A chosen language skips the guess
        entirely — and the retry, since there is no better guess left to make."""
        text, err = None, None
        try:
            wav = os.path.join(tempfile.gettempdir(), "gizmo_whisper.wav")
            p = _run([self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                      "-ss", f"{start:.3f}", "-t", f"{span:.3f}", "-i", self.audio_src,
                      "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav])
            if p.returncode != 0:
                raise RuntimeError("could not extract the segment")
            from transformers import pipeline
            if not hasattr(self, "_whisper_pipe"):
                self._whisper_pipe = pipeline("automatic-speech-recognition",
                                              model="openai/whisper-base", device=-1)

            # Raw samples, not the file path: handed a path, transformers shells out to its
            # own ffmpeg to decode it — without the no-window flag, so a black console
            # flashes over the app on every transcription. We just extracted this wav with
            # our flashless ffmpeg; reading it back is a wave-module one-liner.
            import wave as _wave

            import numpy as _np
            with _wave.open(wav) as _w:
                _raw = _np.frombuffer(_w.readframes(_w.getnframes()),
                                      dtype=_np.int16).astype(_np.float32) / 32768.0
            _sample = {"raw": _raw, "sampling_rate": 16000}

            def hear(language):
                kw = ({"generate_kwargs": {"language": language, "task": "transcribe"}}
                      if language else {})
                return (self._whisper_pipe(dict(_sample), **kw).get("text") or "").strip()

            text = hear(lang)
            if lang is None and self._whisper_degenerate(text, span):
                text = hear("english")
            if self._whisper_degenerate(text, span):
                text = None
                err = ("Whisper looped on this segment — it hallucinated repeating text "
                       "instead of hearing it, which it does now and then on short or "
                       "ambiguous audio.\n\nNudge the start a little (even 100 ms changes the "
                       "outcome) and try again, pick the language from the dropdown, or just "
                       "type the words.")
        except Exception as exc:
            err = (f"Whisper could not run: {exc}\n\nIt needs one ~150 MB download the first "
                   f"time (internet required once). The caption still works without it — "
                   f"describe the voice and skip the transcript.")
        self.root.after(0, self._whisper_done, text, err)

    def _whisper_done(self, text, err):
        self._whisper_busy = False
        self.audio_whisper_btn.configure(state=tk.NORMAL, text="🎤 Transcribe")
        if err:
            self.audio_status.configure(text="transcription failed", fg=COLORS["error"])
            messagebox.showerror("Gizmo — Whisper", err)
            return
        self.audio_status.configure(text="", fg=COLORS["text_secondary"])
        if not text:
            self.audio_status.configure(text="Whisper heard no words in that segment",
                                        fg=COLORS["text_secondary"])
            return
        # The box is CLEARED first, always — the transcript describes this segment and this
        # segment only, and anything left over from the previous one (its description, its
        # words) would silently ride along into the caption. Type the voice description in
        # front of the transcript after it lands.
        self.audio_caption.delete("1.0", tk.END)
        self.audio_caption.insert("1.0", f'saying "{text}"')

    # -- audio: queue + export ----------------------------------------------------------------------
    def _audio_caption_text(self):
        cap = " ".join(self.audio_caption.get("1.0", tk.END).split())
        trig = self.audio_trigger_var.get().strip().rstrip(",")
        if trig and cap:
            return f"{trig}, {cap}"
        return trig or cap

    def _audio_claimed(self):
        return {os.path.basename(j["dst"]) for j in self.audio_queue}

    def _audio_refresh_planned_name(self):
        pass                                    # reserved: the queue card shows names on Add

    def audio_add_to_queue(self):
        if not self.audio_src:
            return
        out_dir = self.audio_out_var.get().strip()
        if not out_dir:
            messagebox.showwarning("Gizmo", "Pick a save folder first.")
            return
        if not self._audio_caption_text():
            # A voice item with no caption is dropped by Fizgig's caption filter — refusing at
            # the mark is kinder than a warning at training time.
            messagebox.showwarning("Gizmo — caption first",
                                   "Write the caption (or at least a trigger word) before "
                                   "marking — every exported segment needs its .txt to train.")
            return
        frames = self.audio_frames_var.get()
        span = self._audio_span()
        if self.audio_pos + span > self.audio_dur + 0.01:
            messagebox.showwarning("Gizmo", "That segment runs past the end of the recording.")
            return
        dst = voice_output_name(self.audio_src, out_dir, self._audio_claimed())
        self.audio_queue.append({
            "src": self.audio_src, "dst": dst, "start": self.audio_pos, "frames": frames,
            "caption": self._audio_caption_text(), "done": False,
        })
        self._audio_refresh_queue_box()
        self.audio_status.configure(text=f"marked {os.path.basename(dst)}",
                                    fg=COLORS["text_secondary"])

    def _audio_refresh_queue_box(self):
        self.audio_queue_box.delete(0, tk.END)
        for j in self.audio_queue:
            secs = hop_exact_samples(j["frames"]) / AUDIO_SAMPLE_RATE
            mark = "✓" if j["done"] else "·"
            cap = j["caption"][:46] + ("…" if len(j["caption"]) > 46 else "")
            self.audio_queue_box.insert(
                tk.END, f'{mark} {os.path.basename(j["dst"]):<22} {j["start"]:7.2f}s '
                        f'{secs:4.1f}s  {cap}')

    def _audio_recall(self):
        sel = self.audio_queue_box.curselection()
        if not sel:
            return
        j = self.audio_queue[sel[0]]
        if j["done"]:
            return
        self.audio_frames_var.set(j["frames"])
        self.audio_caption.delete("1.0", tk.END)
        cap = j["caption"]
        trig = self.audio_trigger_var.get().strip().rstrip(",")
        if trig and cap.startswith(trig + ", "):
            cap = cap[len(trig) + 2:]
        self.audio_caption.insert("1.0", cap)
        self._audio_seek(j["start"])

    def _audio_remove_selected(self):
        sel = self.audio_queue_box.curselection()
        if sel:
            del self.audio_queue[sel[0]]
            self._audio_refresh_queue_box()

    def _audio_clear_queue(self):
        self.audio_queue = [j for j in self.audio_queue if j["done"]]
        self._audio_refresh_queue_box()

    def audio_export_queue(self):
        jobs = [j for j in self.audio_queue if not j["done"]]
        if not jobs:
            self.audio_status.configure(text="nothing waiting", fg=COLORS["text_secondary"])
            return
        self.audio_export_btn.configure(state=tk.DISABLED)
        threading.Thread(target=self._audio_export_worker, args=(jobs,), daemon=True).start()

    def _audio_export_worker(self, jobs):
        wrote, failed = 0, []
        for j in jobs:
            try:
                os.makedirs(os.path.dirname(j["dst"]), exist_ok=True)
                n = hop_exact_samples(j["frames"])
                # aresample first, then trim BY SAMPLE COUNT at the target rate — the exact
                # length Fizgig's cache validation demands, no tolerance spent.
                p = _run([self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                          "-ss", f'{j["start"]:.3f}', "-i", j["src"], "-vn",
                          "-af", f"aresample={AUDIO_SAMPLE_RATE},atrim=end_sample={n},"
                                 f"apad=whole_len={n}",
                          "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_SAMPLE_RATE),
                          "-c:a", "pcm_s16le", j["dst"]])
                if p.returncode != 0 or not os.path.exists(j["dst"]):
                    raise RuntimeError((p.stderr or b"").decode("utf-8", "replace")[-200:])
                with open(os.path.splitext(j["dst"])[0] + ".txt", "w", encoding="utf-8") as f:
                    f.write(j["caption"])
                j["done"] = True
                wrote += 1
            except Exception as exc:
                failed.append(f'{os.path.basename(j["dst"])}: {exc}')
            self.root.after(0, self._audio_refresh_queue_box)
        def done():
            self.audio_export_btn.configure(state=tk.NORMAL)
            if failed:
                self.audio_status.configure(text=f"wrote {wrote}, {len(failed)} failed",
                                            fg=COLORS["error"])
                messagebox.showerror("Gizmo — some segments failed", "\n".join(failed[:6]))
            else:
                self.audio_status.configure(
                    text=f"wrote {wrote} segment(s) + captions", fg=COLORS["success"])
        self.root.after(0, done)

    def _audio_pick_output(self):
        d = filedialog.askdirectory(title="Where should the voice segments go?",
                                    initialdir=self.audio_out_var.get() or os.path.expanduser("~"))
        if d:
            self.audio_out_var.set(d)

    def _audio_open_folder(self):
        d = self.audio_out_var.get().strip()
        if not d or not os.path.isdir(d):
            messagebox.showinfo("Gizmo", "Nothing there yet — export a segment first.")
            return
        try:
            if os.name == "nt":
                os.startfile(d)                                     # noqa: S606
            else:
                subprocess.Popen(["xdg-open", d], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except Exception as exc:
            messagebox.showerror("Gizmo", f"Cannot open {d}\n\n{exc}")

    def _audio_set_enabled(self, on):
        state = tk.NORMAL if on else tk.DISABLED
        for w in (self.audio_listen_btn, self.audio_play_btn, self.audio_add_btn,
                  self.audio_export_btn, self.audio_whisper_btn):
            w.configure(state=state)

    # -- cost -------------------------------------------------------------------------------------
    def _refresh_length_note(self):
        """What this length can be trained at, per card. The number people actually need is not
        the token count — it is whether their own card can run it."""
        frames = self._frames()
        by_card = CLIP_VRAM.get(frames, {})
        parts = []
        for gb in (16, 24, 32):
            mp = by_card.get(gb)
            parts.append(f"{gb} GB: {'—' if mp is None else f'up to {mp:g} MP'}")
        note = "Trainable at  " + "   ·   ".join(parts)
        if by_card.get(16) is None:
            note += "   (a 16 GB card cannot place a clip this long at all)"
        self.len_note.configure(
            text=note + "\nEstimated from Fizgig's own swap planner and not yet confirmed by a "
                        "real run at these lengths — where it is wrong it will be optimistic.",
            fg=COLORS["warning"] if by_card.get(24) is None else COLORS["text_explain"])

    def _refresh_cost(self):
        self._refresh_length_note()
        frames = self._frames()
        w, h = self._size()
        if not w:
            self.size_label.configure(text="—")
            self.cost_label.configure(text="")
            return
        self.size_label.configure(
            text=f"{w} x {h}   ({w * h / 1e6:.2f} MP)"
                 + ("   — the crop, at its own resolution" if self.crop
                    else "   — the whole frame, at its own resolution"))
        # Costed at the TRAINING resolution, not at the size on disk. The stored size is native
        # and training rescales it, so counting tokens at 1888x1056 would report a number no run
        # ever pays — the useful figure is what a step costs at the megapixels you train at.
        lat = LATENT_FRAMES[frames]
        tok = int(0.25e6 / (SIZE_STEP * SIZE_STEP)) * lat
        self.cost_label.configure(
            text=f"{frames} frames is {lat} latent frames — about {tok:,} tokens a step at "
                 f"0.25 MP training, {lat}x a still, and four times that at 1 MP. Attention "
                 f"scales with the square of it, which is what the line above is really saying.",
            fg=COLORS["warning"] if lat > 12 else COLORS["text_explain"])
        if self.src:
            self._update_pos_labels()

    # -- saving -----------------------------------------------------------------------------------
    def pick_output(self):
        d = filedialog.askdirectory(title="Where should the clips go?",
                                    initialdir=self.out_var.get() or os.path.expanduser("~"))
        if d:
            self.out_var.set(d)

    def open_output_folder(self):
        d = self.out_var.get().strip()
        if not d or not os.path.isdir(d):
            messagebox.showinfo("Gizmo", "Nothing there yet — export a clip first.")
            return
        try:
            if os.name == "nt":
                os.startfile(d)                                     # noqa: S606
            else:
                subprocess.Popen(["xdg-open", d], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except Exception as exc:
            messagebox.showerror("Gizmo", f"Cannot open {d}\n\n{exc}")

    # -- the queue ---------------------------------------------------------------------------------
    def _claimed_names(self):
        """Names already spoken for by the queue. output_name only checks the disk, so without
        this two clips added before either is exported would both be handed _01."""
        return {os.path.basename(j["dst"]) for j in self.queue}

    def _planned_job(self):
        """The job Add would create right now, or None if there isn't one."""
        out_dir = self.out_var.get().strip()
        if not self.src or not out_dir:
            return None
        w, h = self._size()
        if not w:
            return None
        muted = bool(self.mute_var.get())
        return {
            "src": self.src, "dst": output_name(self.src, out_dir, muted, self._claimed_names()),
            "start_frame": self.frame_pos, "start": self.frame_pos / self.info["fps"],
            "frames": self._frames(), "w": w, "h": h,
            "keep_every": self._keep_every(), "muted": muted,
            "crop": self.crop, "sar": self.info["sar"],
            # A muted clip still carries its audio: the _mute suffix is the instruction, and
            # keeping the track means the decision is reversible by rename, not by re-export.
            "with_audio": bool(self.info["has_audio"]),
            "motion_index": self.motion_box.current(),
            "done": False, "error": None,
        }

    def _refresh_planned_name(self, *_a):
        job = self._planned_job()
        self.name_label.configure(
            text=(f"next:  {os.path.basename(job['dst'])}" if job else ""))

    def add_to_queue(self):
        if not self.src:
            return
        out_dir = self.out_var.get().strip()
        if not out_dir:
            messagebox.showwarning("Gizmo", "Choose a folder to save the clips into first.")
            return
        job = self._planned_job()
        if job is None:
            return
        # Refused at ADD time rather than at export, so a queue of thirty does not stop halfway
        # for something that was knowable the moment it was marked.
        duration = self.info["duration"] or 0
        k = job["keep_every"]
        span = (job["frames"] * k / self.info["fps"]) if k else job["frames"] / FPS
        if duration and job["start"] + span > duration + 1e-3:
            messagebox.showwarning(
                "Gizmo — that clip runs off the end",
                f"The source ends at {duration:.2f} s, and this clip would need up to "
                f"{job['start'] + span:.2f} s.\n\nMove the playhead earlier, or choose a shorter "
                f"length.")
            return
        self.queue.append(job)
        self._refresh_queue_box()
        self._refresh_planned_name()
        self.status.configure(text=f"Queued {os.path.basename(job['dst'])}",
                              fg=COLORS["text_secondary"])

    def _queue_row(self, i, job):
        mark = "✗" if job["error"] else ("✓" if job["done"] else "·")
        sound = "muted" if job["muted"] else ("sound" if job["with_audio"] else "silent")
        slow = f" {self.info['fps'] / job['keep_every'] / FPS:.2f}x slow" \
            if job["keep_every"] and self.info else ""
        crop = "  cropped" if job.get("crop") else ""
        return (f"{mark} {i + 1:2d}  {os.path.basename(job['dst']):<28} "
                f"{job['frames']:>2}f  {job['w']}x{job['h']}  @{job['start']:6.2f}s  "
                f"{sound}{slow}{crop}")

    def _refresh_queue_box(self):
        self.queue_box.delete(0, tk.END)
        for i, job in enumerate(self.queue):
            self.queue_box.insert(tk.END, self._queue_row(i, job))
            if job["error"]:
                self.queue_box.itemconfigure(i, foreground=COLORS["error"])
            elif job["done"]:
                self.queue_box.itemconfigure(i, foreground=COLORS["success"])
        pending = sum(1 for j in self.queue if not j["done"])
        self.export_btn.configure(
            text=f"Export queue ({pending})" if pending else "Export queue")
        self.queue_box.see(tk.END)

    def recall_selected(self):
        """Put the playhead and settings back where they were when a row was added — the undo
        for 'I queued that one two frames late'."""
        sel = self.queue_box.curselection()
        if not sel or self._busy:
            return
        job = self.queue[sel[0]]
        if job["done"]:
            return
        if job["src"] != self.src:
            messagebox.showinfo("Gizmo", "That clip came from a different source video.")
            return
        self.len_box.current(GRID_FRAMES.index(job["frames"]))
        # Crop first: it rebuilds the size menu, so restoring the size before it would be undone.
        self.crop = job.get("crop")
        self.crop_var.set(bool(self.crop))
        self._describe_crop()
        if 0 <= job["motion_index"] < len(self.motion_box["values"]):
            self.motion_box.current(job["motion_index"])
        self.mute_var.set(job["muted"])
        self.scale.set(job["start_frame"])
        self._refresh_cost()
        self._refresh_planned_name()

    def remove_selected(self):
        sel = self.queue_box.curselection()
        if not sel or self._busy:
            return
        if self.queue[sel[0]]["done"]:
            messagebox.showinfo("Gizmo", "That one is already written. Delete the file itself if "
                                         "you don't want it.")
            return
        del self.queue[sel[0]]
        self._refresh_queue_box()
        self._refresh_planned_name()

    def clear_queue(self):
        if self._busy:
            return
        self.queue = [j for j in self.queue if j["done"]]
        self._refresh_queue_box()
        self._refresh_planned_name()

    # -- exporting ---------------------------------------------------------------------------------
    def export_queue(self):
        if self._busy:
            return
        pending = [j for j in self.queue if not j["done"]]
        if not pending:
            messagebox.showinfo("Gizmo", "Nothing queued. Mark a clip and press Add to queue.")
            return
        out_dir = self.out_var.get().strip()
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Gizmo", f"Cannot create {out_dir}\n\n{exc}")
            return

        self._busy = True
        self.export_btn.configure(state=tk.DISABLED)
        self.add_btn.configure(state=tk.DISABLED)
        threading.Thread(target=self._export_worker, args=(pending,), daemon=True).start()

    def _export_worker(self, jobs):
        for n, job in enumerate(jobs, 1):
            self.root.after(0, lambda n=n, job=job: self.status.configure(
                text=f"Exporting {n} of {len(jobs)} — {os.path.basename(job['dst'])}",
                fg=COLORS["text_secondary"]))
            job["error"] = self._export_one(job)
            job["done"] = job["error"] is None
            self.root.after(0, self._refresh_queue_box)
        self.root.after(0, self._export_done, jobs)

    def _export_one(self, job):
        """Write one clip. Returns an error string, or None on success."""
        try:
            cmd = build_export_command(self.ffmpeg, job["src"], job["dst"], job["start"],
                                       job["frames"], job["w"], job["h"], job["keep_every"],
                                       job["with_audio"], job.get("crop"), job.get("sar", 1.0))
            p = _run(cmd, text=True)
            if p.returncode != 0 or not os.path.exists(job["dst"]):
                return (p.stderr or "").strip()[-400:] or "ffmpeg failed"
            # Verified, not assumed: frame count is the one thing Fizgig refuses a clip over, and
            # a source that ends mid-clip silently yields a short one.
            got = count_frames(self.ffmpeg, job["dst"])
            if got is not None and got != job["frames"]:
                os.remove(job["dst"])
                return (f"only {got} frames, not {job['frames']} — the source runs out before "
                        f"the clip does")
            return None
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def _export_done(self, jobs):
        self._busy = False
        self.export_btn.configure(state=tk.NORMAL)
        self.add_btn.configure(state=tk.NORMAL)
        failed = [j for j in jobs if j["error"]]
        ok = len(jobs) - len(failed)
        if failed:
            self.status.configure(text=f"{ok} written, {len(failed)} failed",
                                  fg=COLORS["warning"])
            detail = "\n\n".join(f"{os.path.basename(j['dst'])}: {j['error']}" for j in failed[:6])
            messagebox.showwarning(
                "Gizmo — some clips did not export",
                f"{ok} written, {len(failed)} failed. The failures are marked ✗ in the queue and "
                f"can be adjusted and exported again.\n\n{detail}")
        else:
            self.status.configure(text=f"{ok} clip{'' if ok == 1 else 's'} written",
                                  fg=COLORS["success"])
        self._refresh_planned_name()


def main():
    root = tk.Tk()
    app = Gizmo(root)
    # A path on the command line, which is what Windows hands over when a video is dropped onto
    # the launcher or opened with Gizmo from Explorer.
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        root.after(50, lambda: app.open_path(sys.argv[1]))
    root.mainloop()


if __name__ == "__main__":
    main()
