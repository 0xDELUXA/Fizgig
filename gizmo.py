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
GRID_FRAMES = (5, 22, 39)            # 17n+5 for n = 0..2
LATENT_FRAMES = {5: 2, 22: 7, 39: 12}  # 5n+2 — what each length costs in the DiT
SIZE_STEP = 32
AUDIO_SAMPLE_RATE = 32000
AUDIO_CHANNELS = 2
MUTE_SUFFIX = "_mute"

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
            "has_audio": False, "sample_rate": None, "channels": None, "vcodec": None}

    m = re.search(r"Stream #\d+:\d+.*?: Video: (\w+).*?, (\d{2,5})x(\d{2,5})", out, re.S)
    if m:
        info["vcodec"] = m.group(1)
        info["width"], info["height"] = int(m.group(2)), int(m.group(3))
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
    return info


def grab_frame(ffmpeg, path, seconds, box=(PREVIEW_W, PREVIEW_H)):
    """One frame at `seconds`, scaled to fit `box`, as a PIL image.

    -ss BEFORE -i is the fast form, and has been frame-accurate since ffmpeg 2.1 (it seeks to the
    preceding keyframe and decodes forward). Scrubbing a long source with output-seek instead
    would decode everything up to the mark and take seconds per frame.
    """
    bw, bh = box
    vf = f"scale={bw}:{bh}:force_original_aspect_ratio=decrease"
    p = _run([ffmpeg, "-hide_banner", "-loglevel", "error",
              "-ss", f"{max(0.0, seconds):.3f}", "-i", path,
              "-frames:v", "1", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    if not p.stdout:
        return None
    # The scaler preserves aspect, so the real size has to be recovered from the byte count.
    n = len(p.stdout) // 3
    for w in range(bw, 0, -1):
        if n % w == 0 and n // w <= bh:
            return Image.frombytes("RGB", (w, n // w), p.stdout)
    return None


# --- writing a clip ----------------------------------------------------------------------------

def snap(value):
    """Down to a multiple of 32, never to zero."""
    return max(SIZE_STEP, int(value) // SIZE_STEP * SIZE_STEP)


def target_size(src_w, src_h, megapixels):
    """Source aspect ratio at roughly `megapixels`, both sides snapped to /32.

    Never larger than the source. Upscaling here would invent detail and then charge tokens for
    it — a 640x480 clip blown up to 864x640 costs 1.8x the attention to train on exactly the same
    information. So a small source simply comes out at its own size.
    """
    scale = min(1.0, (megapixels * 1_000_000 / (src_w * src_h)) ** 0.5)
    return snap(src_w * scale), snap(src_h * scale)


def tokens_for(width, height, frames):
    """What this clip costs the DiT: 32x32 pixel patches, times latent frames. Attention is
    quadratic in this number, which is the whole reason clips stop at 39 frames."""
    return (width // SIZE_STEP) * (height // SIZE_STEP) * LATENT_FRAMES[frames]


def build_export_command(ffmpeg, src, dst, start_s, frames, width, height,
                         keep_every=None, with_audio=True):
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
    vf += f",scale={width}:{height}:flags=lanczos,setsar=1"

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
        root.title("Gizmo — video clip prep for Fizgig")
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
        self._playing = False
        self._play_job = None

        self._style()
        self._build()
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
                    arrowcolor=COLORS["text_secondary"], bordercolor=COLORS["border"])
        s.configure("G.Horizontal.TScale", background=COLORS["bg_surface"],
                    troughcolor=COLORS["bg_deep"])

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
        bg = {"normal": COLORS["bg_hover"], "accent": COLORS["accent"]}[kind]
        fg = COLORS["text_primary"]
        b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                      activebackground=COLORS["accent_hover"], activeforeground=fg,
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
        tk.Label(head, text="Cut training clips Fizgig will accept — 24 fps, on the frame grid, "
                            "sized and with the sound sorted out.",
                 font=(FONT_FAMILY, 11), bg=COLORS["bg_deep"],
                 fg=COLORS["text_secondary"]).pack(anchor="w")

        # --- source ---------------------------------------------------------------------------
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
        self.src_info = tk.Label(c, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                 bg=COLORS["bg_surface"], fg=COLORS["text_explain"])
        self.src_info.pack(anchor="w", pady=(8, 0))
        ToolTip(self.src_info,
                "What Gizmo read from the file. None of it has to be right — the whole point is "
                "that Gizmo converts it. It is here so you can see what you are starting from.")

        # --- scrub ----------------------------------------------------------------------------
        c = self._card(body, "2. Find the moment",
                       "The playhead is where the clip STARTS. Drag it, step with the arrow keys "
                       "(Shift for ten at a time, Home for the beginning), or type an exact frame "
                       "below.")
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
        ToolTip(self.canvas, "The first frame of the clip you are about to save.")

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

        # --- settings -------------------------------------------------------------------------
        c = self._card(body, "3. Clip settings")
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
                   "Not a free choice: H3 encodes video in groups, so only 5, 22 and 39 frames "
                   "exist. 22 is the useful one — 5 is barely movement, and 39 costs nearly "
                   "twice what 22 does.")
        label(0, "Length:", LEN_TIP)
        self.len_var = tk.StringVar()
        self.len_box = ttk.Combobox(grid, textvariable=self.len_var, state="readonly",
                                    style="G.TCombobox", width=46,
                                    values=[self._length_label(f) for f in GRID_FRAMES])
        self.len_box.current(1)
        self.len_box.grid(row=0, column=1, sticky="w", pady=5)
        self.len_box.bind("<<ComboboxSelected>>", lambda _e: self._on_length_changed())
        ToolTip(self.len_box, LEN_TIP)

        SIZE_TIP = ("How big the saved clip is.\n\n"
                    "Both sides land on a multiple of 32, which is what H3 needs, and the aspect "
                    "ratio is kept. Gizmo never scales UP — a small source stays its own size "
                    "rather than being blown up into detail that was never there.")
        label(1, "Size:", SIZE_TIP)
        self.size_var = tk.StringVar()
        self.size_box = ttk.Combobox(grid, textvariable=self.size_var, state="readonly",
                                     style="G.TCombobox", width=46, values=[])
        self.size_box.grid(row=1, column=1, sticky="w", pady=5)
        self.size_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh_cost())
        ToolTip(self.size_box, SIZE_TIP)

        MOTION_TIP = ("Real time is what you want almost always: whatever the source runs at, one "
                      "real second stays one second.\n\n"
                      "The slow-motion options keep MORE of the original frames instead of "
                      "resampling, so the action plays back slower than life. Only worth it when "
                      "the slow-motion look is the thing you are training.")
        label(2, "Motion:", MOTION_TIP)
        self.motion_var = tk.StringVar()
        self.motion_box = ttk.Combobox(grid, textvariable=self.motion_var, state="readonly",
                                       style="G.TCombobox", width=46, values=[])
        self.motion_box.grid(row=2, column=1, sticky="w", pady=5)
        self.motion_box.bind("<<ComboboxSelected>>", lambda _e: self._on_motion_changed())
        ToolTip(self.motion_box, MOTION_TIP)

        SOUND_TIP = ("H3 generates sound as well as video, so a clip's audio can be training data "
                     "too.\n\n"
                     "Mute the ones where it should not be — a cough, the wrong speaker, music "
                     "over the top. A muted clip trains its video exactly the same; only the "
                     "sound is ignored.")
        label(3, "Sound:", SOUND_TIP)
        snd = tk.Frame(grid, bg=COLORS["bg_surface"])
        snd.grid(row=3, column=1, sticky="w", pady=5)
        # Sticky between saves on purpose: one source video routinely has sections worth training
        # on and sections with a cough, the wrong speaker, or music over the top, so this gets
        # toggled far more often than it gets reset.
        self.mute_var = tk.BooleanVar(value=False)
        self._sound_radios = []
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
            self._sound_radios.append(rb)

        # Listen before deciding. Muting is a judgement about what the clip actually sounds like,
        # and there is no way to make that judgement from a picture of it.
        label(4, "Listen:", "Play the marked section so you can hear what you would be training "
                            "on before you choose.")
        listen = tk.Frame(grid, bg=COLORS["bg_surface"])
        listen.grid(row=4, column=1, sticky="w", pady=5)
        self.play_btn = self._button(
            listen, "▶  Play sound", self.toggle_play, pad=10,
            tip="Play the audio under the clip you have marked — exactly the section that would "
                "be saved.\n\nTakes a moment to extract the first time you press it.")
        self.play_btn.pack(side=tk.LEFT)
        tk.Label(listen, text="Volume", font=(FONT_FAMILY, 9), bg=COLORS["bg_surface"],
                 fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=(14, 6))
        self.volume_var = tk.DoubleVar(value=70)
        self.volume_scale = ttk.Scale(listen, from_=0, to=100, orient=tk.HORIZONTAL, length=150,
                                      variable=self.volume_var, style="G.Horizontal.TScale")
        self.volume_scale.pack(side=tk.LEFT)
        ToolTip(self.volume_scale,
                "Listening volume only. The saved clip's audio is never touched — a quiet source "
                "stays quiet, and H3 sees exactly what your camera recorded.")

        self.sound_note = tk.Label(grid, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                   wraplength=820, bg=COLORS["bg_surface"],
                                   fg=COLORS["text_explain"])
        self.sound_note.grid(row=5, column=1, sticky="w")

        OUT_TIP = ("Where the clips land.\n\n"
                   "Point it at your Fizgig training folder and the clips are ready to caption "
                   "the moment you are done here. Defaults to a fizgig_clips folder beside the "
                   "source video.")
        label(6, "Save to:", OUT_TIP)
        outrow = tk.Frame(grid, bg=COLORS["bg_surface"])
        outrow.grid(row=6, column=1, sticky="ew", pady=5)
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

        # --- queue ----------------------------------------------------------------------------
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

        self._bind_keys()
        self._set_enabled(False)
        self._refresh_cost()
        self._refresh_queue_box()

    # -- state ----------------------------------------------------------------------------------
    def _length_label(self, frames):
        return f"{frames} frames — {frames / FPS:.2f} s"

    def _frames(self):
        return GRID_FRAMES[self.len_box.current()] if self.len_box.current() >= 0 else 22

    def _size(self):
        """The chosen output size, as (w, h)."""
        m = re.search(r"(\d+)\s*x\s*(\d+)", self.size_var.get())
        return (int(m.group(1)), int(m.group(2))) if m else (None, None)

    def _keep_every(self):
        """None for real time, else keep every k-th source frame. Read from a list held beside
        the dropdown rather than parsed back out of its label — the label is prose and 'keep
        every frame' has no number in it to parse."""
        i = self.motion_box.current()
        return self._motion_keep[i] if 0 <= i < len(self._motion_keep) else None

    def _set_enabled(self, on):
        state = tk.NORMAL if on else tk.DISABLED
        for w in (self.add_btn, self.export_btn, self.scale, self.open_btn, self.pos_entry,
                  *self._sound_radios):
            w.configure(state=state)
        for box in (self.len_box, self.size_box, self.motion_box):
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
        for seq, fn in (("<Left>", lambda: self.step(-1)),
                        ("<Right>", lambda: self.step(1)),
                        ("<Shift-Left>", lambda: self.step(-10)),
                        ("<Shift-Right>", lambda: self.step(10)),
                        ("<Home>", self.go_to_start),
                        ("<Control-s>", self.add_to_queue)):
            self.root.bind(seq, lambda _e, f=fn: (f(), "break")[1])

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

    def load_video(self, path):
        """Everything opening a video does, with no dialog in the way — so it can be tested."""
        info = probe_source(self.ffmpeg, path)
        self._stop_audio()
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
        facts = (f"{info['width']}x{info['height']}  ·  {info['fps']:g} fps  ·  "
                 f"{info['duration'] or 0:.1f} s  ·  {audio}")
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
        self._fill_sizes()
        self._fill_motion()
        self._fill_sound_note()
        self._set_enabled(True)
        self.show_frame()
        self._refresh_cost()
        self._refresh_planned_name()

    def _fill_sizes(self):
        w, h = self.info["width"], self.info["height"]
        # De-duplicated because target_size never upscales: for a 640x480 source all three
        # presets land on the same dimensions, and offering "small / medium / large" that are
        # secretly identical is worse than offering one.
        seen, opts = {}, []
        for mp, name in ((0.26, "small"), (0.59, "medium"), (1.05, "large")):
            tw, th = target_size(w, h, mp)
            if (tw, th) in seen:
                continue
            seen[(tw, th)] = True
            note = "source size" if (tw, th) == (snap(w), snap(h)) else name
            opts.append(f"{tw} x {th}  ({note}, {tw * th / 1e6:.2f} MP)")
        self.size_box.configure(values=opts)
        # Medium by default where there is a choice: 768-ish is where H3 stills already train
        # well, and a clip's token cost climbs fast enough that "large" should be a decision.
        self.size_box.current(min(1, len(opts) - 1))

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
        try:
            first = grab_frame(self.ffmpeg, self.src, start_s)
        except Exception:
            first = None
        try:
            last = grab_frame(self.ffmpeg, self.src, end_s, box=(END_W, END_H))
        except Exception:
            last = None
        self.root.after(0, self._show_images, first, last)

    def _show_images(self, first, last):
        self.photo = self._paint(self.canvas, first, PREVIEW_W, PREVIEW_H)
        self.end_photo = self._paint(self.end_canvas, last, END_W, END_H)

    def _paint(self, canvas, img, w, h):
        """Draw into a canvas and hand back the PhotoImage — which the caller MUST keep, or Tk
        garbage-collects it and the canvas goes blank."""
        canvas.delete("all")
        if img is None:
            canvas.create_text(w // 2, h // 2, text="(no frame here)",
                               fill=COLORS["text_muted"], font=(FONT_FAMILY, 10))
            return None
        photo = ImageTk.PhotoImage(img)
        canvas.create_image(w // 2, h // 2, image=photo)
        return photo

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
            return
        if not self.src or not self.info or not self.info["has_audio"]:
            return
        fps = self.info["fps"]
        start = self.frame_pos / fps
        k = self._keep_every()
        # Real time: the audio you would get. Slow motion: the source audio under the section,
        # which is what you are judging — the saved clip's audio comes from the same span.
        span = self._frames() * k / fps if k else self._frames() / FPS
        gain = max(0.0, float(self.volume_var.get())) / 100.0
        self.play_btn.configure(text="■  Stop")
        self._playing = True
        threading.Thread(target=self._play_worker, args=(start, span, gain), daemon=True).start()

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

    # -- cost -------------------------------------------------------------------------------------
    def _refresh_cost(self):
        frames = self._frames()
        w, h = self._size()
        if not w:
            self.cost_label.configure(text="")
            return
        tok = tokens_for(w, h, frames)
        still = (w // SIZE_STEP) * (h // SIZE_STEP)
        self.cost_label.configure(
            text=f"This clip is {tok:,} tokens for the DiT — {tok / still:.0f}x what one still of "
                 f"the same size costs, and attention scales with the square of that. It is the "
                 f"reason clips stop at 39 frames.",
            fg=COLORS["warning"] if tok > 6000 else COLORS["text_explain"])
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
            # A muted clip still carries its audio: the _mute suffix is the instruction, and
            # keeping the track means the decision is reversible by rename, not by re-export.
            "with_audio": bool(self.info["has_audio"]),
            "size_index": self.size_box.current(), "motion_index": self.motion_box.current(),
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
        return (f"{mark} {i + 1:2d}  {os.path.basename(job['dst']):<28} "
                f"{job['frames']:>2}f  {job['w']}x{job['h']}  @{job['start']:6.2f}s  "
                f"{sound}{slow}")

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
        if 0 <= job["size_index"] < len(self.size_box["values"]):
            self.size_box.current(job["size_index"])
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
                                       job["with_audio"])
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
    Gizmo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
