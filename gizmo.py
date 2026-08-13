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

PREVIEW_W, PREVIEW_H = 640, 360

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


def output_name(src_path, out_dir, muted):
    """<source stem>_01.mp4, or _01_mute.mp4, skipping past anything already there.

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
                    for suffix in ("", MUTE_SUFFIX))
        if not taken:
            return os.path.join(out_dir, f"{stem}_{i:02d}{MUTE_SUFFIX if muted else ''}.mp4")
    raise RuntimeError("1000 clips from one source — give the output folder a clean start.")


def count_frames(ffmpeg, path):
    """Decoded frame count. The container banner cannot be trusted for this (duration x fps
    rounds), and it is the one value Fizgig refuses a clip over, so it is checked for real."""
    out = _run([ffmpeg, "-hide_banner", "-i", path, "-map", "0:v:0",
                "-c", "copy", "-f", "null", "-"], text=True).stderr
    hits = re.findall(r"frame=\s*(\d+)", out)
    return int(hits[-1]) if hits else None


# --- the app -----------------------------------------------------------------------------------

class Gizmo:
    def __init__(self, root):
        self.root = root
        root.title("Gizmo — video clip prep for Fizgig")
        root.configure(bg=COLORS["bg_deep"])
        root.geometry("1000x900")
        root.minsize(880, 700)

        self.ffmpeg = find_ffmpeg()
        self.src = None
        self.info = None
        self.frame_pos = 0            # in SOURCE frames
        self.photo = None             # a live reference, or Tk garbage-collects the image
        self._scrub_job = None
        self._busy = False
        self._motion_keep = [None]    # parallel to the Motion dropdown; see _keep_every
        self.saved = []

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

    def _button(self, parent, text, command, kind="normal"):
        bg = {"normal": COLORS["bg_hover"], "accent": COLORS["accent"]}[kind]
        fg = COLORS["text_primary"]
        b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                      activebackground=COLORS["accent_hover"], activeforeground=fg,
                      font=(FONT_FAMILY, 10, "bold" if kind == "accent" else "normal"),
                      relief=tk.FLAT, bd=0, padx=14, pady=7, cursor="hand2")
        return b

    def _build(self):
        head = tk.Frame(self.root, bg=COLORS["bg_deep"])
        head.pack(fill=tk.X, padx=16, pady=(14, 10))
        tk.Label(head, text="Gizmo", font=(FONT_FAMILY, 22, "bold"),
                 bg=COLORS["bg_deep"], fg=COLORS["text_primary"]).pack(anchor="w")
        tk.Label(head, text="Cut training clips Fizgig will accept — 24 fps, on the frame grid, "
                            "sized and with the sound sorted out.",
                 font=(FONT_FAMILY, 11), bg=COLORS["bg_deep"],
                 fg=COLORS["text_secondary"]).pack(anchor="w")

        # --- source ---------------------------------------------------------------------------
        c = self._card(self.root, "1. Source video", "Any format, any frame rate, any size.")
        row = tk.Frame(c, bg=COLORS["bg_surface"])
        row.pack(fill=tk.X)
        self._button(row, "Open video…", self.open_video, "accent").pack(side=tk.LEFT)
        self.src_label = tk.Label(row, text="No video open", font=(FONT_FAMILY, 10),
                                  bg=COLORS["bg_surface"], fg=COLORS["text_muted"])
        self.src_label.pack(side=tk.LEFT, padx=12)
        self.src_info = tk.Label(c, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                 bg=COLORS["bg_surface"], fg=COLORS["text_explain"])
        self.src_info.pack(anchor="w", pady=(8, 0))

        # --- scrub ----------------------------------------------------------------------------
        c = self._card(self.root, "2. Find the moment",
                       "The playhead is where the clip STARTS. Step a frame at a time to land it "
                       "exactly.")
        self.canvas = tk.Canvas(c, width=PREVIEW_W, height=PREVIEW_H, bg="#000000",
                                highlightthickness=1, highlightbackground=COLORS["border"])
        self.canvas.pack()
        self.scale = ttk.Scale(c, from_=0, to=100, orient=tk.HORIZONTAL,
                               command=self._on_scrub, style="G.Horizontal.TScale")
        self.scale.pack(fill=tk.X, pady=(10, 6))
        nav = tk.Frame(c, bg=COLORS["bg_surface"])
        nav.pack()
        for label, delta in (("⏮", -1000), ("◀◀", -10), ("◀", -1), ("▶", 1), ("▶▶", 10)):
            self._button(nav, label, lambda d=delta: self.step(d)).pack(side=tk.LEFT, padx=3)
        self.pos_label = tk.Label(c, text="", font=(FONT_FAMILY, 10, "bold"),
                                  bg=COLORS["bg_surface"], fg=COLORS["text_primary"])
        self.pos_label.pack(pady=(8, 0))
        self.span_label = tk.Label(c, text="", font=(FONT_FAMILY, 9),
                                   bg=COLORS["bg_surface"], fg=COLORS["text_explain"])
        self.span_label.pack()

        # --- settings -------------------------------------------------------------------------
        c = self._card(self.root, "3. Clip settings")
        grid = tk.Frame(c, bg=COLORS["bg_surface"])
        grid.pack(fill=tk.X)
        grid.columnconfigure(1, weight=1)

        def label(r, text):
            tk.Label(grid, text=text, font=(FONT_FAMILY, 10), bg=COLORS["bg_surface"],
                     fg=COLORS["text_secondary"]).grid(row=r, column=0, sticky="w", pady=5,
                                                       padx=(0, 12))

        label(0, "Length:")
        self.len_var = tk.StringVar()
        self.len_box = ttk.Combobox(grid, textvariable=self.len_var, state="readonly",
                                    style="G.TCombobox", width=42,
                                    values=[self._length_label(f) for f in GRID_FRAMES])
        self.len_box.current(1)
        self.len_box.grid(row=0, column=1, sticky="w", pady=5)
        self.len_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh_cost())

        label(1, "Size:")
        self.size_var = tk.StringVar()
        self.size_box = ttk.Combobox(grid, textvariable=self.size_var, state="readonly",
                                     style="G.TCombobox", width=42, values=[])
        self.size_box.grid(row=1, column=1, sticky="w", pady=5)
        self.size_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh_cost())

        label(2, "Motion:")
        self.motion_var = tk.StringVar()
        self.motion_box = ttk.Combobox(grid, textvariable=self.motion_var, state="readonly",
                                       style="G.TCombobox", width=42, values=[])
        self.motion_box.grid(row=2, column=1, sticky="w", pady=5)

        label(3, "Sound:")
        snd = tk.Frame(grid, bg=COLORS["bg_surface"])
        snd.grid(row=3, column=1, sticky="w", pady=5)
        # Sticky between saves on purpose: one source video routinely has sections worth training
        # on and sections with a cough, the wrong speaker, or music over the top, so this gets
        # toggled far more often than it gets reset.
        self.mute_var = tk.BooleanVar(value=False)
        for text, val in (("Train on this clip's sound", False), ("Mute — video only", True)):
            tk.Radiobutton(snd, text=text, variable=self.mute_var, value=val,
                           bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
                           selectcolor=COLORS["bg_deep"], activebackground=COLORS["bg_surface"],
                           activeforeground=COLORS["text_primary"], font=(FONT_FAMILY, 10),
                           highlightthickness=0, bd=0).pack(side=tk.LEFT, padx=(0, 16))
        self.sound_note = tk.Label(grid, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                   wraplength=820, bg=COLORS["bg_surface"],
                                   fg=COLORS["text_explain"])
        self.sound_note.grid(row=4, column=1, sticky="w")

        label(5, "Save to:")
        outrow = tk.Frame(grid, bg=COLORS["bg_surface"])
        outrow.grid(row=5, column=1, sticky="ew", pady=5)
        self.out_var = tk.StringVar(value="")
        tk.Entry(outrow, textvariable=self.out_var, bg=COLORS["bg_hover"],
                 fg=COLORS["text_primary"], insertbackground=COLORS["text_primary"],
                 relief=tk.FLAT, font=(FONT_FAMILY, 10)).pack(side=tk.LEFT, fill=tk.X,
                                                              expand=True, ipady=4)
        self._button(outrow, "Browse…", self.pick_output).pack(side=tk.LEFT, padx=(8, 0))

        self.cost_label = tk.Label(c, text="", font=(FONT_FAMILY, 9), justify=tk.LEFT,
                                   wraplength=900, bg=COLORS["bg_surface"],
                                   fg=COLORS["text_explain"])
        self.cost_label.pack(anchor="w", pady=(10, 0))

        # --- save -----------------------------------------------------------------------------
        c = self._card(self.root, "4. Save it",
                       "Then move the playhead and save again — a source video usually gives up "
                       "several good clips.")
        row = tk.Frame(c, bg=COLORS["bg_surface"])
        row.pack(fill=tk.X)
        self.save_btn = self._button(row, "Save clip", self.save_clip, "accent")
        self.save_btn.pack(side=tk.LEFT)
        self.status = tk.Label(row, text="", font=(FONT_FAMILY, 10), bg=COLORS["bg_surface"],
                               fg=COLORS["text_secondary"])
        self.status.pack(side=tk.LEFT, padx=12)
        self.saved_box = tk.Listbox(c, height=6, bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
                                    font=("Consolas", 9), relief=tk.FLAT, highlightthickness=1,
                                    highlightbackground=COLORS["border"],
                                    selectbackground=COLORS["accent"])
        self.saved_box.pack(fill=tk.X, pady=(10, 0))

        self._set_enabled(False)
        self._refresh_cost()

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
        for w in (self.save_btn, self.scale):
            w.configure(state=state)

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
        self.src, self.info = path, info
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
        self.sound_note.configure(
            text="Muting adds _mute to the filename and Fizgig trains that clip's video only. The "
                 "audio stays in the file, so you can still play it back and change your mind by "
                 "renaming.",
            fg=COLORS["text_explain"])

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

    def _update_pos_labels(self):
        fps = self.info["fps"]
        t = self.frame_pos / fps
        self.pos_label.configure(text=f"frame {self.frame_pos}   ·   {t:6.2f} s")
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
        seconds = self.frame_pos / self.info["fps"]
        threading.Thread(target=self._grab_worker, args=(seconds,), daemon=True).start()

    def _grab_worker(self, seconds):
        try:
            img = grab_frame(self.ffmpeg, self.src, seconds)
        except Exception:
            img = None
        self.root.after(0, self._show_image, img)

    def _show_image(self, img):
        self.canvas.delete("all")
        if img is None:
            self.canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2, text="(no frame here)",
                                    fill=COLORS["text_muted"], font=(FONT_FAMILY, 11))
            return
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(PREVIEW_W // 2, PREVIEW_H // 2, image=self.photo)

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

    def save_clip(self):
        if not self.src or self._busy:
            return
        out_dir = self.out_var.get().strip()
        if not out_dir:
            messagebox.showwarning("Gizmo", "Choose a folder to save the clips into.")
            return
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Gizmo", f"Cannot create {out_dir}\n\n{exc}")
            return

        frames = self._frames()
        w, h = self._size()
        muted = self.mute_var.get()
        # A muted clip still carries its audio: the _mute suffix is the instruction, and keeping
        # the track means the decision is reversible by rename rather than by re-export.
        with_audio = bool(self.info["has_audio"])
        dst = output_name(self.src, out_dir, muted)
        cmd = build_export_command(self.ffmpeg, self.src, dst, self.frame_pos / self.info["fps"],
                                   frames, w, h, self._keep_every(), with_audio)

        self._busy = True
        self.save_btn.configure(state=tk.DISABLED)
        self.status.configure(text="Saving…", fg=COLORS["text_secondary"])
        threading.Thread(target=self._save_worker, args=(cmd, dst, frames), daemon=True).start()

    def _save_worker(self, cmd, dst, want_frames):
        try:
            p = _run(cmd, text=True)
            if p.returncode != 0 or not os.path.exists(dst):
                raise RuntimeError((p.stderr or "").strip()[-600:] or "ffmpeg failed")
            # Verified, not assumed: frame count is the one thing Fizgig refuses a clip over, and
            # a source that ends mid-clip silently yields a short one.
            got = count_frames(self.ffmpeg, dst)
            if got is not None and got != want_frames:
                os.remove(dst)
                raise RuntimeError(
                    f"that mark only yields {got} frames, not {want_frames} — the source runs out "
                    f"before the clip does. Move the playhead earlier, or pick a shorter length.")
            err = None
        except Exception as exc:
            err = str(exc)
        self.root.after(0, self._save_done, dst, err)

    def _save_done(self, dst, err):
        self._busy = False
        self.save_btn.configure(state=tk.NORMAL)
        if err:
            self.status.configure(text="Not saved", fg=COLORS["error"])
            messagebox.showerror("Gizmo — could not save that clip", err)
            return
        name = os.path.basename(dst)
        self.saved.append(dst)
        self.saved_box.insert(tk.END, f"{len(self.saved):3d}  {name}")
        self.saved_box.see(tk.END)
        self.status.configure(text=f"Saved {name}", fg=COLORS["success"])


def main():
    root = tk.Tk()
    Gizmo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
