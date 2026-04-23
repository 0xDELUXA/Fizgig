import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu, scrolledtext, simpledialog
import subprocess
import sys
import threading
import json
import os
import signal
import math
import re
import webbrowser
import glob
import time
import socket
from http.server import HTTPServer, SimpleHTTPRequestHandler
from PIL import Image, ImageTk

# Face detection imports (optional - graceful fallback if not installed)
try:
    from face_utils import FaceDetector, crop_to_face, draw_face_boxes, is_face_detection_available
    FACE_DETECTION_AVAILABLE = is_face_detection_available()
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    FaceDetector = None

# Refined Color Palette (Fizgig Visual Style Guide)
COLORS = {
    "bg_deep": "#1E2530",        # Main window background
    "bg_surface": "#252D38",     # Cards, panels, inputs
    "bg_hover": "#2A3542",       # Hover states
    "bg_header": "#1A2028",      # Collapsible section headers

    "text_primary": "#F0F4F8",   # Main text
    "text_secondary": "#8A9BAE", # Labels, hints
    "text_muted": "#5A6B7E",     # Disabled, placeholders

    "accent": "#3B82F6",         # Primary actions, links
    "accent_hover": "#60A5FA",   # Accent hover
    "accent_subtle": "#1E3A5F",  # Accent backgrounds

    "border": "#3A4555",         # Borders, dividers
    "border_focus": "#3B82F6",   # Focus rings

    "success": "#10B981",        # Success states
    "warning": "#F59E0B",        # Warnings
    "error": "#EF4444",          # Errors
}

# Typography
FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"

# Legacy color constants (for backwards compatibility during transition)
BG_COLOR = COLORS["bg_deep"]
FG_COLOR = COLORS["text_primary"]
ACCENT_COLOR = COLORS["accent"]
ENTRY_BG = COLORS["bg_surface"]
BUTTON_ACTIVE = COLORS["bg_hover"]
BORDER_COLOR = COLORS["border"]
ACTIVE_ENTRY_BG = "white"  # Background color for active entry field
ACTIVE_ENTRY_FG = "black"  # Text color for active entry field


class ToolTip:
    """Simple tooltip class for tkinter widgets"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window:
            return
        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background=COLORS["bg_surface"], foreground=COLORS["text_primary"],
                        relief=tk.SOLID, borderwidth=1,
                        font=(FONT_FAMILY, 9), padx=8, pady=6)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class CollapsibleFrame(tk.Frame):
    """
    A frame that can be collapsed/expanded with a header.

    Features:
    - Click header to toggle
    - Arrow indicator (▶/▼)
    - Optional badge showing field status
    - Maintains child widget state when collapsed
    - All child widgets remain accessible via parent.entries[]

    Styled as a Start-tab-style card (bg_surface body, bordered outer frame).
    """

    def __init__(self, parent, title, default_expanded=True, badge_callback=None):
        """
        Args:
            parent: Parent widget
            title: Section title text
            default_expanded: Whether section starts expanded
            badge_callback: Optional function returning (filled, total) tuple for badge
        """
        super().__init__(
            parent,
            bg=COLORS["bg_surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            bd=0,
        )
        self.expanded = default_expanded
        self.title = title
        self.badge_callback = badge_callback

        # Create header frame
        self.header = tk.Frame(
            self,
            bg=COLORS["bg_header"],
            cursor="hand2"
        )
        self.header.pack(fill=tk.X)

        # Arrow indicator
        self.arrow = tk.Label(
            self.header,
            text="▼" if self.expanded else "▶",
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_header"],
            cursor="hand2"
        )
        self.arrow.pack(side=tk.LEFT, padx=(16, 10), pady=12)

        # Title label — matches Start-tab card headers at 12pt bold
        self.title_label = tk.Label(
            self.header,
            text=title,
            font=(FONT_FAMILY, 12, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_header"],
            cursor="hand2"
        )
        self.title_label.pack(side=tk.LEFT, pady=12)

        # Badge label (shows filled/total fields)
        self.badge = tk.Label(
            self.header,
            text="",
            font=(FONT_FAMILY, 9),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_header"],
            cursor="hand2"
        )
        self.badge.pack(side=tk.RIGHT, padx=(8, 16), pady=12)

        # Content frame — bg_surface, padded from the card edge so children don't
        # touch the border. Children grid into this directly.
        self.content = tk.Frame(self, bg=COLORS["bg_surface"])
        if self.expanded:
            self.content.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 12))

        # Bind click events to all header elements
        for widget in [self.header, self.arrow, self.title_label, self.badge]:
            widget.bind("<Button-1>", self.toggle)
            widget.bind("<Enter>", self._on_header_enter)
            widget.bind("<Leave>", self._on_header_leave)

    def _on_header_enter(self, event=None):
        """Highlight header on hover"""
        hover_color = COLORS["bg_hover"]
        self.header.configure(bg=hover_color)
        self.arrow.configure(bg=hover_color)
        self.title_label.configure(bg=hover_color)
        self.badge.configure(bg=hover_color)

    def _on_header_leave(self, event=None):
        """Reset header color on mouse leave"""
        header_color = COLORS["bg_header"]
        self.header.configure(bg=header_color)
        self.arrow.configure(bg=header_color)
        self.title_label.configure(bg=header_color)
        self.badge.configure(bg=header_color)

    def toggle(self, event=None):
        """Toggle between expanded and collapsed states"""
        if self.expanded:
            self.content.pack_forget()
            self.arrow.config(text="▶")
        else:
            self.content.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 12))
            self.arrow.config(text="▼")
        self.expanded = not self.expanded

    def expand(self):
        """Expand the section if collapsed"""
        if not self.expanded:
            self.toggle()

    def collapse(self):
        """Collapse the section if expanded"""
        if self.expanded:
            self.toggle()

    def update_badge(self, filled=None, total=None):
        """Update the badge showing filled/total fields"""
        if self.badge_callback:
            filled, total = self.badge_callback()
        if filled is not None and total is not None:
            self.badge.config(text=f"[{filled}/{total}]")
        else:
            self.badge.config(text="")

    def get_content_frame(self):
        """Return the content frame where child widgets should be added"""
        return self.content


# Architecture configurations
ARCHITECTURES = {
    "Flux 2 Klein Base 9B": {
        "train_script": "FizgigIndependent/src/fizgig/scripts/train.py",
        "cache_latents_script": "FizgigIndependent/src/fizgig/scripts/cache_latents.py",
        "cache_text_script": "FizgigIndependent/src/fizgig/scripts/cache_text.py",
        "network_module": "fizgig.networks.lora_klein",
        "use_fizgig_venv": True,
        "timestep_sampling": "flux2_shift",
        "discrete_flow_shift": None,
        "weighting_scheme": "none",
        "blocks_swap_max": 16,
        "fp8_text_encoder_flag": "--fp8_text_encoder",
        "uses_clip": False,
        "uses_t5": False,
        "uses_text_encoder": True,
        "uses_model_type": False,
        "uses_model_version": True,
        "model_version": "klein-base-9b",
        "vae_label": "AE Model (ae.safetensors from FLUX.2-dev — NOT the Diffusers subfolder VAE)",
        "text_encoder_label": "Text Encoder (Qwen3-8B)",
        "is_distilled": False,  # Recommended for training
        "supports_weighting_scheme": False,  # Architecture uses hardcoded "none"
        "supports_discrete_flow_shift": False,  # Uses flux2_shift automatic
        # Sample generation settings
        "supports_samples": True,
        "sample_guidance_default": 3.5,
        "sample_cfg_default": 3.5,
        "sample_flow_shift_default": None,
        "sample_steps_default": 20,
        "sample_width_default": 1024,
        "sample_height_default": 1024,
    },
}

ARCHITECTURE_LIST = list(ARCHITECTURES.keys())

# Fizgig installation directory (where this GUI lives)
FIZGIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory for custom presets (per architecture)
PRESETS_DIR = os.path.join(os.path.dirname(__file__), "presets")

# Snapshot of settings from the most recent training launch — restorable via "Load Last Train" button
LAST_TRAIN_FILE = os.path.join(PRESETS_DIR, ".last_train_settings.json")

# Built-in presets — always available in the Load Preset dropdown, prefixed with ✨ to distinguish
# from user-saved presets. Defined in code so they ship with the app and can't be deleted accidentally.
# Tune these as empirical findings accumulate.
BUILT_IN_PRESETS = {
    "✨ Old Reliable (rank 16, single subject)": {
        "NETWORK_DIM": 16, "NETWORK_ALPHA": 16, "LEARNING_RATE": 1e-4,
        "MAX_TRAIN_EPOCHS": 55, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "1e-4", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Full Model", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
    },
    "✨ Identity (rank 4, single subject)": {
        "NETWORK_DIM": 4, "NETWORK_ALPHA": 4, "LEARNING_RATE": 4e-4,
        "MAX_TRAIN_EPOCHS": 15, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "1e-4", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Identity", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
    },
    "✨ Identity (rank 8, harder dataset)": {
        "NETWORK_DIM": 8, "NETWORK_ALPHA": 8, "LEARNING_RATE": 4e-4,
        "MAX_TRAIN_EPOCHS": 20, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "1e-4", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Identity", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
    },
    "✨ Multi-Character (rank 16, multi character or concept)": {
        "NETWORK_DIM": 16, "NETWORK_ALPHA": 16, "LEARNING_RATE": 2e-4,
        "MAX_TRAIN_EPOCHS": 50, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "1e-4", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Identity", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
    },
    "✨ Style (late timesteps)": {
        "NETWORK_DIM": 4, "NETWORK_ALPHA": 4, "LEARNING_RATE": 4e-4,
        "MAX_TRAIN_EPOCHS": 15, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "1e-5", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Style", "MIN_TIMESTEP": "0", "MAX_TIMESTEP": "400",
        "OPTIMIZER_TYPE": "adamw8bit",
    },
    "✨ Style+Composition (all timesteps)": {
        "NETWORK_DIM": 4, "NETWORK_ALPHA": 4, "LEARNING_RATE": 4e-4,
        "MAX_TRAIN_EPOCHS": 15, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "1e-5", "ADAPTIVE_LR_MAX": "4e-4",
        "TARGET_LAYERS": "Style+Composition", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
    },
}

# Directory for dataset configurations
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

# Directory for cache (latents and text encoder outputs)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# Directory for output LoRAs
OUTPUT_LORAS_DIR = os.path.join(os.path.dirname(__file__), "output_loras")

# File for storing last-used folder paths
LAST_USED_FILE = os.path.join(os.path.dirname(__file__), ".last_used.json")


def load_last_used():
    """Load last-used folder paths from config file"""
    defaults = {
        "image_prep_source": "",
        "image_folder": "",  # Start tab: training image folder (shared with Captions)
        "caption_trigger": "trigger_word",
        "dataset_cache_dir": CACHE_DIR,
        "sample_prompt": "A high quality photo",
    }
    if os.path.exists(LAST_USED_FILE):
        try:
            with open(LAST_USED_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                defaults.update(saved)
        except Exception:
            pass
    # Migrate pre-Start-tab last_used files: if image_folder isn't set but one
    # of the legacy keys (caption_folder / dataset_image_dir / image_prep_source)
    # has a value, seed image_folder from the best candidate.
    if not defaults.get("image_folder"):
        for legacy_key in ("caption_folder", "dataset_image_dir", "image_prep_source"):
            legacy_val = defaults.get(legacy_key, "")
            if legacy_val:
                defaults["image_folder"] = legacy_val
                break
    return defaults


def save_last_used(data):
    """Save last-used folder paths to config file"""
    try:
        with open(LAST_USED_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Preferences — centralized model and directory paths (Klein 9B only for now)
# ---------------------------------------------------------------------------

PREFS_FILE = os.path.join(os.path.dirname(__file__), "prefs.json")
HELP_FILE = os.path.join(os.path.dirname(__file__), "help.json")
_FIZGIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Prefs whose VALUE is a directory we want to keep portable across repo
# clones/moves. When saved to disk, paths inside _FIZGIG_DIR are stored as
# relative strings (with forward slashes); when loaded, they're resolved back
# to absolute so every consumer works unchanged. Paths outside the repo (e.g.
# a user pointing Cache to another drive) stay absolute on both sides.
#
# Model paths (base_dit, vae, text_encoder, ...) are NOT in this set — they
# point to external HuggingFace downloads and are always absolute.
#
# Note: `dataset_dir` is deliberately NOT a pref — the dataset TOML path is
# fully hardcoded to FIZGIG_DIR/dataset/ via the DATASET_DIR module constant
# (Dataset tab auto-saves to Fizgig_train.toml; no browse UI anywhere), so
# exposing it in Preferences was dead weight.
_PORTABLE_DIR_KEYS = {"lora_output_dir", "profiles_dir", "cache_dir"}


def _resolve_pref_path(value: str) -> str:
    """Convert a possibly-relative stored path to an absolute path by joining
    with _FIZGIG_DIR. Absolute paths are returned unchanged."""
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(_FIZGIG_DIR, value))


def _serialize_pref_path(value: str) -> str:
    """Store a path as relative-to-_FIZGIG_DIR if it lives inside the repo;
    otherwise store as an absolute path. Uses forward slashes for
    cross-platform portability of the JSON file."""
    if not value:
        return value
    try:
        abs_value = os.path.abspath(value)
        fizgig_abs = os.path.abspath(_FIZGIG_DIR)
        rel = os.path.relpath(abs_value, fizgig_abs)
    except ValueError:
        # Windows: path is on a different drive than the repo — relpath raises.
        return os.path.abspath(value).replace(os.sep, "/")
    if rel.startswith("..") or os.path.isabs(rel):
        # Outside the repo — keep absolute.
        return os.path.abspath(value).replace(os.sep, "/")
    return rel.replace(os.sep, "/")


DEFAULT_PREFS = {
    # Model paths (absolute — point to external model downloads).
    # Blank on first launch; user fills these in via the Preferences tab. Each
    # row has a "Download" link that opens the correct HuggingFace repo.
    "base_dit": "",
    "distilled_dit": "",
    "vae": "",
    "text_encoder": "",
    # Output directories — relative to repo root, portable across clones/moves.
    # Resolved to absolute in load_prefs(); in-memory pref values are absolute.
    # All three live as top-level folders inside the repo:
    #   FizgigIndependent/output_loras/
    #   FizgigIndependent/profiles/
    #   FizgigIndependent/cache/
    # (Dataset TOMLs are always in FIZGIG_DIR/dataset/ via DATASET_DIR — not
    # a pref; see _PORTABLE_DIR_KEYS note above.)
    "lora_output_dir": "output_loras",
    "profiles_dir": "profiles",
    "cache_dir": "cache",
    # Inference DiT block swap — int 0-16 for Klein 9B. 0 = no swap (needs
    # 32GB+ VRAM), 16 = max swap (fits a 16GB card via PCIe offload). Applies
    # to Repair Studio, Profiler, and Extractor. The Training tab has its own
    # separate BLOCKS_SWAP setting that's not affected.
    "inference_blocks_to_swap": 0,
}


def _auto_detect_blocks_to_swap() -> int:
    """Pick a DiT block-swap preset based on the GPU's total VRAM.

    Only called when the user hasn't saved an explicit preference yet.
    Returns the leading integer for the swap-preset labels (0/4/8/12/16).
    """
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            if vram_gb >= 28:
                return 0   # 32 GB+ — no swap needed
            if vram_gb >= 20:
                return 4   # 24 GB cards (RTX 3090 / 4090)
            if vram_gb >= 14:
                return 8   # 16-20 GB
            if vram_gb >= 10:
                return 12  # 12-14 GB
            return 16      # <10 GB — maximum swap
    except Exception:
        pass
    return 0  # safe fallback


def load_prefs() -> dict:
    """Load user preferences from prefs.json, falling back to defaults.
    Relative portable-dir paths are resolved to absolute for in-memory use."""
    prefs = dict(DEFAULT_PREFS)
    user_set_swap = False
    if os.path.exists(PREFS_FILE):
        try:
            with open(PREFS_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                user_set_swap = "inference_blocks_to_swap" in saved
                prefs.update(saved)
        except Exception:
            pass
    # Auto-detect block swap from GPU VRAM if user hasn't explicitly chosen one.
    first_run = not os.path.exists(PREFS_FILE)
    if not user_set_swap:
        prefs["inference_blocks_to_swap"] = _auto_detect_blocks_to_swap()
    # Resolve portable directory paths to absolute.
    for key in _PORTABLE_DIR_KEYS:
        if key in prefs and isinstance(prefs[key], str):
            prefs[key] = _resolve_pref_path(prefs[key])
    # On first run, persist defaults so prefs.json exists immediately.
    if first_run:
        save_prefs(prefs)
    return prefs


def save_prefs(prefs: dict) -> None:
    """Save preferences to prefs.json. Portable-dir paths inside the repo are
    stored as relative strings so a cloned/moved repo finds its own defaults."""
    to_save = {}
    for key, value in prefs.items():
        if key in _PORTABLE_DIR_KEYS and isinstance(value, str):
            to_save[key] = _serialize_pref_path(value)
        else:
            to_save[key] = value
    try:
        with open(PREFS_FILE, 'w', encoding='utf-8') as f:
            json.dump(to_save, f, indent=2)
    except Exception:
        pass


# Maps Training tab settings keys to prefs keys — entries with matching keys
# will be bound to the shared prefs StringVar (two-way sync with Preferences tab)
SETTING_TO_PREF = {
    "DIT_MODEL": "base_dit",
    "VAE_MODEL": "vae",
    "TEXT_ENCODER": "text_encoder",
    "LORA_OUTPUT_DIR": "lora_output_dir",
}


# Default preset for Klein Base 9B — the only supported architecture.
# Applied on first launch and when "Reset to Defaults" is used.
PRESETS = {
    "Flux 2 Klein Base 9B": {
        "LEARNING_RATE": 0.0004,       # Rank 4:4 optimal LR for 80+ image datasets
        "NETWORK_DIM": 4,              # Low rank — identity signal only
        "NETWORK_ALPHA": 4,            # 1:1 alpha:rank ratio
        "MAX_TRAIN_EPOCHS": 12,        # Fast convergence at low rank
        "OPTIMIZER_TYPE": "adamw8bit",
        "TIMESTEP_SAMPLING": "flux2_shift",
        "DISCRETE_FLOW_SHIFT": "0",
        "WEIGHTING_SCHEME": "none",
        "BLOCKS_SWAP": "auto",
        # Model paths come from Preferences at runtime — leave blank in the preset.
        "VAE_MODEL": "",
        "TEXT_ENCODER": "",
        "DIT_MODEL": "",
        "FP8": True,
        "SCALED": True,  # BF16 model, use fp8_scaled for memory efficiency
    },
}


class LoRATrainerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Fizgig — Klein 9B LoRA Studio")
        master.geometry("1280x1024")
        master.minsize(1100, 800)  # ensures all tabs visible at top + tab content not cut off
        master.configure(bg=BG_COLOR)

        self.current_process = None
        self.training_thread = None
        self.process_group_id = None
        self.user_scrolled = False  # Flag for manual console scrolling
        self.samples_watcher_running = False  # For live gallery updates
        self.samples_watcher_thread = None

        # HTTP server for samples gallery (avoids CORS issues)
        self.gallery_server = None
        self.gallery_server_port = None
        self.gallery_server_thread = None

        # Load last-used folder paths
        self.last_used = load_last_used()

        # Load user preferences (model paths, output directories)
        self.prefs = load_prefs()
        self.prefs_vars = {}
        for key, default in DEFAULT_PREFS.items():
            var = tk.StringVar(value=self.prefs.get(key, default))
            var.trace_add("write", lambda *a, k=key: self._save_pref(k))
            self.prefs_vars[key] = var

        # Caption Generator variables
        # Unified training-folder var — shared between Start tab (authoritative),
        # Captions tab, and the Fizgig_train.toml auto-saver. Replaces the old
        # caption_folder_var / dataset_image_dir_var pair + their propagation.
        self.image_folder_var = tk.StringVar(value=self.last_used.get("image_folder", ""))
        self.caption_text_var = tk.StringVar(value=self.last_used.get("caption_trigger", "trigger_word"))
        self.overwrite_captions_var = tk.BooleanVar(value=True)
        self.skip_bilingual_var = tk.BooleanVar(value=True)

        # Image Converter variables — source is unified with self.image_folder_var
        # (the Start-tab picker); only the output folder is Image-Prep-specific.
        self.convert_output_var = tk.StringVar()
        self.max_size_var = tk.StringVar(value="1024")
        self.delete_originals_var = tk.BooleanVar(value=True)

        # Prep Mode and Face Cropping variables
        self.prep_mode_var = tk.StringVar(value=self.last_used.get("prep_mode", "Auto Prep (Face Crops)"))
        self.face_selection_var = tk.StringVar(value="Largest Face")
        self.face_padding_var = tk.StringVar(value="20")

        # Face detector instance (lazy loaded)
        self._face_detector = None

        # Dataset Manager variables
        # Hardcoded: dataset name and type are fixed; num_repeats always 1; cache dir comes from Preferences.
        self.dataset_name_var = tk.StringVar(value="Fizgig_train")
        self.dataset_type_var = tk.StringVar(value="Image with Captions")
        # dataset_image_dir_var removed — unified into self.image_folder_var above.
        self.dataset_video_dir_var = tk.StringVar()
        self.dataset_cache_dir_var = tk.StringVar()  # legacy/back-compat — UI removed; cache dir now lives in prefs_vars["cache_dir"]
        self.dataset_caption_ext_var = tk.StringVar(value=".txt")
        self.dataset_jsonl_file_var = tk.StringVar()
        self.dataset_megapixels_var = tk.StringVar(value="1.0")
        self.dataset_batch_size_var = tk.StringVar(value="1")
        self.dataset_num_repeats_var = tk.StringVar(value="1")
        self.dataset_enable_bucket_var = tk.BooleanVar(value=True)
        self.dataset_no_upscale_var = tk.BooleanVar(value=True)
        self.dataset_target_frames_var = tk.StringVar(value="1, 25, 45")
        self.dataset_frame_extraction_var = tk.StringVar(value="head")
        self.dataset_source_fps_var = tk.StringVar(value="30.0")

        # Ensure all three portable output directories exist (honours user's
        # prefs overrides when present; otherwise creates the defaults:
        # output_loras/, profiles/, cache/ inside FIZGIG_DIR).
        for pref_key in ("lora_output_dir", "profiles_dir", "cache_dir"):
            try:
                os.makedirs(self.prefs_vars[pref_key].get(), exist_ok=True)
            except Exception:
                pass
        # Dataset dir is hardcoded to FIZGIG_DIR/dataset/ (never a pref).
        # The Dataset tab auto-writes Fizgig_train.toml here via
        # auto_save_dataset_config_silent(); no example template needed.
        os.makedirs(DATASET_DIR, exist_ok=True)

        # Add traces to auto-save last-used folder paths and settings.
        # image_folder_var is the single source of truth shared between the
        # Start tab, Captions tab, and the Fizgig_train.toml auto-saver —
        # no propagation helpers needed.
        self.image_folder_var.trace_add("write", self._save_last_used_paths)
        self.caption_text_var.trace_add("write", self._save_last_used_paths)
        self.prep_mode_var.trace_add("write", self._save_last_used_paths)
        # Auto-save the dataset TOML on every relevant change (no Save button needed)
        def _auto_save_ds(*_a):
            if hasattr(self, "auto_save_dataset_config_silent"):
                self.auto_save_dataset_config_silent()
        for _v in (self.image_folder_var, self.dataset_caption_ext_var,
                   self.dataset_megapixels_var, self.dataset_batch_size_var,
                   self.dataset_enable_bucket_var, self.dataset_no_upscale_var):
            _v.trace_add("write", _auto_save_ds)

        # Initialize settings with default values, including conversion settings
        # Klein 9B Base is the only supported architecture. A few legacy keys
        # (CLIP_MODEL / T5_MODEL / MODEL_TYPE) remain as empty defaults so dead
        # code paths gated behind `config["uses_*"]` flags don't KeyError.
        default_arch = "Flux 2 Klein Base 9B"
        self.settings = {
            "ARCHITECTURE": default_arch,
            "DATASET_CONFIG": os.path.join(DATASET_DIR, "Fizgig_train.toml"),
            # Model paths resolved at runtime from prefs_vars — blank fallback.
            "VAE_MODEL": "",
            "CLIP_MODEL": "",
            "T5_MODEL": "",
            "TEXT_ENCODER": "",
            "DIT_MODEL": "",
            "LORA_OUTPUT_DIR": OUTPUT_LORAS_DIR,
            "LORA_NAME": "LoraName_TokenName_k9b",
            "MODEL_TYPE": "",
            "LEARNING_RATE": 4e-4,
            "LORA_LR_RATIO": 1,
            "NETWORK_DIM": 4,
            "NETWORK_ALPHA": 4,
            "MAX_TRAIN_EPOCHS": 12,
            "SAVE_EVERY_N_EPOCHS": 1,
            "SEED": 42,
            "BLOCKS_SWAP": "auto",  # Klein valid range 0-16; "auto" detects from GPU
            "RESUME_TRAINING": "",
            "OPTIMIZER_TYPE": "adamw8bit",
            "OPTIMIZER_ARGS": "",
            "GRADIENT_ACCUMULATION": 1,  # Effective batch size = batch × this
            "MAX_GRAD_NORM": 1.0,  # Gradient clipping (0 to disable)
            "NETWORK_DROPOUT": 0,  # LoRA dropout for regularization
            "ATTENTION_MECHANISM": "sdpa",  # Default attention (was "none", causing duplicate flags)
            "LOGGING_DIR": "",
            "LOG_WITH": "none",
            "LOG_PREFIX": "",
            "IMG_IN_TXT_IN_OFFLOADING": False,
            "LR_SCHEDULER": "constant",
            "LR_WARMUP_STEPS": "",
            "LR_DECAY_STEPS": "",
            "ADAPTIVE_LR": False,
            "ADAPTIVE_LR_MIN": "1e-5",
            "ADAPTIVE_LR_MAX": "4e-4",
            "CONTEXT_LORA_PATH": "",
            "CONTEXT_LORA_STRENGTH": "1.0",
            "TIMESTEP_SAMPLING": "shift",
            "DISCRETE_FLOW_SHIFT": "3.0",
            "SIGMOID_SCALE": "1.0",
            "MIN_TIMESTEP": "",
            "MAX_TIMESTEP": "",
            "PRESERVE_DISTRIBUTION": False,
            "WEIGHTING_SCHEME": "none",
            "LOGIT_MEAN": "0.0",
            "LOGIT_STD": "1.0",
            "MODE_SCALE": "1.29",
            "METADATA_TITLE": "",
            "METADATA_AUTHOR": "",
            "METADATA_DESCRIPTION": "",
            "METADATA_LICENSE": "",
            "METADATA_TAGS": "",
            "FP8": True,  # Default FP8 setting (--fp8_base)
            "SCALED": True,  # Default Scaled setting (--fp8_scaled, recommended with fp8_base)
            "FP8_TEXT_ENCODER": True,  # FP8 for text encoder (T5/LLM)
            # Sample generation settings
            "SAMPLE_ENABLED": True,
            "SAMPLE_PROMPT": "A high quality photo",
            "SAMPLE_WIDTH": 1024,
            "SAMPLE_HEIGHT": 1024,
            "SAMPLE_STEPS": 25,
            "SAMPLE_SEED": 1234,
            "SAMPLE_EVERY_N_EPOCHS": 1,
            "SAMPLE_EVERY_N_STEPS": 0,
            "SAMPLE_AT_FIRST": True,
            "SAMPLE_FLOW_SHIFT": 3.0,
            "SAMPLE_GUIDANCE": 4.0,
            "SAMPLE_NEGATIVE": "bad photo, bad, low quality",
            "SAMPLE_CFG_SCALE": 1.0,
            # Florence captioning settings
            "CAPTION_TRIGGER_WORD": "",
            "CAPTION_MODEL": "MiaoshouAI/Florence-2-base-PromptGen",
            "CAPTION_TASK": "<DETAILED_CAPTION>",
            "CAPTION_MAX_TOKENS": 256,
        }

        # Backing store for the active dataset config path. The Model Paths section was removed
        # from the Training tab; Dataset-tab callbacks write here, training command builders read here.
        self._dataset_config_var = tk.StringVar(value=self.settings["DATASET_CONFIG"])

        # Override with last-used LoRA output directory if available
        if self.last_used.get("lora_output_dir"):
            self.settings["LORA_OUTPUT_DIR"] = self.last_used["lora_output_dir"]

        self.optimizer_types = ["adamw", "adamw8bit", "adafactor", "torch.optim.AdamW", "bitsandbytes.optim.AdEMAMix8bit", "bitsandbytes.optim.PagedAdEMAMix8bit", "came"]

        self.setup_styles()

        # Create notebook and tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tabs ordered by natural workflow: Start -> Prep -> Caption -> Train -> everything else.
        # The old Dataset tab was folded into Training (Other Options → Dataset subsection);
        # its Image Directory field is now the Start tab's folder picker (shared with Captions).
        self.start_tab = ttk.Frame(self.notebook)
        self.start_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.start_tab, text="1. Start")

        self.image_converter_tab = ttk.Frame(self.notebook)
        self.image_converter_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.image_converter_tab, text="2. Image Prep")

        self.caption_gen_tab = ttk.Frame(self.notebook)
        self.caption_gen_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.caption_gen_tab, text="3. Captions")

        self.samples_tab = ttk.Frame(self.notebook)
        self.samples_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.samples_tab, text="4. Samples")

        self.training_tab = ttk.Frame(self.notebook)
        self.training_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.training_tab, text="5. Training")

        self.profiler_tab = ttk.Frame(self.notebook)
        self.profiler_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.profiler_tab, text="Profiler")

        self.repair_studio_tab = ttk.Frame(self.notebook)
        self.repair_studio_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.repair_studio_tab, text="Repair Studio")

        self.explorer_tab = ttk.Frame(self.notebook)
        self.explorer_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.explorer_tab, text="LoRA the Explorer")

        self.extract_tab = ttk.Frame(self.notebook)
        self.extract_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.extract_tab, text="Extract")

        self.prefs_tab = ttk.Frame(self.notebook)
        self.prefs_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.prefs_tab, text="Preferences")

        # Initialize tab contents
        self.entries = {}
        self.labels = {}  # Store label widgets for dynamic updates
        self.rows = {}    # Store row widgets for show/hide
        self.create_start_tab()
        self.create_training_settings()
        self.create_caption_generator()
        self.create_image_converter()
        self.create_samples_settings()
        self.create_profiler_tab()
        self.create_repair_studio_tab()
        self.create_explorer_tab()
        self.create_extract_tab()
        self.create_prefs_tab()

        # Florence model state (lazy loaded)
        self.florence_model = None
        self.florence_processor = None
        self.florence_device = None
        self.captioning_stop_flag = False
        self.caption_thumbnails = {}
        self.current_caption_page = 0
        self.images_per_page = 12
        self.selected_images = set()

        # Load architecture defaults first (populates optimizer / fp8 / timestep
        # fields that the built-in presets don't explicitly set), then overlay
        # the first built-in preset ("Old Reliable") on top. Load Settings From
        # Last Train still works — it just overrides whenever the user clicks it.
        self.load_default_preset(show_message=False)
        try:
            first_preset = next(iter(BUILT_IN_PRESETS))
            self._apply_preset_values(BUILT_IN_PRESETS[first_preset])
            self.custom_preset_var.set(first_preset)
        except Exception:
            pass

        # Create context menu for copying console text
        self.context_menu = Menu(self.master, tearoff=0)
        self.context_menu.add_command(label="Copy", command=self.copy_selected_text)

        # Bind tab change event to load caption images when visiting Captions tab
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.caption_images_loaded = False  # Track if we've loaded images for current folder

        # Reset flag when folder changes so images reload on next tab visit
        self.image_folder_var.trace_add("write", self._on_caption_folder_changed)

        # Auto-save dataset config on startup if all fields are valid
        # This ensures training works immediately without manual "Save and Activate"
        self.auto_save_dataset_config_silent()

    def _on_caption_folder_changed(self, *args):
        """Reset caption images loaded flag when folder changes"""
        self.caption_images_loaded = False

    def on_tab_changed(self, event):
        """Handle notebook tab changes"""
        selected_tab = self.notebook.select()
        tab_text = self.notebook.tab(selected_tab, "text")

        # When Captions tab is selected, load images if folder is set
        if tab_text == "3. Captions":
            folder = self.image_folder_var.get()
            if folder and os.path.isdir(folder) and not self.caption_images_loaded:
                self.refresh_caption_images()
                self.caption_images_loaded = True

        # When leaving Repair Studio or Explorer, unload pipeline to free VRAM
        if tab_text != "Repair Studio":
            self._unload_repair_studio_models()
        if tab_text != "LoRA the Explorer":
            self._unload_explorer_models()

    def remove_focus(self, event):
        """Remove focus from active widget when clicking background"""
        self.master.focus_set()

    def _open_in_file_manager(self, path: str):
        """Open a file or folder in the OS's native file manager.

        Uses os.startfile on Windows, `open` on macOS, and `xdg-open` on Linux.
        `os.name == 'posix'` matches both macOS and Linux, so we fall back to
        sys.platform for the Mac/Linux split.
        """
        try:
            if os.name == 'nt':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', path])
            else:
                subprocess.run(['xdg-open', path])
        except Exception as e:
            messagebox.showerror("Open Failed", f"Could not open:\n{path}\n\n{e}")

    def _save_last_used_paths(self, *args):
        """Save last-used folder paths and settings to config file"""
        data = {
            "prep_mode": self.prep_mode_var.get(),
            "image_folder": self.image_folder_var.get(),
            "caption_trigger": self.caption_text_var.get(),
            "dataset_cache_dir": self.dataset_cache_dir_var.get(),
        }
        # Save architecture if variable exists
        if hasattr(self, 'architecture_var'):
            data["architecture"] = self.architecture_var.get()
        # Save sample prompt if widget exists
        if hasattr(self, 'sample_prompt_text'):
            data["sample_prompt"] = self.sample_prompt_text.get("1.0", tk.END).strip()
        # Save LoRA output directory if entry exists
        if "LORA_OUTPUT_DIR" in self.entries:
            data["lora_output_dir"] = self.entries["LORA_OUTPUT_DIR"].get()
        save_last_used(data)

    def _save_pref(self, key):
        """Save a single pref value back to prefs.json."""
        if key in self.prefs_vars:
            self.prefs[key] = self.prefs_vars[key].get()
            save_prefs(self.prefs)

    def setup_styles(self):
        """Set up styles for refined dark theme (Fizgig Visual Style Guide)"""
        style = ttk.Style()
        style.theme_use("clam")

        # Base styles with new palette
        style.configure(".",
            background=COLORS["bg_deep"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10)
        )
        style.configure("TFrame", background=COLORS["bg_deep"])
        style.configure("TLabel",
            background=COLORS["bg_deep"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10)
        )

        # Surface style for cards/panels
        style.configure("Surface.TFrame", background=COLORS["bg_surface"])

        # Collapsible header style
        style.configure("CollapsibleHeader.TFrame", background=COLORS["bg_header"])

        # Section header label style
        style.configure("SectionHeader.TLabel",
            background=COLORS["bg_deep"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 12, "bold")
        )

        # Secondary label style (for field labels)
        style.configure("Secondary.TLabel",
            background=COLORS["bg_deep"],
            foreground=COLORS["text_secondary"],
            font=(FONT_FAMILY, 10)
        )

        # Default button (Secondary style)
        style.configure(
            "TButton",
            background=COLORS["bg_surface"],
            foreground=COLORS["text_primary"],
            bordercolor=COLORS["border"],
            borderwidth=1,
            focusthickness=2,
            focuscolor=COLORS["accent"],
            padding=[16, 8],
            font=(FONT_FAMILY, 10, "bold")
        )
        style.map(
            "TButton",
            background=[("active", COLORS["bg_hover"]), ("pressed", COLORS["bg_hover"])],
            foreground=[("active", COLORS["text_primary"]), ("pressed", COLORS["text_primary"])]
        )

        # Primary button (accent color)
        style.configure(
            "Primary.TButton",
            background=COLORS["accent"],
            foreground="white",
            bordercolor=COLORS["accent"],
            borderwidth=1,
            focusthickness=2,
            focuscolor=COLORS["accent_hover"],
            padding=[16, 8],
            font=(FONT_FAMILY, 10, "bold")
        )
        style.map(
            "Primary.TButton",
            background=[("active", COLORS["accent_hover"]), ("pressed", COLORS["accent_hover"])],
            foreground=[("active", "white"), ("pressed", "white")]
        )

        # Danger button (for stop actions)
        style.configure(
            "Danger.TButton",
            background=COLORS["error"],
            foreground="white",
            bordercolor=COLORS["error"],
            borderwidth=1,
            focusthickness=2,
            focuscolor=COLORS["error"],
            padding=[16, 8],
            font=(FONT_FAMILY, 10, "bold")
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#DC2626"), ("pressed", "#DC2626")],
            foreground=[("active", "white"), ("pressed", "white")]
        )

        # Checkbutton
        style.configure("TCheckbutton",
            background=COLORS["bg_deep"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10)
        )
        style.map("TCheckbutton",
            background=[("active", COLORS["bg_deep"])],
            foreground=[("active", COLORS["text_primary"])]
        )
        style.configure("TRadiobutton",
            background=COLORS["bg_deep"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10)
        )
        style.map("TRadiobutton",
            background=[("active", COLORS["bg_deep"])],
            foreground=[("active", COLORS["text_primary"])]
        )
        style.configure("Surface.TRadiobutton",
            background=COLORS["bg_surface"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10)
        )
        style.map("Surface.TRadiobutton",
            background=[("active", COLORS["bg_surface"])],
            foreground=[("active", COLORS["text_primary"])]
        )

        # Surface checkbutton (for use inside collapsible sections)
        style.configure("Surface.TCheckbutton",
            background=COLORS["bg_surface"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10)
        )
        style.map("Surface.TCheckbutton",
            background=[("active", COLORS["bg_surface"])],
            foreground=[("active", COLORS["text_primary"])]
        )

        # Notebook (tabs)
        style.configure("TNotebook",
            background=COLORS["bg_deep"],
            borderwidth=0
        )
        style.configure("TNotebook.Tab",
            background=COLORS["bg_surface"],
            foreground=COLORS["text_primary"],
            padding=[12, 6],
            font=(FONT_FAMILY, 11, "bold")
        )
        style.map("TNotebook.Tab",
            background=[("selected", COLORS["accent"])],
            foreground=[("selected", "white")]
        )

        # Entry field — explicit insert cursor settings so the caret is visible on click
        style.configure(
            "TEntry",
            fieldbackground=COLORS["bg_surface"],
            foreground=COLORS["text_primary"],
            bordercolor=COLORS["border"],
            insertcolor=COLORS["accent"],
            insertwidth=2,
            font=(FONT_FAMILY, 10)
        )
        style.map("TEntry",
            fieldbackground=[("focus", ACTIVE_ENTRY_BG)],
            foreground=[("focus", ACTIVE_ENTRY_FG)],
            bordercolor=[("focus", COLORS["border_focus"])],
            insertcolor=[("focus", COLORS["accent"])],
        )

        # Combobox — explicit insert cursor settings so editable comboboxes show a caret
        style.configure(
            "TCombobox",
            fieldbackground=COLORS["bg_surface"],
            background=COLORS["bg_surface"],
            foreground=COLORS["text_primary"],
            bordercolor=COLORS["border"],
            arrowcolor=COLORS["text_secondary"],
            insertcolor=COLORS["accent"],
            insertwidth=2,
            font=(FONT_FAMILY, 10)
        )
        style.map("TCombobox",
            fieldbackground=[("focus", ACTIVE_ENTRY_BG), ("readonly", COLORS["bg_surface"]), ("!disabled", COLORS["bg_surface"])],
            foreground=[("focus", ACTIVE_ENTRY_FG), ("readonly", COLORS["text_primary"]), ("!disabled", COLORS["text_primary"])],
            selectbackground=[("readonly", COLORS["bg_surface"]), ("!disabled", COLORS["bg_surface"])],
            selectforeground=[("readonly", COLORS["text_primary"]), ("!disabled", COLORS["text_primary"])],
            bordercolor=[("focus", COLORS["border_focus"])]
        )

        # Scrollbar
        style.configure(
            "Vertical.TScrollbar",
            background=COLORS["bg_header"],
            troughcolor=COLORS["bg_deep"],
            bordercolor=COLORS["border"],
            arrowcolor=COLORS["text_secondary"],
            darkcolor=COLORS["bg_deep"],
            lightcolor=COLORS["bg_deep"],
            width=12
        )
        style.map(
            "Vertical.TScrollbar",
            background=[("active", COLORS["border"]), ("pressed", COLORS["border"])]
        )

        # LabelFrame
        style.configure("TLabelframe",
            background=COLORS["bg_deep"],
            bordercolor=COLORS["border"]
        )
        style.configure("TLabelframe.Label",
            background=COLORS["bg_deep"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10, "bold")
        )

    def create_scrollable_frame(self, parent):
        """Create a scrollable frame within a parent widget"""
        # Create a canvas
        canvas = tk.Canvas(parent, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # Configure the scrollable frame
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # Create window inside canvas
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # Configure canvas to expand horizontally
        def configure_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", configure_canvas)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        scrollable_frame.bind("<Enter>", bind_mousewheel)
        scrollable_frame.bind("<Leave>", unbind_mousewheel)

        return scrollable_frame, canvas

    def _add_tab_banner(self, parent, title, subtitle):
        """Start-tab-style banner (22pt title + 11pt subtitle on bg_deep).
        Packs into `parent`; returns the banner frame in case the caller wants to tweak it."""
        banner = tk.Frame(parent, bg=COLORS["bg_deep"])
        banner.pack(fill=tk.X, padx=36, pady=(28, 20))
        tk.Label(banner, text=title,
                 font=(FONT_FAMILY, 22, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_deep"]).pack(anchor=tk.W)
        if subtitle:
            tk.Label(banner, text=subtitle,
                     font=(FONT_FAMILY, 11),
                     fg=COLORS["text_secondary"], bg=COLORS["bg_deep"],
                     wraplength=1050, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))
        return banner

    def _add_youtube_help_button(self, parent, tab_key="start", prominent=False):
        """Add a 'Get help on YouTube' button at the bottom of a tab's outer frame.

        `tab_key` selects the URL from help.json's `youtube_urls` dict.
        `prominent=True` uses a larger button with a hint about per-tab help (for the Start tab).
        """
        fallback = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        try:
            with open(HELP_FILE, "r", encoding="utf-8") as f:
                urls = json.load(f).get("youtube_urls", {})
            url = urls.get(tab_key, fallback)
        except Exception:
            url = fallback
        btn_frame = tk.Frame(parent, bg=COLORS["bg_deep"])
        btn_frame.pack(fill=tk.X, padx=36, pady=(0 if prominent else 8, 8))
        if prominent:
            tk.Label(btn_frame, text="Need help? Every tab has a YouTube guide at the bottom.",
                     font=(FONT_FAMILY, 11), fg=COLORS["text_secondary"],
                     bg=COLORS["bg_deep"]).pack(anchor=tk.W, pady=(0, 8))
        row = tk.Frame(btn_frame, bg=COLORS["bg_deep"])
        row.pack(anchor=tk.W)
        btn = tk.Button(
            row, text="\u25b6  Get help on YouTube",
            font=(FONT_FAMILY, 12 if prominent else 10, "bold"),
            fg="#FFFFFF", bg="#CC0000", activeforeground="#FFFFFF", activebackground="#990000",
            relief="flat", bd=0, padx=20 if prominent else 16,
            pady=10 if prominent else 6, cursor="hand2",
            command=lambda: __import__("webbrowser").open(url),
        )
        btn.pack(side=tk.LEFT)
        if prominent:
            coffee = tk.Button(
                row, text="\u2615  Buy me a coffee",
                font=(FONT_FAMILY, 12, "bold"),
                fg="#000000", bg="#FFDD00", activeforeground="#000000", activebackground="#E5C700",
                relief="flat", bd=0, padx=20, pady=10, cursor="hand2",
                command=lambda: __import__("webbrowser").open(
                    "https://buymeacoffee.com/lorasandlenses"),
            )
            coffee.pack(side=tk.LEFT, padx=(12, 0))

    def _start_section_card(self, parent, title, description=None, accent_border=False):
        """Start-tab-style surface card with an optional description line.
        Returns the inner content frame (caller packs/grids its widgets into it).
        The card is packed into `parent` with horizontal padding matching the banner."""
        outer = tk.Frame(parent, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.X, padx=36, pady=(0, 16))

        card = tk.Frame(outer, bg=COLORS["bg_surface"],
                        highlightbackground=COLORS["accent"] if accent_border else COLORS["border"],
                        highlightthickness=1, bd=0)
        card.pack(fill=tk.X)

        if title:
            tk.Label(card, text=title,
                     font=(FONT_FAMILY, 12, "bold"),
                     fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(
                anchor=tk.W, padx=20, pady=(16, 2 if description else 10)
            )
        if description:
            tk.Label(card, text=description,
                     font=(FONT_FAMILY, 9),
                     fg=COLORS["text_muted"], bg=COLORS["bg_surface"],
                     wraplength=760, justify=tk.LEFT).pack(
                anchor=tk.W, padx=20, pady=(0, 10)
            )

        content = tk.Frame(card, bg=COLORS["bg_surface"])
        content.pack(fill=tk.X, padx=20, pady=(0, 16))
        return content

    def _add_field_to_section(self, parent, key, label_text, input_type, row):
        """Helper method to add a field to a section (collapsible or regular frame)"""
        # Create label
        label = tk.Label(
            parent,
            text=f"{label_text}:",
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_surface"]
        )
        label.grid(row=row, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        self.labels[key] = label

        # Create entry/combobox based on type
        if input_type == "dropdown":
            if key == "MODEL_TYPE":
                arch = self.settings["ARCHITECTURE"]
                arch_config = ARCHITECTURES.get(arch, ARCHITECTURES["Flux 2 Klein Base 9B"])
                model_types = arch_config.get("model_types", ["t2v-14B", "i2v-14B"])
                var = tk.StringVar(value=self.settings[key])
                self.entries[key] = ttk.Combobox(parent, textvariable=var, values=model_types, state="readonly", width=38)
                if self.settings[key] in model_types:
                    self.entries[key].current(model_types.index(self.settings[key]))
                else:
                    self.entries[key].current(0)
            elif key == "OPTIMIZER_TYPE":
                var = tk.StringVar(value=self.settings[key])
                self.entries[key] = ttk.Combobox(parent, textvariable=var, values=self.optimizer_types, state="readonly", width=38)
                self.entries[key].current(self.optimizer_types.index(self.settings[key]))
            elif key == "LR_SCHEDULER":
                lr_scheduler_options = ["constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear", "polynomial"]
                self.lr_scheduler_var = tk.StringVar(value=self.settings["LR_SCHEDULER"])
                self.entries[key] = ttk.Combobox(parent, textvariable=self.lr_scheduler_var, values=lr_scheduler_options, state="readonly", width=38)
        else:
            # Check if this entry should be bound to a shared pref var
            pref_key = SETTING_TO_PREF.get(key)
            if pref_key and pref_key in self.prefs_vars:
                self.entries[key] = ttk.Entry(parent, width=40, textvariable=self.prefs_vars[pref_key])
            else:
                self.entries[key] = ttk.Entry(parent, width=40)
                self.entries[key].insert(0, str(self.settings.get(key, "")))

        self.entries[key].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=4)

        # Create browse button if needed
        browse_btn = None
        if input_type in ["file", "directory"]:
            browse_btn = ttk.Button(parent, text="Browse", command=lambda k=key, t=input_type: self.browse_file(k, t))
            browse_btn.grid(row=row, column=2, sticky=tk.W, padx=(5, 12), pady=4)

        # Store row info for show/hide functionality
        self.rows[key] = {"row": row, "label": label, "entry": self.entries[key], "browse": browse_btn, "parent": parent}

    def create_start_tab(self):
        """Welcome screen + Training image folder picker — the single source of truth shared with
        the Image Prep / Captions tabs and the Fizgig_train.toml auto-saver."""
        scrollable_frame, _ = self.create_scrollable_frame(self.start_tab)

        container = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        container.pack(fill=tk.BOTH, expand=True, padx=36, pady=(28, 0))

        # Title + subtitle
        tk.Label(container, text="Welcome to Fizgig",
                 font=(FONT_FAMILY, 22, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_deep"]).pack(anchor=tk.W)
        tk.Label(container,
                 text="A focused, local trainer and workbench for Flux 2 Klein 9B LoRAs — "
                      "train, profile, repair, explore, and extract, all in one place.",
                 font=(FONT_FAMILY, 11),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_deep"],
                 wraplength=800, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 24))

        # Workflow card
        workflow_card = tk.Frame(container, bg=COLORS["bg_surface"],
                                 highlightbackground=COLORS["border"],
                                 highlightthickness=1, bd=0)
        workflow_card.pack(fill=tk.X, pady=(0, 20))

        # Use grid on workflow_card: col 0 = title+steps, col 1 = logo (no padding)
        workflow_card.columnconfigure(0, weight=1)

        tk.Label(workflow_card, text="Training Workflow",
                 font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).grid(
            row=0, column=0, sticky=tk.W, padx=20, pady=(16, 10))

        steps_frame = tk.Frame(workflow_card, bg=COLORS["bg_surface"])
        steps_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=20, pady=(0, 16))

        # Logo — spans both rows, fills full card height, zero padding
        try:
            from PIL import Image as _PILImage, ImageTk as _PILImageTk
            _logo_path = os.path.join(os.path.dirname(__file__), "logo.jpg")
            if os.path.exists(_logo_path):
                _logo_pil_src = _PILImage.open(_logo_path)
                _logo_label = tk.Label(workflow_card, bg=COLORS["bg_surface"])
                _logo_label.grid(row=0, column=1, rowspan=2, sticky=tk.NS, padx=0, pady=0)

                def _fit_logo(_event=None, _src=_logo_pil_src, _lbl=_logo_label):
                    h = workflow_card.winfo_height()
                    if h < 20:
                        h = 260
                    w = int(_src.width * h / _src.height)
                    resized = _src.resize((w, h), _PILImage.LANCZOS)
                    self._start_logo_tk = _PILImageTk.PhotoImage(resized)
                    _lbl.configure(image=self._start_logo_tk)

                self.master.after(100, _fit_logo)
        except Exception:
            pass  # PIL not available or logo missing — skip silently

        steps = [
            ("1", "Start",      "Choose your training image folder below.",                     False),
            ("2", "Image Prep", "Resize, convert to PNG, or face-crop.",                        True),   # optional
            ("3", "Captions",   "Write trigger-word captions or generate with Florence AI.",    False),
            ("4", "Samples",    "Configure in-training preview prompts.",                       False),
            ("5", "Training",   "Pick a preset, tune settings, click Start Training.",          False),
        ]

        for num, tab_name, desc, is_optional in steps:
            row = tk.Frame(steps_frame, bg=COLORS["bg_surface"])
            row.pack(fill=tk.X, pady=4)

            # Step number badge
            tk.Label(row, text=num,
                     font=(FONT_FAMILY, 11, "bold"),
                     fg=COLORS["accent"], bg=COLORS["bg_surface"],
                     width=2).pack(side=tk.LEFT, padx=(0, 12))

            # Tab name
            tk.Label(row, text=tab_name,
                     font=(FONT_FAMILY, 11, "bold"),
                     fg=COLORS["text_primary"], bg=COLORS["bg_surface"],
                     width=12, anchor="w").pack(side=tk.LEFT)

            # Optional badge
            if is_optional:
                tk.Label(row, text="OPTIONAL",
                         font=(FONT_FAMILY, 8, "bold"),
                         fg=COLORS["warning"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 10))

            # Description
            tk.Label(row, text=desc,
                     font=(FONT_FAMILY, 10),
                     fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
                     anchor="w", justify=tk.LEFT).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Folder picker card
        picker_card = tk.Frame(container, bg=COLORS["bg_surface"],
                               highlightbackground=COLORS["accent"],
                               highlightthickness=1, bd=0)
        picker_card.pack(fill=tk.X, pady=(0, 4))

        tk.Label(picker_card, text="Training image folder",
                 font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(anchor=tk.W, padx=20, pady=(16, 2))
        tk.Label(picker_card,
                 text="This is the single place you set your dataset folder. Image Prep, Captions, "
                      "and Training all read from it automatically.",
                 font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 12))

        row = tk.Frame(picker_card, bg=COLORS["bg_surface"])
        row.pack(fill=tk.X, padx=20, pady=(0, 16))
        ttk.Entry(row, textvariable=self.image_folder_var, width=70, font=(FONT_FAMILY, 10)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=4
        )
        ttk.Button(row, text="Browse…", command=self._browse_image_folder).pack(side=tk.LEFT)

        # Setup prompt — shown when model paths are not configured yet
        self._setup_prompt_frame = tk.Frame(container, bg=COLORS["warning"],
                                             highlightbackground=COLORS["warning"],
                                             highlightthickness=1, bd=0)
        setup_inner = tk.Frame(self._setup_prompt_frame, bg="#2A2200")
        setup_inner.pack(fill=tk.X, padx=1, pady=1)
        tk.Label(setup_inner, text="\u26a0  Model files not configured",
                 font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["warning"], bg="#2A2200").pack(anchor=tk.W, padx=20, pady=(12, 4))
        tk.Label(setup_inner,
                 text="Head to the Preferences tab to set your model paths before training or using the tools. "
                      "Each model row has a Download link that opens the correct HuggingFace page.",
                 font=(FONT_FAMILY, 10),
                 fg=COLORS["text_primary"], bg="#2A2200",
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 4))
        ttk.Button(setup_inner, text="Open Preferences",
                   command=lambda: self.notebook.select(self.prefs_tab)).pack(
            anchor=tk.W, padx=20, pady=(4, 12))

        def _check_model_paths(*_args):
            model_keys = ["base_dit", "distilled_dit", "vae", "text_encoder"]
            any_empty = any(not self.prefs_vars[k].get().strip() for k in model_keys)
            if any_empty:
                self._setup_prompt_frame.pack(fill=tk.X, pady=(20, 0),
                                               before=tools_card)
            else:
                self._setup_prompt_frame.pack_forget()

        # Re-check whenever a model path changes
        for _mk in ("base_dit", "distilled_dit", "vae", "text_encoder"):
            self.prefs_vars[_mk].trace_add("write", _check_model_paths)

        # Initial check (deferred so tools_card exists)
        self.master.after(100, _check_model_paths)

        # Tools card — highlights the post-training workbench tabs
        tools_card = tk.Frame(container, bg=COLORS["bg_surface"],
                              highlightbackground=COLORS["border"],
                              highlightthickness=1, bd=0)
        tools_card.pack(fill=tk.X, pady=(20, 0))

        tk.Label(tools_card, text="Post-Training Tools",
                 font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(anchor=tk.W, padx=20, pady=(16, 2))
        tk.Label(tools_card,
                 text="Fizgig is more than a trainer — these tabs let you understand and tune any Klein LoRA "
                      "you've made (or downloaded).",
                 font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 12))

        tools = [
            ("Profiler",
             "Analyze a LoRA's per-block activation profile and produce an HTML report."),
            ("Repair Studio",
             "Live per-block sliders with side-by-side preview. Blend in a donor LoRA and "
             "bake the result to a new .safetensors."),
            ("LoRA the Explorer",
             "Evolutionary discovery — the computer proposes random mutations, you pick favourites, "
             "and the LoRA evolves. Seamlessly connected to Repair Studio."),
            ("Extract",
             "Distill a LoRA to a lower rank with optional block- and timestep-targeted presets. "
             "Supports LyCORIS (LoKR / LoHa) sources."),
        ]

        for name, desc in tools:
            row = tk.Frame(tools_card, bg=COLORS["bg_surface"])
            row.pack(fill=tk.X, padx=20, pady=4)

            tk.Label(row, text=name,
                     font=(FONT_FAMILY, 11, "bold"),
                     fg=COLORS["accent"], bg=COLORS["bg_surface"],
                     width=16, anchor="w").pack(side=tk.LEFT, padx=(0, 12))
            tk.Label(row, text=desc,
                     font=(FONT_FAMILY, 10),
                     fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
                     anchor="w", justify=tk.LEFT, wraplength=620).pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Frame(tools_card, bg=COLORS["bg_surface"], height=12).pack()

        self._add_youtube_help_button(scrollable_frame, "start", prominent=True)

    def _browse_image_folder(self):
        """Folder picker for the Start tab (unified image folder)."""
        folder = filedialog.askdirectory(initialdir=self.image_folder_var.get() or os.getcwd())
        if folder:
            self.image_folder_var.set(folder)

    def create_training_settings(self):
        """Create the Training tab (Start-tab styled)."""
        scrollable_frame, self.training_canvas = self.create_scrollable_frame(self.training_tab)

        # Outer bg_deep container — all sections pack into this so the banner
        # and collapsibles share the same horizontal alignment.
        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        # Store collapsible sections for later access
        self.collapsible_sections = {}

        self._add_tab_banner(
            outer,
            "Training",
            "Pick a preset or dial in a custom run. Sections below collapse to reduce clutter — "
            "click any header to toggle. Dataset-config knobs live in Other Options.",
        )

        # Klein Base 9B is the only supported architecture; var kept for downstream code lookups.
        self.architecture_var = tk.StringVar(value="Flux 2 Klein Base 9B")

        # === Presets card ===
        preset_card = self._start_section_card(
            outer, "Presets",
            "Save the current settings under a name, load a saved preset, or restore the exact "
            "configuration from your last training run.",
        )
        # Row 1: Save + Load Preset
        preset_row1 = tk.Frame(preset_card, bg=COLORS["bg_surface"])
        preset_row1.pack(anchor=tk.W, pady=(0, 8))
        save_preset_btn = ttk.Button(preset_row1, text="Save Preset", command=self.save_custom_preset)
        save_preset_btn.pack(side=tk.LEFT, padx=(0, 12))
        ToolTip(save_preset_btn, "Save current training parameters as a named preset")
        tk.Label(
            preset_row1, text="Load Preset:",
            font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
        ).pack(side=tk.LEFT, padx=(0, 8))
        self.custom_preset_var = tk.StringVar()
        self.custom_preset_combo = ttk.Combobox(preset_row1, textvariable=self.custom_preset_var, state="readonly", width=28)
        self.custom_preset_combo.pack(side=tk.LEFT)
        self.custom_preset_combo.bind("<<ComboboxSelected>>", self.load_custom_preset)
        ToolTip(self.custom_preset_combo, "Your saved training presets")

        # Row 2: Load Settings From Last Train
        load_last_btn = ttk.Button(preset_card, text="Load Settings From Last Train",
                                   command=self._load_last_train_settings, width=32)
        load_last_btn.pack(anchor=tk.W)
        ToolTip(load_last_btn, "Restore the exact settings used in your most recent training launch")

        # === Output Section (Expanded by default) ===
        output_section = CollapsibleFrame(outer,"Output", default_expanded=True)
        output_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["output"] = output_section

        output_content = output_section.get_content_frame()
        output_content.columnconfigure(1, weight=1)

        self._add_field_to_section(output_content, "LORA_OUTPUT_DIR", "Output Directory", "directory", 0)
        self._add_field_to_section(output_content, "LORA_NAME", "LoRA Name", "text", 1)

        # Save LoRA output directory when it changes
        self.entries["LORA_OUTPUT_DIR"].bind("<FocusOut>", lambda e: self._save_last_used_paths())
        self.entries["LORA_OUTPUT_DIR"].bind("<Return>", lambda e: self._save_last_used_paths())

        # === Training Parameters Section (Expanded by default) ===
        training_section = CollapsibleFrame(outer,"Training Parameters", default_expanded=True)
        training_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["training"] = training_section

        training_content = training_section.get_content_frame()
        training_content.columnconfigure(1, weight=1)

        self._add_field_to_section(training_content, "MODEL_TYPE", "Model Type", "dropdown", 0)
        self._add_field_to_section(training_content, "LEARNING_RATE", "Learning Rate", "float", 1)

        # --- Adaptive LR (bi-directional plateau tracker) — placed under Learning Rate so both bracket the starting LR
        self.adaptive_lr_var = tk.BooleanVar(value=False)
        adaptive_cb = ttk.Checkbutton(
            training_content, text="Adaptive LR (auto-adjust based on loss, gradient clipping & weight-norm growth)",
            variable=self.adaptive_lr_var, command=self._on_adaptive_lr_toggle,
        )
        adaptive_cb.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(4, 0))

        adaptive_frame = ttk.Frame(training_content)
        adaptive_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=(20, 5), pady=(0, 2))
        ttk.Label(adaptive_frame, text="Min LR:").pack(side=tk.LEFT, padx=(0, 4))
        self.entries["ADAPTIVE_LR_MIN"] = ttk.Combobox(adaptive_frame, width=8, values=["1e-5", "5e-5", "1e-4"], state="readonly")
        self.entries["ADAPTIVE_LR_MIN"].set("1e-5")
        self.entries["ADAPTIVE_LR_MIN"].pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(adaptive_frame, text="Max LR:").pack(side=tk.LEFT, padx=(0, 4))
        self.entries["ADAPTIVE_LR_MAX"] = ttk.Combobox(adaptive_frame, width=8, values=["1e-4", "2e-4", "3e-4", "4e-4"], state="readonly")
        self.entries["ADAPTIVE_LR_MAX"].set("4e-4")
        self.entries["ADAPTIVE_LR_MAX"].pack(side=tk.LEFT, padx=(0, 12))
        self._adaptive_reset_btn = ttk.Button(adaptive_frame, text="Reset Defaults", command=self._reset_adaptive_lr_defaults)
        self._adaptive_reset_btn.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(training_content,
                  text="When on, starting LR = Learning Rate field. Probes UP on steady loss descent; reduces DOWN "
                       "on loss plateau, heavy gradient clipping, or runaway weight-norm growth (with a rollback to "
                       "the previous epoch's weights on stability events).",
                  foreground="#95A5A6", font=(FONT_FAMILY, 8, "italic"), justify=tk.LEFT, wraplength=720,
                 ).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=(20, 5), pady=(0, 6))
        self._on_adaptive_lr_toggle()  # sync initial enabled/disabled state

        self._add_field_to_section(training_content, "LORA_LR_RATIO", "LoRA LR Ratio", "int", 5)
        self._add_field_to_section(training_content, "NETWORK_DIM", "Network Dim (Rank)", "int", 6)
        self._add_field_to_section(training_content, "NETWORK_ALPHA", "Network Alpha", "float", 7)
        self._add_field_to_section(training_content, "MAX_TRAIN_EPOCHS", "Max Epochs", "int", 8)
        self._add_field_to_section(training_content, "SAVE_EVERY_N_EPOCHS", "Save Every N Epochs", "int", 9)
        self._add_field_to_section(training_content, "SEED", "Seed", "int", 10)

        # Model Area to Train dropdown (blocks + timestep auto-fill)
        ttk.Label(training_content, text="Model Area to Train:").grid(row=11, column=0, sticky=tk.W, padx=5, pady=2)
        self.training_preset_var = tk.StringVar(value="Full Model")
        training_preset_combo = ttk.Combobox(
            training_content, textvariable=self.training_preset_var,
            values=["Full Model", "Identity", "Style", "Style+Composition", "Details", "Custom"],
            state="readonly", width=20
        )
        training_preset_combo.grid(row=11, column=1, sticky=tk.W, padx=5, pady=2)
        training_preset_combo.bind("<<ComboboxSelected>>", self._on_training_preset_changed)
        ttk.Label(training_content,
                  text="Identity = single 1-16  |  Style = style+comp blocks @ late ts (0-400)  |  Style+Composition = double 0-7 + single 0-1  |  Details = single 12-23",
                  foreground="#95A5A6", font=(FONT_FAMILY, 8, "italic")).grid(row=12, column=0, columnspan=2, sticky=tk.W, padx=5)

        # Custom block picker panel (hidden unless preset == Custom)
        self._training_custom_frame = ttk.Frame(training_content)
        self._training_custom_frame.grid(row=13, column=0, columnspan=2, sticky=tk.W, padx=15, pady=(4, 4))
        self.training_block_vars = {}  # block_name -> BooleanVar

        tc_header = ttk.Frame(self._training_custom_frame)
        tc_header.pack(anchor=tk.W, fill=tk.X, pady=(0, 4))
        ttk.Label(tc_header, text="Select blocks to train:",
                  font=(FONT_FAMILY, 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(tc_header, text="All", width=5,
                   command=lambda: self._set_all_training_blocks(True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(tc_header, text="None", width=5,
                   command=lambda: self._set_all_training_blocks(False)).pack(side=tk.LEFT, padx=2)
        ttk.Button(tc_header, text="Identity", width=8,
                   command=lambda: self._set_category_training_blocks("identity")).pack(side=tk.LEFT, padx=2)
        ttk.Button(tc_header, text="Style+Comp", width=10,
                   command=lambda: self._set_category_training_blocks("style_composition")).pack(side=tk.LEFT, padx=2)
        ttk.Button(tc_header, text="Details", width=8,
                   command=lambda: self._set_category_training_blocks("details")).pack(side=tk.LEFT, padx=2)

        tc_double = ttk.Frame(self._training_custom_frame)
        tc_double.pack(anchor=tk.W, pady=2)
        ttk.Label(tc_double, text="Double:", width=8, foreground="#5B9BD5").pack(side=tk.LEFT)
        for i in range(8):
            key = f"double_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.training_block_vars[key] = var
            ttk.Checkbutton(tc_double, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        tc_single1 = ttk.Frame(self._training_custom_frame)
        tc_single1.pack(anchor=tk.W, pady=2)
        ttk.Label(tc_single1, text="Single:", width=8, foreground="#70AD47").pack(side=tk.LEFT)
        for i in range(12):
            key = f"single_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.training_block_vars[key] = var
            ttk.Checkbutton(tc_single1, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        tc_single2 = ttk.Frame(self._training_custom_frame)
        tc_single2.pack(anchor=tk.W, pady=2)
        ttk.Label(tc_single2, text="Single:", width=8, foreground="#ED7D31").pack(side=tk.LEFT)
        for i in range(12, 24):
            key = f"single_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.training_block_vars[key] = var
            ttk.Checkbutton(tc_single2, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        ttk.Label(self._training_custom_frame,
                  text="double + single 0-1 = style+composition  |  single 1-16 = identity (overlaps at 1 and 12-16)  |  single 12-23 = details  |  edit MIN/MAX_TIMESTEP on Advanced tab",
                  foreground="#95A5A6", font=(FONT_FAMILY, 8, "italic")).pack(anchor=tk.W, pady=(4, 0))

        self._training_custom_frame.grid_remove()  # hidden until preset == Custom

        # Context LoRA (optional) — train new LoRA with an existing one frozen + active on the base
        ttk.Label(training_content, text="Context LoRA:").grid(row=14, column=0, sticky=tk.W, padx=5, pady=(8, 2))
        ctx_frame = ttk.Frame(training_content)
        ctx_frame.grid(row=14, column=1, sticky=tk.W, padx=5, pady=(8, 2))
        self.entries["CONTEXT_LORA_PATH"] = ttk.Entry(ctx_frame, width=42)
        self.entries["CONTEXT_LORA_PATH"].pack(side=tk.LEFT)
        ttk.Button(ctx_frame, text="Browse",
                   command=lambda: self._browse_context_lora()).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(ctx_frame, text="Strength:").pack(side=tk.LEFT, padx=(4, 4))
        self.entries["CONTEXT_LORA_STRENGTH"] = ttk.Entry(ctx_frame, width=6)
        self.entries["CONTEXT_LORA_STRENGTH"].insert(0, "1.0")
        self.entries["CONTEXT_LORA_STRENGTH"].pack(side=tk.LEFT)
        ttk.Label(training_content,
                  text="Train this LoRA with an existing LoRA already active on the base model. "
                       "Pair with same context+strength at inference.",
                  foreground="#95A5A6", font=(FONT_FAMILY, 8, "italic")).grid(
            row=15, column=0, columnspan=2, sticky=tk.W, padx=5)
        ttk.Label(training_content,
                  text="⚠ Context LoRAs usually look better in ComfyUI than in training samples — "
                       "don't worry if previews look rough, test the output LoRA in ComfyUI.",
                  foreground="#E67E22", font=(FONT_FAMILY, 8, "italic")).grid(
            row=16, column=0, columnspan=2, sticky=tk.W, padx=5)

        # === Optimizer Section (Collapsed by default) ===
        optimizer_section = CollapsibleFrame(outer,"Optimizer", default_expanded=False)
        optimizer_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["optimizer"] = optimizer_section

        optimizer_content = optimizer_section.get_content_frame()
        optimizer_content.columnconfigure(1, weight=1)

        self._add_field_to_section(optimizer_content, "OPTIMIZER_TYPE", "Optimizer Type", "dropdown", 0)
        self._add_field_to_section(optimizer_content, "OPTIMIZER_ARGS", "Optimizer Args", "text", 1)
        self._add_field_to_section(optimizer_content, "GRADIENT_ACCUMULATION", "Gradient Accumulation", "int", 2)
        self._add_field_to_section(optimizer_content, "MAX_GRAD_NORM", "Max Grad Norm", "float", 3)
        self._add_field_to_section(optimizer_content, "NETWORK_DROPOUT", "Network Dropout", "float", 4)

        # === Scheduler Section (Collapsed by default) ===
        scheduler_section = CollapsibleFrame(outer,"Other Options", default_expanded=False)
        scheduler_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["scheduler"] = scheduler_section

        scheduler_content = scheduler_section.get_content_frame()
        scheduler_content.columnconfigure(1, weight=1)

        # === Dataset subsection (migrated from the removed Dataset tab) ===
        tk.Label(
            scheduler_content,
            text="Dataset",
            font=(FONT_FAMILY, 10, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_surface"],
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=12, pady=(8, 4))

        ttk.Label(scheduler_content, text="Caption Extension:").grid(row=1, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        ttk.Entry(scheduler_content, textvariable=self.dataset_caption_ext_var, width=16).grid(row=1, column=1, sticky=tk.W, padx=5, pady=4)
        tk.Label(scheduler_content, text="(default .txt)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(row=1, column=2, sticky=tk.W, padx=5)

        ttk.Label(scheduler_content, text="Target Megapixels:").grid(row=2, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        mp_frame = tk.Frame(scheduler_content, bg=COLORS["bg_surface"])
        mp_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=4)
        ttk.Combobox(mp_frame, textvariable=self.dataset_megapixels_var,
                     values=["0.25", "0.5", "0.75", "1.0", "1.5", "2.0"], width=8).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(mp_frame, text="MP  (1.0 = 1024×1024 area)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)

        ttk.Label(scheduler_content, text="Batch Size:").grid(row=3, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        bs_frame = tk.Frame(scheduler_content, bg=COLORS["bg_surface"])
        bs_frame.grid(row=3, column=1, columnspan=2, sticky=tk.W, padx=5, pady=4)
        ttk.Entry(bs_frame, textvariable=self.dataset_batch_size_var, width=6).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(bs_frame, text="(recommended: 1 — higher values need more VRAM)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)

        ttk.Label(scheduler_content, text="Bucket Options:").grid(row=4, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        bucket_frame = tk.Frame(scheduler_content, bg=COLORS["bg_surface"])
        bucket_frame.grid(row=4, column=1, columnspan=2, sticky=tk.W, padx=5, pady=4)
        ttk.Checkbutton(bucket_frame, text="Enable Bucket", variable=self.dataset_enable_bucket_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(bucket_frame, text="No Upscale (keep small images at native size)",
                        variable=self.dataset_no_upscale_var).pack(side=tk.LEFT)

        ttk.Separator(scheduler_content, orient="horizontal").grid(row=5, column=0, columnspan=3, sticky="ew", padx=12, pady=(10, 6))

        # === LR Scheduler ===
        tk.Label(
            scheduler_content,
            text="LR Scheduler",
            font=(FONT_FAMILY, 10, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_surface"],
        ).grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=12, pady=(4, 4))

        self._add_field_to_section(scheduler_content, "LR_SCHEDULER", "LR Scheduler", "dropdown", 7)

        # Warmup/Decay steps in a sub-frame
        lr_steps_label = tk.Label(
            scheduler_content,
            text="Warmup / Decay:",
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_surface"]
        )
        lr_steps_label.grid(row=8, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        lr_steps_frame = tk.Frame(scheduler_content, bg=COLORS["bg_surface"])
        lr_steps_frame.grid(row=8, column=1, sticky=tk.W, padx=5, pady=4)

        tk.Label(lr_steps_frame, text="Warmup:", font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 4))
        self.entries["LR_WARMUP_STEPS"] = ttk.Entry(lr_steps_frame, width=10)
        self.entries["LR_WARMUP_STEPS"].pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(lr_steps_frame, text="Decay:", font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 4))
        self.entries["LR_DECAY_STEPS"] = ttk.Entry(lr_steps_frame, width=10)
        self.entries["LR_DECAY_STEPS"].pack(side=tk.LEFT)

        # Separator before the migrated Advanced fields
        ttk.Separator(scheduler_content, orient="horizontal").grid(row=9, column=0, columnspan=3, sticky="ew", padx=12, pady=(8, 4))
        # Inline Attention / Logging / Memory / Metadata fields (formerly the Advanced tab)
        self._populate_other_options(scheduler_content, start_row=10)

        # === Memory & FP8 Section (Collapsed by default) ===
        memory_section = CollapsibleFrame(outer,"Memory & FP8", default_expanded=True)
        memory_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["memory"] = memory_section

        memory_content = memory_section.get_content_frame()
        memory_content.columnconfigure(1, weight=1)

        # Blocks Swap dropdown — labeled VRAM presets first, then leftover numbers (Klein 9B max=16)
        ttk.Label(memory_content, text="Blocks Swap:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        blocks_swap_options = [
            "Auto (detect from GPU)",
            "0  (No swap — 32GB+ VRAM)",
            "4  (Light — 24GB)",
            "8  (Moderate — 16GB)",
            "12 (Aggressive — 12GB)",
            "16 (Very conservative — 8-10GB)",
            "1", "2", "3", "5", "6", "7", "9", "10", "11", "13", "14", "15",
        ]
        _bs_max_len = max(len(v) for v in blocks_swap_options)
        self.entries["BLOCKS_SWAP"] = ttk.Combobox(memory_content, values=blocks_swap_options, width=_bs_max_len + 2)
        self.entries["BLOCKS_SWAP"].grid(row=0, column=1, sticky=tk.W, padx=5, pady=4)
        try:
            _bs_val = self.settings.get("BLOCKS_SWAP", "auto")
            if str(_bs_val).lower() == "auto":
                self.entries["BLOCKS_SWAP"].set(blocks_swap_options[0])
            else:
                _bs_int = int(_bs_val)
                _label_map = {0: blocks_swap_options[1], 4: blocks_swap_options[2], 8: blocks_swap_options[3],
                              12: blocks_swap_options[4], 16: blocks_swap_options[5]}
                self.entries["BLOCKS_SWAP"].set(_label_map.get(_bs_int, str(_bs_int)))
        except (ValueError, TypeError):
            self.entries["BLOCKS_SWAP"].set(blocks_swap_options[0])

        self._add_field_to_section(memory_content, "RESUME_TRAINING", "Resume Training", "directory", 1)

        # FP8 Checkboxes
        fp8_label = tk.Label(
            memory_content,
            text="Weight Optimization:",
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_surface"]
        )
        fp8_label.grid(row=2, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        fp8_frame = tk.Frame(memory_content, bg=COLORS["bg_surface"])
        fp8_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=4)

        self.fp8_var = tk.BooleanVar(value=self.settings["FP8"])
        self.scaled_var = tk.BooleanVar(value=self.settings["SCALED"])

        self.fp8_check = ttk.Checkbutton(fp8_frame, text="FP8 Base", variable=self.fp8_var, command=self.toggle_scaled, style="Surface.TCheckbutton")
        self.fp8_check.pack(side=tk.LEFT, padx=(0, 16))

        self.scaled_check = ttk.Checkbutton(fp8_frame, text="FP8 Scaled", variable=self.scaled_var, state=tk.DISABLED if not self.fp8_var.get() else tk.NORMAL, style="Surface.TCheckbutton")
        self.scaled_check.pack(side=tk.LEFT)
        tk.Label(memory_content,
                 text="Converts a bf16 model to fp8 at load time. If your Base DiT is already fp8 "
                      "(e.g. flux-2-klein-base-9b-fp8), leave this unchecked — Fizgig detects "
                      "pre-quantised fp8 files automatically.",
                 font=(FONT_FAMILY, 8, "italic"),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"],
                 wraplength=600, justify=tk.LEFT).grid(
            row=3, column=1, sticky=tk.W, padx=5, pady=(0, 4))

        # FP8 Text Encoder
        self.fp8_text_encoder_label = tk.Label(
            memory_content,
            text="FP8 Text Encoder:",
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_surface"]
        )
        self.fp8_text_encoder_label.grid(row=4, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        self.fp8_text_encoder_var = tk.BooleanVar(value=self.settings["FP8_TEXT_ENCODER"])
        self.fp8_text_encoder_check = ttk.Checkbutton(memory_content, text="Enable FP8 T5/LLM", variable=self.fp8_text_encoder_var, style="Surface.TCheckbutton")
        self.fp8_text_encoder_check.grid(row=4, column=1, sticky=tk.W, padx=5, pady=4)

        # === Timestep & Noise Schedule Section (Collapsed by default) ===
        timestep_section = CollapsibleFrame(outer,"Timestep & Noise Schedule", default_expanded=False)
        timestep_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        self.collapsible_sections["timestep"] = timestep_section

        ts_content = timestep_section.get_content_frame()
        ts_content.columnconfigure(1, weight=1)

        ts_row = 0

        # Quick Preset Buttons
        preset_btn_frame = tk.Frame(ts_content, bg=COLORS["bg_surface"])
        preset_btn_frame.grid(row=ts_row, column=0, columnspan=2, sticky=tk.W, padx=12, pady=(8, 4))

        tk.Label(preset_btn_frame, text="Quick Presets:", font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 8))

        for preset_name, preset_fn in [
            ("Full Range", self._ts_preset_full_range),
            ("Structure Focus", self._ts_preset_structure),
            ("Detail Focus", self._ts_preset_detail),
            ("Balanced Sigmoid", self._ts_preset_sigmoid),
        ]:
            btn = ttk.Button(preset_btn_frame, text=preset_name, command=preset_fn)
            btn.pack(side=tk.LEFT, padx=2)

        ts_row += 1

        # Timestep Sampling (editable dropdown)
        ts_sampling_label = tk.Label(ts_content, text="Timestep Sampling:",
                                     font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        ts_sampling_label.grid(row=ts_row, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        self.ts_sampling_var = tk.StringVar(value=self.settings["TIMESTEP_SAMPLING"])
        ts_sampling_options = ["sigma", "uniform", "sigmoid", "shift", "flux_shift", "flux2_shift", "qwen_shift", "logsnr"]
        self.ts_sampling_combo = ttk.Combobox(ts_content, textvariable=self.ts_sampling_var,
                                               values=ts_sampling_options, state="readonly", width=20)
        self.ts_sampling_combo.grid(row=ts_row, column=1, sticky=tk.W, padx=5, pady=4)
        self.ts_sampling_combo.bind("<<ComboboxSelected>>", self._on_timestep_sampling_changed)
        self.entries["TIMESTEP_SAMPLING"] = self.ts_sampling_combo
        ts_row += 1

        # Discrete Flow Shift (not used by Klein 9B — uses flux2_shift automatic)
        self.ts_flow_shift_label = tk.Label(ts_content, text="Discrete Flow Shift (not Klein 9B):",
                                            font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self.ts_flow_shift_label.grid(row=ts_row, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        self.entries["DISCRETE_FLOW_SHIFT"] = ttk.Entry(ts_content, width=12)
        self.entries["DISCRETE_FLOW_SHIFT"].insert(0, self.settings["DISCRETE_FLOW_SHIFT"])
        self.entries["DISCRETE_FLOW_SHIFT"].grid(row=ts_row, column=1, sticky=tk.W, padx=5, pady=4)
        ts_row += 1

        # Sigmoid Scale
        self.ts_sigmoid_label = tk.Label(ts_content, text="Sigmoid Scale:",
                                         font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self.ts_sigmoid_label.grid(row=ts_row, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        self.entries["SIGMOID_SCALE"] = ttk.Entry(ts_content, width=12)
        self.entries["SIGMOID_SCALE"].insert(0, self.settings["SIGMOID_SCALE"])
        self.entries["SIGMOID_SCALE"].grid(row=ts_row, column=1, sticky=tk.W, padx=5, pady=4)
        ts_row += 1

        # Min / Max Timestep on one row
        ts_range_label = tk.Label(ts_content, text="Timestep Range:",
                                  font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        ts_range_label.grid(row=ts_row, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        ts_range_frame = tk.Frame(ts_content, bg=COLORS["bg_surface"])
        ts_range_frame.grid(row=ts_row, column=1, sticky=tk.W, padx=5, pady=4)

        tk.Label(ts_range_frame, text="Min:", font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 4))
        self.entries["MIN_TIMESTEP"] = ttk.Entry(ts_range_frame, width=8)
        self.entries["MIN_TIMESTEP"].insert(0, self.settings["MIN_TIMESTEP"])
        self.entries["MIN_TIMESTEP"].pack(side=tk.LEFT, padx=(0, 16))
        self.entries["MIN_TIMESTEP"].bind("<KeyRelease>", lambda e: self._update_noise_range_label())

        tk.Label(ts_range_frame, text="Max:", font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 4))
        self.entries["MAX_TIMESTEP"] = ttk.Entry(ts_range_frame, width=8)
        self.entries["MAX_TIMESTEP"].insert(0, self.settings["MAX_TIMESTEP"])
        self.entries["MAX_TIMESTEP"].pack(side=tk.LEFT)
        self.entries["MAX_TIMESTEP"].bind("<KeyRelease>", lambda e: self._update_noise_range_label())

        ts_row += 1

        # Noise range description label
        self.noise_range_label = tk.Label(ts_content, text="", font=(FONT_FAMILY, 9),
                                          fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
        self.noise_range_label.grid(row=ts_row, column=0, columnspan=2, sticky=tk.W, padx=(12, 8), pady=(0, 4))
        self._update_noise_range_label()
        ts_row += 1

        # Preserve Distribution Shape
        self.preserve_dist_var = tk.BooleanVar(value=self.settings["PRESERVE_DISTRIBUTION"])
        self.preserve_dist_check = ttk.Checkbutton(ts_content, text="Preserve Distribution Shape",
                                                    variable=self.preserve_dist_var, style="Surface.TCheckbutton")
        self.preserve_dist_check.grid(row=ts_row, column=0, columnspan=2, sticky=tk.W, padx=(12, 8), pady=4)
        ToolTip(self.preserve_dist_check, "Use rejection sampling to preserve the original\n"
                "distribution shape within the min/max range.\n"
                "Only effective when min/max timestep is set.")
        self.entries["PRESERVE_DISTRIBUTION"] = self.preserve_dist_var
        ts_row += 1

        # Separator
        ttk.Separator(ts_content, orient="horizontal").grid(row=ts_row, column=0, columnspan=2, sticky="ew", padx=12, pady=8)
        ts_row += 1

        # Weighting Scheme (not used by Klein 9B)
        self.ts_weighting_label = tk.Label(ts_content, text="Weighting Scheme (not Klein 9B):",
                                           font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self.ts_weighting_label.grid(row=ts_row, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        self.weighting_scheme_var = tk.StringVar(value=self.settings["WEIGHTING_SCHEME"])
        weighting_options = ["none", "logit_normal", "mode", "cosmap", "sigma_sqrt"]
        self.ts_weighting_combo = ttk.Combobox(ts_content, textvariable=self.weighting_scheme_var,
                                                values=weighting_options, state="readonly", width=20)
        self.ts_weighting_combo.grid(row=ts_row, column=1, sticky=tk.W, padx=5, pady=4)
        self.ts_weighting_combo.bind("<<ComboboxSelected>>", self._on_weighting_scheme_changed)
        self.entries["WEIGHTING_SCHEME"] = self.ts_weighting_combo
        ts_row += 1

        # Logit Mean / Logit Std on one row
        self.ts_logit_label = tk.Label(ts_content, text="Logit Normal:",
                                       font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self.ts_logit_label.grid(row=ts_row, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        logit_frame = tk.Frame(ts_content, bg=COLORS["bg_surface"])
        logit_frame.grid(row=ts_row, column=1, sticky=tk.W, padx=5, pady=4)

        tk.Label(logit_frame, text="Mean:", font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 4))
        self.entries["LOGIT_MEAN"] = ttk.Entry(logit_frame, width=8)
        self.entries["LOGIT_MEAN"].insert(0, self.settings["LOGIT_MEAN"])
        self.entries["LOGIT_MEAN"].pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(logit_frame, text="Std:", font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 4))
        self.entries["LOGIT_STD"] = ttk.Entry(logit_frame, width=8)
        self.entries["LOGIT_STD"].insert(0, self.settings["LOGIT_STD"])
        self.entries["LOGIT_STD"].pack(side=tk.LEFT)
        ts_row += 1

        # Mode Scale
        self.ts_mode_label = tk.Label(ts_content, text="Mode Scale:",
                                      font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self.ts_mode_label.grid(row=ts_row, column=0, sticky=tk.W, padx=(12, 8), pady=4)

        self.entries["MODE_SCALE"] = ttk.Entry(ts_content, width=12)
        self.entries["MODE_SCALE"].insert(0, self.settings["MODE_SCALE"])
        self.entries["MODE_SCALE"].grid(row=ts_row, column=1, sticky=tk.W, padx=5, pady=4)
        ts_row += 1

        # Initial state for conditional fields
        self._on_timestep_sampling_changed()
        self._on_weighting_scheme_changed()

        # === Reorder collapsible sections: Training → Memory & FP8 → Timestep → Optimizer → Other Options ===
        # Sections were created in declaration order; re-pack in the desired display order.
        # Training Parameters and Memory & FP8 are open by default since most users need both.
        for _sec in (training_section, memory_section, timestep_section, optimizer_section, scheduler_section):
            try:
                _sec.pack_forget()
                _sec.pack(fill=tk.X, padx=36, pady=(0, 16))
            except Exception:
                pass

        # === Run card — Enable Cache + Start/Pause/Resume/Stop buttons ===
        run_card = self._start_section_card(outer, "Run", None)

        self.enable_cache_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(run_card, text="Enable Cache Preparation", variable=self.enable_cache_var).pack(anchor=tk.W, pady=(0, 12))

        button_frame = tk.Frame(run_card, bg=COLORS["bg_surface"])
        button_frame.pack(anchor=tk.W)

        self._start_training_btn = ttk.Button(button_frame, text="Start Training", command=self.start_training, style="Primary.TButton")
        self._start_training_btn.pack(side=tk.LEFT, padx=(0, 12))

        self._pause_training_btn = ttk.Button(button_frame, text="Pause Training", command=self._pause_training)
        self._pause_training_btn.pack(side=tk.LEFT, padx=(0, 12))
        self._pause_training_btn.pack_forget()  # hidden until training is running

        self._resume_training_btn = ttk.Button(button_frame, text="Resume Training", command=self._resume_training, style="Primary.TButton")
        self._resume_training_btn.pack(side=tk.LEFT, padx=(0, 12))
        self._resume_training_btn.pack_forget()  # hidden until paused state exists

        stop_btn = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, style="Danger.TButton")
        stop_btn.pack(side=tk.LEFT)

        # === Console Output card ===
        console_card = self._start_section_card(outer, "Console Output", None)
        self.console_frame = tk.Frame(console_card, bg=COLORS["bg_surface"])
        self.console_frame.pack(fill=tk.BOTH, expand=True)

        self.console_output = tk.Text(
            self.console_frame,
            height=12,
            width=80,
            bg=COLORS["bg_deep"],
            fg=COLORS["text_primary"],
            font=(FONT_MONO, 9),
            wrap="word",
            state="disabled",
            selectbackground=COLORS["accent"],
            selectforeground="white",
            borderwidth=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["border_focus"],
            padx=12,
            pady=8
        )
        self.console_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.console_scrollbar = ttk.Scrollbar(
            self.console_frame,
            orient="vertical",
            command=self.console_output.yview,
            style="Vertical.TScrollbar"
        )
        self.console_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.console_output.configure(yscrollcommand=self.console_scrollbar.set)

        self.console_output.bind("<MouseWheel>", self.on_mousewheel)
        self.console_output.bind("<Button-4>", self.on_mousewheel)
        self.console_output.bind("<Button-5>", self.on_mousewheel)
        self.console_output.bind("<Button-3>", self.show_context_menu)

        # Initial UI update based on architecture
        self.update_ui_for_architecture()
        self.refresh_preset_combobox()

        self._add_youtube_help_button(outer, "training")

    def on_architecture_changed(self, event=None):
        """Handle architecture change"""
        self.update_ui_for_architecture()
        self.refresh_preset_combobox()
        self.load_default_preset(show_message=False)  # Auto-load defaults for new architecture
        # Update samples tab UI for new architecture
        if hasattr(self, 'sample_settings_frame'):
            self.update_samples_ui_for_architecture()

    def get_preset_dir_for_architecture(self, arch):
        """Get the preset directory for an architecture, creating if needed"""
        preset_dir = os.path.join(PRESETS_DIR, arch)
        os.makedirs(preset_dir, exist_ok=True)
        return preset_dir

    def get_saved_presets(self, arch):
        """Get list of saved preset names for an architecture"""
        preset_dir = self.get_preset_dir_for_architecture(arch)
        presets = []
        for f in os.listdir(preset_dir):
            if f.endswith('.json'):
                presets.append(f[:-5])  # Remove .json extension
        return sorted(presets)

    def refresh_preset_combobox(self):
        """Refresh the preset combobox: built-in presets first, then user-saved presets."""
        arch = self.architecture_var.get()
        user_presets = self.get_saved_presets(arch)
        builtins = list(BUILT_IN_PRESETS.keys())
        # Built-ins first; if a user saves a preset with same name as a built-in, it appears once (under user)
        combined = builtins + [p for p in user_presets if p not in BUILT_IN_PRESETS]
        self.custom_preset_combo['values'] = combined
        # Dynamic width: fit longest entry so names like "✨ Multi-Character (rank 16, noisy dataset)" don't truncate
        max_len = max((len(v) for v in combined), default=20)
        self.custom_preset_combo.config(width=max(20, min(max_len + 2, 60)))
        self.custom_preset_var.set('')  # Clear selection

    def load_default_preset(self, show_message=True):
        """Load recommended preset values for the current architecture"""
        arch = self.architecture_var.get()
        if arch not in PRESETS:
            if show_message:
                messagebox.showinfo("Info", f"No preset available for {arch}")
            return

        preset = PRESETS[arch]
        self._apply_preset_values(preset)
        if show_message:
            messagebox.showinfo("Preset Loaded", f"Loaded recommended preset for {arch}")

    def _apply_preset_values(self, preset):
        """Apply preset values to the UI (shared by load_default_preset and load_custom_preset)"""
        for key, value in preset.items():
            if key in self.entries:
                entry = self.entries[key]
                if isinstance(entry, ttk.Combobox):
                    entry.set(str(value))
                elif isinstance(entry, tk.BooleanVar):
                    # Some boolean settings (e.g. IMG_IN_TXT_IN_OFFLOADING, PRESERVE_DISTRIBUTION)
                    # are stored in self.entries as BooleanVars — they don't support .delete/.insert.
                    entry.set(bool(value))
                else:
                    try:
                        entry.delete(0, tk.END)
                        entry.insert(0, str(value))
                    except (AttributeError, tk.TclError):
                        # Unknown widget type — skip rather than crash
                        pass

        # Update timestep settings from preset
        if "TIMESTEP_SAMPLING" in preset:
            self.ts_sampling_var.set(preset["TIMESTEP_SAMPLING"])
        if "WEIGHTING_SCHEME" in preset:
            self.weighting_scheme_var.set(preset["WEIGHTING_SCHEME"])
        if "PRESERVE_DISTRIBUTION" in preset:
            self.preserve_dist_var.set(preset["PRESERVE_DISTRIBUTION"])
        # Refresh conditional field states after preset load
        if hasattr(self, 'ts_sampling_var'):
            self._on_timestep_sampling_changed()
            self._on_weighting_scheme_changed()
            self._update_noise_range_label()

        # Update FP8/SCALED checkboxes from preset
        if "FP8" in preset:
            self.fp8_var.set(preset["FP8"])
        if "SCALED" in preset:
            self.scaled_var.set(preset["SCALED"])

        # Adaptive LR checkbox + sync enabled state of Min/Max LR dropdowns
        if "ADAPTIVE_LR" in preset and hasattr(self, 'adaptive_lr_var'):
            self.adaptive_lr_var.set(bool(preset["ADAPTIVE_LR"]))
            if hasattr(self, '_on_adaptive_lr_toggle'):
                self._on_adaptive_lr_toggle()

        # Model Area to Train (training preset dropdown)
        if "TARGET_LAYERS" in preset and hasattr(self, 'training_preset_var'):
            legacy_map = {
                "All Layers": "Full Model",
                "Identity Blocks": "Identity",
                "Style+Composition Blocks": "Style+Composition",
                "Details Blocks": "Details",
            }
            raw = preset["TARGET_LAYERS"]
            mapped = legacy_map.get(raw, raw)
            valid = ("Full Model", "Identity", "Style", "Style+Composition", "Details", "Custom")
            self.training_preset_var.set(mapped if mapped in valid else "Full Model")
            if hasattr(self, '_on_training_preset_changed'):
                self._on_training_preset_changed()
        if "FP8_TEXT_ENCODER" in preset:
            self.fp8_text_encoder_var.set(preset["FP8_TEXT_ENCODER"])
        if "ENABLE_BUCKET" in preset:
            self.dataset_enable_bucket_var.set(preset["ENABLE_BUCKET"])
        if "BUCKET_NO_UPSCALE" in preset:
            self.dataset_no_upscale_var.set(preset["BUCKET_NO_UPSCALE"])
        # Dataset subsection (Training → Other Options → Dataset)
        if "DATASET_CAPTION_EXT" in preset and hasattr(self, "dataset_caption_ext_var"):
            self.dataset_caption_ext_var.set(preset["DATASET_CAPTION_EXT"])
        if "DATASET_MEGAPIXELS" in preset and hasattr(self, "dataset_megapixels_var"):
            self.dataset_megapixels_var.set(preset["DATASET_MEGAPIXELS"])
        if "DATASET_BATCH_SIZE" in preset and hasattr(self, "dataset_batch_size_var"):
            self.dataset_batch_size_var.set(preset["DATASET_BATCH_SIZE"])
        # Run card's Enable Cache checkbox
        if "ENABLE_CACHE" in preset and hasattr(self, "enable_cache_var"):
            self.enable_cache_var.set(bool(preset["ENABLE_CACHE"]))
        # Per-block custom training selection (only meaningful when TARGET_LAYERS=Custom)
        if "TRAINING_BLOCKS" in preset and hasattr(self, "training_block_vars"):
            for block_key, block_on in preset["TRAINING_BLOCKS"].items():
                if block_key in self.training_block_vars:
                    self.training_block_vars[block_key].set(bool(block_on))
        self.toggle_scaled()  # Update checkbox state

    def _save_last_train_settings(self):
        """Snapshot current settings just before launching training, so 'Load Last Train' can restore them."""
        try:
            os.makedirs(PRESETS_DIR, exist_ok=True)
            snapshot = self._collect_preset_values()
            with open(LAST_TRAIN_FILE, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str)
        except Exception as e:
            print(f"[last_train] Failed to save snapshot: {e}")

    def _load_last_train_settings(self):
        """Restore settings from the most recent training launch."""
        if not os.path.exists(LAST_TRAIN_FILE):
            messagebox.showinfo(
                "No Last Train",
                "No previous training settings found.\n\n"
                "Launch a training run first; afterwards this button will restore those settings."
            )
            return
        try:
            with open(LAST_TRAIN_FILE, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            self._apply_preset_values(snapshot)
            messagebox.showinfo("Loaded", "Restored settings from your last training launch.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load last train settings:\n{e}")

    # Keys in self.entries that belong to OTHER tabs — skipped when collecting
    # a training-tab preset. Everything else in self.entries is fair game.
    _NON_TRAINING_ENTRY_KEYS = {
        "SAMPLE_ENABLED", "SAMPLE_WIDTH", "SAMPLE_HEIGHT", "SAMPLE_STEPS",
        "SAMPLE_SEED", "SAMPLE_EVERY_N_EPOCHS", "SAMPLE_EVERY_N_STEPS",
        "SAMPLE_AT_FIRST", "SAMPLE_FLOW_SHIFT", "SAMPLE_GUIDANCE",
        "SAMPLE_NEGATIVE", "SAMPLE_CFG_SCALE",
    }

    def _collect_preset_values(self):
        """Snapshot every user-editable value on the Training tab into a preset dict.

        Iterates all of self.entries (skipping keys that belong to other tabs or to
        system-level settings) plus every known Training-tab Boolean/StringVar — so
        saved presets capture anything the user touched in the Training UI, not just
        a hand-curated subset.
        """
        preset = {}

        # Everything in self.entries that's on the Training tab
        for key, entry in self.entries.items():
            if key in self._NON_TRAINING_ENTRY_KEYS:
                continue
            try:
                if isinstance(entry, (tk.BooleanVar, tk.StringVar, tk.IntVar, tk.DoubleVar)):
                    preset[key] = entry.get()
                else:
                    # ttk.Entry / ttk.Combobox / ttk.Spinbox all expose .get()
                    preset[key] = entry.get()
            except Exception:
                pass

        # Training-tab toggles that live on dedicated vars (not in self.entries)
        def _grab(attr, key):
            if hasattr(self, attr):
                try:
                    preset[key] = getattr(self, attr).get()
                except Exception:
                    pass

        _grab("preserve_dist_var", "PRESERVE_DISTRIBUTION")
        _grab("fp8_var", "FP8")
        _grab("scaled_var", "SCALED")
        _grab("fp8_text_encoder_var", "FP8_TEXT_ENCODER")
        _grab("adaptive_lr_var", "ADAPTIVE_LR")
        _grab("training_preset_var", "TARGET_LAYERS")
        _grab("ts_sampling_var", "TIMESTEP_SAMPLING")
        _grab("weighting_scheme_var", "WEIGHTING_SCHEME")
        _grab("enable_cache_var", "ENABLE_CACHE")
        # Dataset subsection (now living in Training → Other Options)
        _grab("dataset_enable_bucket_var", "ENABLE_BUCKET")
        _grab("dataset_no_upscale_var", "BUCKET_NO_UPSCALE")
        _grab("dataset_caption_ext_var", "DATASET_CAPTION_EXT")
        _grab("dataset_megapixels_var", "DATASET_MEGAPIXELS")
        _grab("dataset_batch_size_var", "DATASET_BATCH_SIZE")
        # Per-block custom training selection (only meaningful when TARGET_LAYERS=Custom)
        if hasattr(self, "training_block_vars") and self.training_block_vars:
            preset["TRAINING_BLOCKS"] = {k: v.get() for k, v in self.training_block_vars.items()}

        return preset

    def save_custom_preset(self):
        """Save current settings as a custom preset for the current architecture"""
        arch = self.architecture_var.get()

        # Prompt for preset name
        preset_name = simpledialog.askstring(
            "Save Preset",
            f"Enter a name for your preset (for {arch}):",
            parent=self.master
        )

        if not preset_name:
            return  # User cancelled

        # Validate name (no special chars that could cause filesystem issues)
        invalid_chars = '<>:"/\\|?*'
        if any(c in preset_name for c in invalid_chars):
            messagebox.showerror("Invalid Name", f"Preset name cannot contain: {invalid_chars}")
            return

        preset_name = preset_name.strip()
        if not preset_name:
            messagebox.showerror("Invalid Name", "Preset name cannot be empty")
            return

        # Check if preset already exists
        preset_dir = self.get_preset_dir_for_architecture(arch)
        preset_path = os.path.join(preset_dir, f"{preset_name}.json")

        if os.path.exists(preset_path):
            overwrite = messagebox.askyesno(
                "Preset Exists",
                f"A preset named '{preset_name}' already exists.\nDo you want to overwrite it?"
            )
            if not overwrite:
                return

        # Collect and save preset
        preset = self._collect_preset_values()
        try:
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset, f, indent=4)
            self.refresh_preset_combobox()
            messagebox.showinfo("Preset Saved", f"Preset '{preset_name}' saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preset: {str(e)}")

    def load_custom_preset(self, event=None):
        """Load a preset from the combobox selection — built-in or user-saved."""
        preset_name = self.custom_preset_var.get()
        if not preset_name:
            return

        # Check built-in presets first
        if preset_name in BUILT_IN_PRESETS:
            self._apply_preset_values(BUILT_IN_PRESETS[preset_name])
            messagebox.showinfo("Preset Loaded", f"Loaded built-in preset '{preset_name}'")
            return

        # Otherwise, look for a user-saved preset on disk
        arch = self.architecture_var.get()
        preset_dir = self.get_preset_dir_for_architecture(arch)
        preset_path = os.path.join(preset_dir, f"{preset_name}.json")

        if not os.path.exists(preset_path):
            messagebox.showerror("Error", f"Preset file not found: {preset_name}")
            self.refresh_preset_combobox()
            return

        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                preset = json.load(f)
            self._apply_preset_values(preset)
            messagebox.showinfo("Preset Loaded", f"Loaded preset '{preset_name}'")
        except json.JSONDecodeError:
            messagebox.showerror("Error", f"Preset file is corrupted: {preset_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preset: {str(e)}")

    def delete_custom_preset(self):
        """Delete the currently selected custom preset"""
        preset_name = self.custom_preset_var.get()
        if not preset_name:
            messagebox.showinfo("Info", "Please select a preset to delete")
            return

        arch = self.architecture_var.get()

        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete the preset '{preset_name}'?\n\nThis action cannot be undone."
        )
        if not confirm:
            return

        preset_dir = self.get_preset_dir_for_architecture(arch)
        preset_path = os.path.join(preset_dir, f"{preset_name}.json")

        try:
            if os.path.exists(preset_path):
                os.remove(preset_path)
                self.refresh_preset_combobox()
                messagebox.showinfo("Preset Deleted", f"Preset '{preset_name}' deleted successfully")
            else:
                messagebox.showerror("Error", f"Preset file not found: {preset_name}")
                self.refresh_preset_combobox()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete preset: {str(e)}")

    def update_ui_for_architecture(self):
        """Update UI elements based on selected architecture"""
        arch = self.architecture_var.get()
        config = ARCHITECTURES.get(arch, ARCHITECTURES["Flux 2 Klein Base 9B"])

        # Window title is set once in __init__ and stays put — architecture is
        # hardcoded to Klein Base 9B, no need to mirror it in the title bar.

        # Show/hide CLIP model field
        if config["uses_clip"]:
            self.show_row("CLIP_MODEL")
        else:
            self.hide_row("CLIP_MODEL")

        # Show/hide T5 model field
        if config["uses_t5"]:
            self.show_row("T5_MODEL")
        else:
            self.hide_row("T5_MODEL")

        # Show/hide Text Encoder field (for Z-Image and Flux 2)
        if config["uses_text_encoder"]:
            self.show_row("TEXT_ENCODER")
            if "TEXT_ENCODER" in self.labels:  # Model Paths section may have been removed
                self.labels["TEXT_ENCODER"].config(text=f"{config['text_encoder_label']}:")
        else:
            self.hide_row("TEXT_ENCODER")

        # Show/hide Model Type dropdown (Wan only) and update options
        if config["uses_model_type"]:
            self.show_row("MODEL_TYPE")
            # Update MODEL_TYPE dropdown values for this architecture
            model_types = config.get("model_types", ["t2v-14B", "i2v-14B"])
            self.entries["MODEL_TYPE"]["values"] = model_types
            current_val = self.entries["MODEL_TYPE"].get()
            if current_val not in model_types:
                self.entries["MODEL_TYPE"].current(0)
        else:
            self.hide_row("MODEL_TYPE")

        # Update VAE label (Model Paths section may have been removed)
        if "VAE_MODEL" in self.labels:
            self.labels["VAE_MODEL"].config(text=f"{config['vae_label']}:")

        # Update FP8 text encoder checkbox label
        if arch.startswith("Wan"):
            self.fp8_text_encoder_check.config(text="Enable FP8 T5")
        elif arch.startswith("Z-Image"):
            self.fp8_text_encoder_check.config(text="Enable FP8 LLM")
        else:
            self.fp8_text_encoder_check.config(text="Enable FP8 Text Encoder")

        # Update blocks swap max (enforce limit)
        try:
            current_blocks = self._parse_blocks_swap()
            if current_blocks > config["blocks_swap_max"]:
                self.entries["BLOCKS_SWAP"].delete(0, tk.END)
                self.entries["BLOCKS_SWAP"].insert(0, str(config["blocks_swap_max"]))
        except ValueError:
            pass

        # Update timestep section for architecture
        if hasattr(self, 'ts_sampling_var'):
            # Set timestep sampling to architecture default
            timestep_sampling = config.get("timestep_sampling", "shift")
            self.ts_sampling_var.set(timestep_sampling)

            # Discrete Flow Shift
            supports_shift = config.get("supports_discrete_flow_shift", True)
            if supports_shift:
                self.entries["DISCRETE_FLOW_SHIFT"].config(state="normal")
                self.ts_flow_shift_label.config(fg=COLORS["text_secondary"])
                # Set architecture default
                default_shift = config.get("discrete_flow_shift")
                if default_shift is not None:
                    self.entries["DISCRETE_FLOW_SHIFT"].delete(0, tk.END)
                    self.entries["DISCRETE_FLOW_SHIFT"].insert(0, str(default_shift))
            else:
                self.entries["DISCRETE_FLOW_SHIFT"].config(state="disabled")
                self.ts_flow_shift_label.config(fg=COLORS["text_muted"])

            # Min/Max Timestep defaults from architecture
            min_ts = config.get("min_timestep")
            max_ts = config.get("max_timestep")
            self.entries["MIN_TIMESTEP"].delete(0, tk.END)
            self.entries["MAX_TIMESTEP"].delete(0, tk.END)
            if min_ts is not None:
                self.entries["MIN_TIMESTEP"].insert(0, str(min_ts))
            if max_ts is not None:
                self.entries["MAX_TIMESTEP"].insert(0, str(max_ts))

            # Preserve distribution
            self.preserve_dist_var.set(config.get("preserve_distribution_shape", False))

            # Weighting scheme
            supports_weighting = config.get("supports_weighting_scheme", True)
            if supports_weighting:
                self.ts_weighting_combo.config(state="readonly")
                self.ts_weighting_label.config(fg=COLORS["text_secondary"])
            else:
                self.weighting_scheme_var.set("none")
                self.ts_weighting_combo.config(state="disabled")
                self.ts_weighting_label.config(fg=COLORS["text_muted"])

            # Refresh conditional field states
            self._on_timestep_sampling_changed()
            self._on_weighting_scheme_changed()
            self._update_noise_range_label()

    # ── Timestep section helpers ────────────────────────────────────────

    def _on_adaptive_lr_toggle(self):
        """Enable/disable the Min/Max LR dropdowns based on the Adaptive LR checkbox."""
        if not hasattr(self, 'entries') or "ADAPTIVE_LR_MIN" not in self.entries:
            return
        # Comboboxes: "readonly" when enabled (dropdown active, no free typing), "disabled" when not
        combo_state = "readonly" if self.adaptive_lr_var.get() else "disabled"
        btn_state = "normal" if self.adaptive_lr_var.get() else "disabled"
        self.entries["ADAPTIVE_LR_MIN"].config(state=combo_state)
        self.entries["ADAPTIVE_LR_MAX"].config(state=combo_state)
        if hasattr(self, '_adaptive_reset_btn'):
            self._adaptive_reset_btn.config(state=btn_state)

    def _parse_blocks_swap(self) -> int:
        """Extract integer from the BLOCKS_SWAP combobox value.
        'Auto' resolves to a value based on GPU VRAM (training needs more headroom than inference)."""
        import re as _re
        raw = self.entries["BLOCKS_SWAP"].get().strip()
        if raw.lower().startswith("auto"):
            return self._auto_training_blocks_swap()
        m = _re.match(r'\d+', raw)
        return int(m.group()) if m else 0

    def _auto_training_blocks_swap(self) -> int:
        """Pick training block swap based on GPU VRAM."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if vram_gb >= 30:
                    return 0   # 32 GB+ (5090, A100) — no swap needed
                if vram_gb >= 20:
                    return 4   # 24 GB (RTX 3090 / 4090)
                if vram_gb >= 14:
                    return 8   # 16 GB cards
                if vram_gb >= 10:
                    return 12  # 12 GB cards
                return 16      # <10 GB — maximum swap
        except Exception:
            pass
        return 8  # safe fallback

    def _get_inference_blocks_to_swap(self) -> int:
        """Parse the leading int from the Preferences inference_blocks_to_swap
        pref. Labeled options like '16 (Max — …)' store as the full string; we
        just take the leading integer. Returns 0 on any parse failure."""
        import re as _re
        raw = ""
        try:
            raw = str(self.prefs_vars["inference_blocks_to_swap"].get()).strip()
        except Exception:
            return 0
        m = _re.match(r'\d+', raw)
        return int(m.group()) if m else 0

    def _resolve_script(self, config: dict, script_key: str) -> str:
        """Resolve an absolute script path from an architecture config entry.

        Klein lives under FIZGIG_DIR — strip any legacy "FizgigIndependent/" prefix
        on the config value (back-compat with older config strings) and join onto
        FIZGIG_DIR.
        """
        rel = config[script_key]
        if rel.startswith("FizgigIndependent/"):
            rel = rel[len("FizgigIndependent/"):]
        return os.path.join(FIZGIG_DIR, rel)

    def _get_path(self, key: str) -> str:
        """Resolve a model/path setting from the current source of truth.

        Pulls from prefs_vars (model paths) or from the hidden _dataset_config_var
        (dataset config).
        """
        pref_map = {
            "VAE_MODEL": "vae",
            "DIT_MODEL": "base_dit",
            "TEXT_ENCODER": "text_encoder",
            "LORA_OUTPUT_DIR": "lora_output_dir",
        }
        pref_key = pref_map.get(key)
        if pref_key and pref_key in self.prefs_vars:
            return self.prefs_vars[pref_key].get()
        if key == "DATASET_CONFIG":
            return self._dataset_config_var.get() if hasattr(self, "_dataset_config_var") else ""
        return ""

    def _reset_adaptive_lr_defaults(self):
        """Reset Learning Rate, Min LR, and Max LR to adaptive-mode defaults."""
        # Learning Rate is a free-text entry; Min/Max LR are comboboxes.
        lr_entry = self.entries.get("LEARNING_RATE")
        if lr_entry is not None:
            lr_entry.delete(0, tk.END)
            lr_entry.insert(0, "4e-4")
        for key, value in (("ADAPTIVE_LR_MIN", "1e-5"), ("ADAPTIVE_LR_MAX", "4e-4")):
            entry = self.entries.get(key)
            if entry is not None:
                entry.config(state="readonly")
                entry.set(value)
        # Re-apply enabled/disabled state on the adaptive fields
        self._on_adaptive_lr_toggle()

    def _on_timestep_sampling_changed(self, event=None):
        """Enable/disable sigmoid_scale based on selected sampling method."""
        sampling = self.ts_sampling_var.get()
        uses_sigmoid = sampling in ("sigmoid", "shift")
        state = "normal" if uses_sigmoid else "disabled"
        color = COLORS["text_secondary"] if uses_sigmoid else COLORS["text_muted"]
        self.entries["SIGMOID_SCALE"].config(state=state)
        self.ts_sigmoid_label.config(fg=color)

    def _on_weighting_scheme_changed(self, event=None):
        """Enable/disable logit_mean/std and mode_scale based on weighting scheme."""
        scheme = self.weighting_scheme_var.get()

        # Logit Normal params
        is_logit = (scheme == "logit_normal")
        logit_state = "normal" if is_logit else "disabled"
        logit_color = COLORS["text_secondary"] if is_logit else COLORS["text_muted"]
        self.entries["LOGIT_MEAN"].config(state=logit_state)
        self.entries["LOGIT_STD"].config(state=logit_state)
        self.ts_logit_label.config(fg=logit_color)

        # Mode Scale param
        is_mode = (scheme == "mode")
        mode_state = "normal" if is_mode else "disabled"
        mode_color = COLORS["text_secondary"] if is_mode else COLORS["text_muted"]
        self.entries["MODE_SCALE"].config(state=mode_state)
        self.ts_mode_label.config(fg=mode_color)

    def _update_noise_range_label(self):
        """Update the dynamic noise range description label."""
        if not hasattr(self, 'noise_range_label'):
            return
        min_str = self.entries["MIN_TIMESTEP"].get().strip()
        max_str = self.entries["MAX_TIMESTEP"].get().strip()

        if not min_str and not max_str:
            self.noise_range_label.config(text="Full range (default) - All noise levels",
                                          fg=COLORS["accent"])
            return

        try:
            min_val = int(min_str) if min_str else 0
            max_val = int(max_str) if max_str else 1000
        except ValueError:
            self.noise_range_label.config(text="Invalid timestep values", fg=COLORS["error"])
            return

        if min_val == 0 and max_val >= 1000:
            self.noise_range_label.config(text=f"Full range ({min_val}-{max_val}) - All noise levels",
                                          fg=COLORS["accent"])
        elif max_val <= 300:
            self.noise_range_label.config(text=f"High noise ({min_val}-{max_val}) - Composition/structure",
                                          fg=COLORS["success"])
        elif min_val >= 700:
            self.noise_range_label.config(text=f"Low noise ({min_val}-{max_val}) - Details/textures",
                                          fg="#B388FF")  # purple
        elif min_val >= 300 and max_val <= 700:
            self.noise_range_label.config(text=f"Mid noise ({min_val}-{max_val}) - Features/characteristics",
                                          fg=COLORS["warning"])
        else:
            self.noise_range_label.config(text=f"Custom range ({min_val}-{max_val})",
                                          fg=COLORS["text_secondary"])

    def _ts_preset_full_range(self):
        """Timestep preset: Full Range"""
        self.entries["MIN_TIMESTEP"].delete(0, tk.END)
        self.entries["MAX_TIMESTEP"].delete(0, tk.END)
        self.weighting_scheme_var.set("none")
        self._on_weighting_scheme_changed()
        self._update_noise_range_label()

    def _ts_preset_structure(self):
        """Timestep preset: Structure Focus (high noise)"""
        self.entries["MIN_TIMESTEP"].delete(0, tk.END)
        self.entries["MIN_TIMESTEP"].insert(0, "0")
        self.entries["MAX_TIMESTEP"].delete(0, tk.END)
        self.entries["MAX_TIMESTEP"].insert(0, "300")
        self.weighting_scheme_var.set("none")
        self._on_weighting_scheme_changed()
        self._update_noise_range_label()

    def _ts_preset_detail(self):
        """Timestep preset: Detail Focus (low noise)"""
        self.entries["MIN_TIMESTEP"].delete(0, tk.END)
        self.entries["MIN_TIMESTEP"].insert(0, "700")
        self.entries["MAX_TIMESTEP"].delete(0, tk.END)
        self.entries["MAX_TIMESTEP"].insert(0, "1000")
        self.weighting_scheme_var.set("none")
        self._on_weighting_scheme_changed()
        self._update_noise_range_label()

    def _ts_preset_sigmoid(self):
        """Timestep preset: Balanced Sigmoid"""
        self.ts_sampling_var.set("sigmoid")
        self.entries["MIN_TIMESTEP"].delete(0, tk.END)
        self.entries["MAX_TIMESTEP"].delete(0, tk.END)
        self.weighting_scheme_var.set("none")
        self._on_timestep_sampling_changed()
        self._on_weighting_scheme_changed()
        self._update_noise_range_label()

    def show_row(self, key):
        """Show a row by its key"""
        if key in self.rows:
            row_info = self.rows[key]
            row_info["label"].grid()
            row_info["entry"].grid()
            if row_info["browse"]:
                row_info["browse"].grid()

    def hide_row(self, key):
        """Hide a row by its key"""
        if key in self.rows:
            row_info = self.rows[key]
            row_info["label"].grid_remove()
            row_info["entry"].grid_remove()
            if row_info["browse"]:
                row_info["browse"].grid_remove()

    def toggle_scaled(self):
        """Enable or disable the Scaled checkbox based on FP8 checkbox state"""
        if self.fp8_var.get():
            self.scaled_check.config(state=tk.NORMAL)
        else:
            self.scaled_check.config(state=tk.DISABLED)
            self.scaled_var.set(False)

    def _populate_other_options(self, parent, start_row=0):
        """Populate Attention / Logging / Memory / Metadata fields onto the given parent.
        Used to inline these into the Other Options section on the Training tab."""
        row = start_row

        # Attention Mechanism
        ttk.Label(parent, text="Attention Mechanism:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.attention_var = tk.StringVar(value=self.settings["ATTENTION_MECHANISM"])
        attention_options = ["sdpa", "flash3"]
        self.entries["ATTENTION_MECHANISM"] = ttk.Combobox(parent, textvariable=self.attention_var, values=attention_options, state="readonly")
        self.entries["ATTENTION_MECHANISM"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1
        ttk.Label(parent, text="sdpa works on all GPUs. flash3 requires pip install flash-attn and an "
                  "NVIDIA Hopper/Blackwell GPU (H100, RTX 5090, etc.).",
                  foreground="#95A5A6", font=(FONT_FAMILY, 8, "italic")).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, padx=5)
        row += 1

        # Logging
        ttk.Label(parent, text="Logging Directory:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["LOGGING_DIR"] = ttk.Entry(parent, width=40)
        self.entries["LOGGING_DIR"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(parent, text="Browse", command=lambda: self.browse_directory("LOGGING_DIR")).grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(parent, text="Log With:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.log_with_var = tk.StringVar(value=self.settings["LOG_WITH"])
        log_with_options = ["none", "tensorboard", "wandb", "all"]
        self.entries["LOG_WITH"] = ttk.Combobox(parent, textvariable=self.log_with_var, values=log_with_options, state="readonly")
        self.entries["LOG_WITH"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Log Prefix:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["LOG_PREFIX"] = ttk.Entry(parent, width=40)
        self.entries["LOG_PREFIX"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        # Memory Management
        self.img_in_txt_in_offloading_var = tk.BooleanVar(value=self.settings["IMG_IN_TXT_IN_OFFLOADING"])
        ttk.Checkbutton(parent, text="Offload img_in and txt_in to CPU", variable=self.img_in_txt_in_offloading_var).grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        self.entries["IMG_IN_TXT_IN_OFFLOADING"] = self.img_in_txt_in_offloading_var
        row += 1

        # Metadata
        ttk.Label(parent, text="Metadata Title:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_TITLE"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_TITLE"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Metadata Author:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_AUTHOR"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_AUTHOR"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Metadata Description:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_DESCRIPTION"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_DESCRIPTION"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Metadata License:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_LICENSE"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_LICENSE"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Metadata Tags:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_TAGS"] = ttk.Entry(parent, width=40)
        self.entries["METADATA_TAGS"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        return row

    def create_caption_generator(self):
        """Create the Captions tab (Start-tab styled)."""
        scrollable_frame, self.caption_canvas = self.create_scrollable_frame(self.caption_gen_tab)

        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer,
            "Captions",
            "Write trigger-word captions or generate them with Florence-2 AI. "
            "Skip this tab if your images already have .txt caption files.",
        )

        # Card 1: Captioning Settings
        settings_card = self._start_section_card(
            outer, "Captioning Settings",
            "Trigger word is prepended to every caption. Florence produces detailed image descriptions; "
            "Static Caption writes the trigger word only.",
        )
        settings_card.grid_columnconfigure(1, weight=1)

        ttk.Label(settings_card, text="Image Folder:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self._caption_folder_display = tk.Label(settings_card, textvariable=self.image_folder_var,
                                                font=(FONT_FAMILY, 10),
                                                fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
                                                anchor="w")
        self._caption_folder_display.grid(row=0, column=1, columnspan=2, sticky=tk.W, pady=4)
        tk.Label(settings_card, text="(set on the Start tab)",
                 font=(FONT_FAMILY, 9, "italic"),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=1, column=1, columnspan=2, sticky=tk.W, pady=(0, 8)
        )

        ttk.Label(settings_card, text="Trigger Word:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.caption_trigger_var = tk.StringVar(value=self.settings.get("CAPTION_TRIGGER_WORD", ""))
        ttk.Entry(settings_card, textvariable=self.caption_trigger_var, width=40).grid(row=2, column=1, sticky=tk.W, pady=4)
        tk.Label(settings_card, text="(prepended to all captions)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=2, column=2, sticky=tk.W, padx=(10, 0)
        )

        ttk.Label(settings_card, text="Model:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.caption_model_var = tk.StringVar(value=self.settings.get("CAPTION_MODEL", "MiaoshouAI/Florence-2-base-PromptGen"))
        self.caption_model_combo = ttk.Combobox(
            settings_card, textvariable=self.caption_model_var,
            values=["MiaoshouAI/Florence-2-base-PromptGen",
                    "microsoft/Florence-2-base",
                    "microsoft/Florence-2-large"],
            state="readonly", width=37,
        )
        self.caption_model_combo.grid(row=3, column=1, sticky=tk.W, pady=4)

        ttk.Label(settings_card, text="Task:").grid(row=4, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.caption_task_var = tk.StringVar(value=self.settings.get("CAPTION_TASK", "<DETAILED_CAPTION>"))
        self.caption_task_combo = ttk.Combobox(
            settings_card, textvariable=self.caption_task_var,
            values=["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"],
            state="readonly", width=37,
        )
        self.caption_task_combo.grid(row=4, column=1, sticky=tk.W, pady=4)

        ttk.Label(settings_card, text="Max Tokens:").grid(row=5, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.caption_max_tokens_var = tk.StringVar(value=str(self.settings.get("CAPTION_MAX_TOKENS", 256)))
        ttk.Entry(settings_card, textvariable=self.caption_max_tokens_var, width=10).grid(row=5, column=1, sticky=tk.W, pady=4)

        ttk.Checkbutton(
            settings_card, text="Overwrite existing caption files", variable=self.overwrite_captions_var,
        ).grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))

        # Card 2: Generate Captions
        actions_card = self._start_section_card(outer, "Generate Captions", None)
        action_row = tk.Frame(actions_card, bg=COLORS["bg_surface"])
        action_row.pack(anchor=tk.W)
        ttk.Button(action_row, text="Caption All Images (AI)", command=self.caption_all_florence).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(action_row, text="Caption Selected", command=self.caption_selected_florence).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(action_row, text="Static Caption All", command=self.generate_captions).pack(side=tk.LEFT, padx=(0, 8))
        self.caption_stop_btn = ttk.Button(action_row, text="Stop", command=self.stop_captioning, state=tk.DISABLED)
        self.caption_stop_btn.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(action_row, text="Unload Model", command=self.unload_florence_model).pack(side=tk.LEFT)

        # Card 3: Bilingual translation
        bilingual_card = self._start_section_card(
            outer, "Bilingual Translation (English + Chinese)",
            "Translates each English caption to Chinese via Helsinki-NLP/opus-mt-en-zh (~300MB, auto-downloaded "
            "on first use) and appends as `english - chinese`. Trigger word preserved if it's the first "
            "comma-separated token. Hypothesis: dual-language signal may improve LoRA convergence — "
            "empirical test needed.",
        )
        bilingual_row = tk.Frame(bilingual_card, bg=COLORS["bg_surface"])
        bilingual_row.pack(anchor=tk.W)
        ttk.Checkbutton(
            bilingual_row, text="Skip files that already contain Chinese",
            variable=self.skip_bilingual_var,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Button(
            bilingual_row, text="Translate Captions in Folder",
            command=self._translate_captions_in_folder,
        ).pack(side=tk.LEFT)

        # Card 4: Find & Replace
        fr_card = self._start_section_card(
            outer, "Find & Replace",
            "Bulk-edit every `.txt` caption file in the image folder. Preview first to see which files change.",
        )
        fr_card.grid_columnconfigure(1, weight=1)

        ttk.Label(fr_card, text="Find:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.find_text_var = tk.StringVar()
        ttk.Entry(fr_card, textvariable=self.find_text_var, width=40).grid(row=0, column=1, sticky=tk.EW, pady=4)

        ttk.Label(fr_card, text="Replace:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.replace_text_var = tk.StringVar()
        ttk.Entry(fr_card, textvariable=self.replace_text_var, width=40).grid(row=1, column=1, sticky=tk.EW, pady=4)

        fr_buttons = tk.Frame(fr_card, bg=COLORS["bg_surface"])
        fr_buttons.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))
        ttk.Button(fr_buttons, text="Replace in All .txt Files", command=self.find_replace_in_captions).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(fr_buttons, text="Preview Changes", command=self.preview_find_replace).pack(side=tk.LEFT)

        # Card 5: Image Preview
        preview_card = self._start_section_card(
            outer, "Image Preview",
            "Browse the training folder and pick individual images to caption or inspect.",
        )

        self.caption_grid_frame = tk.Frame(preview_card, bg=COLORS["bg_surface"])
        self.caption_grid_frame.pack(fill=tk.BOTH, expand=True)
        for _c in range(4):
            self.caption_grid_frame.columnconfigure(_c, weight=1)

        pagination_frame = tk.Frame(preview_card, bg=COLORS["bg_surface"])
        pagination_frame.pack(pady=(10, 0))
        ttk.Button(pagination_frame, text="<< Prev", command=self.caption_prev_page).pack(side=tk.LEFT, padx=(0, 8))
        self.caption_page_label = tk.Label(pagination_frame, text="Page 0 of 0",
                                           font=(FONT_FAMILY, 10),
                                           fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self.caption_page_label.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(pagination_frame, text="Next >>", command=self.caption_next_page).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Button(pagination_frame, text="Refresh", command=self.refresh_caption_images).pack(side=tk.LEFT)

        # Card 6: Progress
        progress_card = self._start_section_card(outer, "Progress", None)
        progress_row = tk.Frame(progress_card, bg=COLORS["bg_surface"])
        progress_row.pack(fill=tk.X)
        self.caption_progress_var = tk.DoubleVar(value=0)
        self.caption_progress_bar = ttk.Progressbar(
            progress_row, variable=self.caption_progress_var, maximum=100, length=300,
        )
        self.caption_progress_bar.pack(side=tk.LEFT, padx=(0, 12))
        self.caption_progress_label = tk.Label(progress_row, text="",
                                               font=(FONT_FAMILY, 10),
                                               fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        self.caption_progress_label.pack(side=tk.LEFT)

        # Card 7: Output Log
        log_card = self._start_section_card(outer, "Output Log", None)
        self.caption_log = scrolledtext.ScrolledText(
            log_card, height=10, width=80,
            bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
            wrap="word", state="disabled",
            relief="flat", highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["border_focus"],
        )
        self.caption_log.pack(fill=tk.BOTH, expand=True)

        self._add_youtube_help_button(outer, "captions")

    def browse_caption_folder_and_refresh(self):
        """Browse for caption folder and refresh image grid"""
        folder = filedialog.askdirectory()
        if folder:
            self.image_folder_var.set(folder)
            self.refresh_caption_images()

    def get_caption_image_files(self):
        """Get list of image files in caption folder"""
        folder = self.image_folder_var.get()
        if not folder or not os.path.isdir(folder):
            return []

        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
        images = []

        for filename in os.listdir(folder):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                images.append(os.path.join(folder, filename))

        return sorted(images)

    def refresh_caption_images(self):
        """Refresh the image grid with thumbnails"""
        # Mark as loaded for this folder
        self.caption_images_loaded = True

        # Clear existing thumbnails
        for widget in self.caption_grid_frame.winfo_children():
            widget.destroy()
        self.caption_thumbnails.clear()
        self.selected_images.clear()

        images = self.get_caption_image_files()
        total_images = len(images)
        total_pages = max(1, (total_images + self.images_per_page - 1) // self.images_per_page)

        # Clamp current page
        self.current_caption_page = min(self.current_caption_page, total_pages - 1)
        self.current_caption_page = max(0, self.current_caption_page)

        # Update page label
        self.caption_page_label.config(text=f"Page {self.current_caption_page + 1} of {total_pages} ({total_images} images)")

        # Get images for current page
        start_idx = self.current_caption_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, total_images)
        page_images = images[start_idx:end_idx]

        # Create image cards in a grid (4 columns)
        for i, img_path in enumerate(page_images):
            row_idx = i // 4
            col_idx = i % 4
            self.create_caption_image_card(img_path, row_idx, col_idx)

    def create_caption_image_card(self, img_path, row, col):
        """Create an image card with thumbnail and caption"""
        card_frame = ttk.Frame(self.caption_grid_frame, relief="solid", borderwidth=1)
        card_frame.grid(row=row, column=col, padx=5, pady=5, sticky=tk.NSEW)

        # Create thumbnail
        try:
            with Image.open(img_path) as img:
                img.thumbnail((150, 150), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.caption_thumbnails[img_path] = photo  # Keep reference

                img_label = ttk.Label(card_frame, image=photo)
                img_label.pack(padx=5, pady=5)
        except Exception as e:
            ttk.Label(card_frame, text="Error loading image").pack(padx=5, pady=5)

        # Filename
        filename = os.path.basename(img_path)
        name_label = ttk.Label(card_frame, text=filename[:20] + "..." if len(filename) > 20 else filename)
        name_label.pack()

        # Load and display caption if exists
        caption_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(caption_path):
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                caption_preview = caption[:50] + "..." if len(caption) > 50 else caption
                caption_label = ttk.Label(card_frame, text=caption_preview, wraplength=140, foreground=COLORS["text_secondary"])
                caption_label.pack(pady=2)
            except Exception:
                pass
        else:
            ttk.Label(card_frame, text="[No caption]", foreground=COLORS["warning"]).pack(pady=2)

        # Selection checkbox
        var = tk.BooleanVar(value=img_path in self.selected_images)
        def on_select(path=img_path, v=var):
            if v.get():
                self.selected_images.add(path)
            else:
                self.selected_images.discard(path)
        check = ttk.Checkbutton(card_frame, text="Select", variable=var, command=on_select)
        check.pack()

        # Edit button
        ttk.Button(card_frame, text="Edit", command=lambda p=img_path: self.show_edit_caption_dialog(p)).pack(pady=2)

    def caption_prev_page(self):
        """Go to previous page of images"""
        if self.current_caption_page > 0:
            self.current_caption_page -= 1
            self.refresh_caption_images()

    def caption_next_page(self):
        """Go to next page of images"""
        images = self.get_caption_image_files()
        total_pages = max(1, (len(images) + self.images_per_page - 1) // self.images_per_page)
        if self.current_caption_page < total_pages - 1:
            self.current_caption_page += 1
            self.refresh_caption_images()

    def show_edit_caption_dialog(self, img_path):
        """Show dialog to edit caption for an image"""
        dialog = tk.Toplevel(self.master)
        dialog.title(f"Edit Caption - {os.path.basename(img_path)}")
        dialog.geometry("600x500")
        dialog.configure(bg=BG_COLOR)

        # Image preview
        try:
            with Image.open(img_path) as img:
                img.thumbnail((300, 300), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label = ttk.Label(dialog, image=photo)
                img_label.image = photo
                img_label.pack(pady=10)
        except Exception:
            ttk.Label(dialog, text="Could not load image preview").pack(pady=10)

        # Caption text area
        ttk.Label(dialog, text="Caption:").pack(anchor=tk.W, padx=10)
        caption_text = tk.Text(dialog, height=5, width=60, bg=COLORS["bg_surface"], fg=COLORS["text_primary"], font=(FONT_FAMILY, 10), wrap="word")
        caption_text.pack(padx=10, pady=5, fill=tk.X)

        # Load existing caption
        caption_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(caption_path):
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption_text.insert("1.0", f.read())
            except Exception:
                pass

        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)

        def save_caption():
            text = caption_text.get("1.0", tk.END).strip()
            try:
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("Saved", "Caption saved successfully")
                self.refresh_caption_images()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save caption: {e}")

        def regenerate():
            dialog.destroy()
            self.caption_single_image(img_path)

        ttk.Button(btn_frame, text="Save", command=save_caption).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Regenerate (AI)", command=regenerate).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def load_florence_model(self):
        """Load Florence model (lazy loading)"""
        if self.florence_model is not None:
            return True

        try:
            model_name = self.caption_model_var.get()
            self.update_caption_log(f"Loading {model_name}...\n")
            self.update_caption_log("(First run will download model from Hugging Face - ~500MB-1.5GB)\n")
            self.master.update_idletasks()

            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

            self.update_caption_log("Loading processor...\n")
            self.master.update_idletasks()

            self.florence_processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            self.update_caption_log("Loading model weights...\n")
            self.master.update_idletasks()

            # Florence-2's custom code defines _supports_sdpa as a property that
            # accesses self.language_model — but transformers 4.50+ reads it during
            # __init__ before language_model exists, causing AttributeError.
            # attn_implementation="eager" bypasses the SDPA check entirely.
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                attn_implementation="eager"
            ).to(device)

            self.florence_device = device
            self.update_caption_log(f"Model loaded successfully on {device.upper()}\n")
            return True

        except ImportError as e:
            self.update_caption_log(f"Error: transformers not installed\n")
            self.update_caption_log("Run: pip install transformers einops\n")
            messagebox.showerror("Missing Dependency",
                "transformers library not installed.\n\n"
                "Run the Fizgig installer to add Florence support:\n"
                "python install_fizgig.py\n\n"
                "Or manually: pip install transformers einops")
            return False

        except Exception as e:
            self.update_caption_log(f"Error loading model: {e}\n")
            messagebox.showerror("Error", f"Failed to load Florence model:\n{e}")
            return False

    def unload_florence_model(self, silent=False):
        """Unload Florence model to free memory"""
        if self.florence_model is not None:
            import torch
            del self.florence_model
            del self.florence_processor
            self.florence_model = None
            self.florence_processor = None
            self.florence_device = None
            torch.cuda.empty_cache()
            self.update_caption_log("Model unloaded\n")
            if not silent:
                messagebox.showinfo("Model Unloaded", "Florence model unloaded. VRAM freed.")
        elif not silent:
            messagebox.showinfo("Info", "No model loaded")

    # === Bilingual translation (Qwen3-8B chat) ===

    def _has_cjk(self, text: str) -> bool:
        """Heuristic: does the text contain CJK Unified Ideographs?"""
        return any('\u4e00' <= ch <= '\u9fff' for ch in text)

    def _load_translator(self):
        """Lazy-load Helsinki-NLP/opus-mt-en-zh for English→Chinese translation.

        First-time use downloads ~300MB from HuggingFace to the standard HF cache.
        We use this dedicated MT model (not Klein's Qwen3) because Klein's distributed
        text_encoder safetensors strips the LM head — it can extract hidden states for
        Klein's training but cannot do generation reliably.
        """
        if getattr(self, "_translator_model", None) is not None:
            return True
        try:
            self.update_caption_log("Loading translator (Helsinki-NLP/opus-mt-en-zh)...\n")
            import torch as _torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            device = "cuda" if _torch.cuda.is_available() else "cpu"
            model_id = "Helsinki-NLP/opus-mt-en-zh"
            self._translator_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._translator_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                torch_dtype=_torch.float16 if device == "cuda" else _torch.float32,
            ).to(device)
            self._translator_model.eval()
            self._translator_device = device
            self.update_caption_log(f"Translator loaded on {device}.\n")
            return True
        except Exception as e:
            self.update_caption_log(f"Failed to load translator: {e}\n")
            messagebox.showerror(
                "Error",
                f"Failed to load Helsinki-NLP/opus-mt-en-zh:\n{e}\n\n"
                "First-time use needs internet to download ~300MB from HuggingFace."
            )
            self._translator_model = None
            self._translator_tokenizer = None
            return False

    def _unload_translator(self):
        """Free VRAM after batch translation."""
        if getattr(self, "_translator_model", None) is not None:
            import torch as _torch
            del self._translator_model
            del self._translator_tokenizer
            self._translator_model = None
            self._translator_tokenizer = None
            _torch.cuda.empty_cache()
            self.update_caption_log("Translator unloaded.\n")

    def _translate_to_chinese(self, english: str) -> str:
        """Single-string EN→ZH translation via Helsinki MT. Returns Chinese only, or '' on failure."""
        if not english.strip():
            return ""
        import torch as _torch
        inputs = self._translator_tokenizer(
            english.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._translator_device)
        with _torch.no_grad():
            outputs = self._translator_model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
            )
        translation = self._translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = translation.strip().strip('"').strip("'").strip()
        translation = " ".join(translation.split())
        return translation

    def _translate_captions_in_folder(self):
        """Threaded entry point for the 'Translate Captions in Folder' button."""
        folder = self.image_folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Pick a valid caption folder first.")
            return
        import threading
        threading.Thread(target=self._translate_captions_worker, args=(folder,), daemon=True).start()

    def _translate_captions_worker(self, folder: str):
        """Background worker: load Qwen3, walk folder, translate each .txt, write back."""
        import glob
        files = sorted(glob.glob(os.path.join(folder, "*.txt")))
        if not files:
            self.master.after(0, lambda: self.update_caption_log(f"No .txt files in {folder}\n"))
            return
        self.master.after(0, lambda: self.update_caption_log(
            f"\n=== Bilingual translation: {len(files)} files in {folder} ===\n"
        ))
        if not self._load_translator():
            return
        skip_existing = self.skip_bilingual_var.get()
        translated = 0
        skipped = 0
        failed = 0
        try:
            for i, fp in enumerate(files):
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        original = f.read().strip()
                    if not original:
                        skipped += 1
                        continue
                    if skip_existing and self._has_cjk(original):
                        skipped += 1
                        self.master.after(0, lambda fp=fp: self.update_caption_log(
                            f"  [skip already-bilingual] {os.path.basename(fp)}\n"
                        ))
                        continue
                    # Trigger-word preservation: split on first ", "
                    if ", " in original:
                        trigger, eng_rest = original.split(", ", 1)
                    else:
                        trigger, eng_rest = "", original
                    chinese = self._translate_to_chinese(eng_rest if eng_rest else original)
                    if not chinese:
                        failed += 1
                        self.master.after(0, lambda fp=fp: self.update_caption_log(
                            f"  [fail empty translation] {os.path.basename(fp)} (kept original)\n"
                        ))
                        continue
                    if trigger:
                        new_text = f"{trigger}, {eng_rest} - {chinese}"
                    else:
                        new_text = f"{original} - {chinese}"
                    with open(fp, "w", encoding="utf-8") as f:
                        f.write(new_text)
                    translated += 1
                    self.master.after(0, lambda fp=fp, c=chinese: self.update_caption_log(
                        f"  [ok] {os.path.basename(fp)}  →  ...{c[:40]}\n"
                    ))
                except Exception as e:
                    failed += 1
                    self.master.after(0, lambda fp=fp, e=e: self.update_caption_log(
                        f"  [fail {type(e).__name__}] {os.path.basename(fp)}: {e}\n"
                    ))
                self.master.after(0, lambda i=i: self.update_caption_progress(
                    (i + 1) / len(files) * 100, i + 1, len(files)
                ))
        finally:
            self._unload_translator()
        self.master.after(0, lambda: self.update_caption_log(
            f"=== Done: {translated} translated, {skipped} skipped, {failed} failed ===\n"
        ))

    def generate_florence_caption(self, img_path):
        """Generate caption for a single image using Florence"""
        if self.florence_model is None:
            if not self.load_florence_model():
                return None

        try:
            from PIL import Image
            import torch

            image = Image.open(img_path).convert("RGB")
            task = self.caption_task_var.get()
            max_tokens = int(self.caption_max_tokens_var.get())

            inputs = self.florence_processor(
                text=task,
                images=image,
                return_tensors="pt"
            ).to(self.florence_device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.florence_model.dtype)

            generated_ids = self.florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=3,
                use_cache=False
            )

            caption = self.florence_processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]

            # Parse the output
            parsed = self.florence_processor.post_process_generation(
                caption,
                task=task,
                image_size=(image.width, image.height)
            )

            return parsed.get(task, caption)

        except Exception as e:
            self.update_caption_log(f"Error captioning {os.path.basename(img_path)}: {e}\n")
            return None

    def save_caption_with_trigger(self, img_path, caption):
        """Save caption with trigger word prepended"""
        trigger = self.caption_trigger_var.get().strip()

        if trigger:
            full_caption = f"{trigger}, {caption}"
        else:
            full_caption = caption

        caption_path = os.path.splitext(img_path)[0] + ".txt"
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(full_caption)

    def caption_all_florence(self):
        """Caption all images using Florence AI"""
        folder = self.image_folder_var.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid image folder")
            return

        images = self.get_caption_image_files()
        if not images:
            messagebox.showinfo("Info", "No images found in folder")
            return

        overwrite = self.overwrite_captions_var.get()

        # Filter images that need captioning
        if not overwrite:
            images = [img for img in images if not os.path.exists(os.path.splitext(img)[0] + ".txt")]
            if not images:
                messagebox.showinfo("Info", "All images already have captions. Enable 'Overwrite' to regenerate.")
                return

        # Run in thread
        self.captioning_stop_flag = False
        self.caption_stop_btn.configure(state=tk.NORMAL)

        def caption_thread():
            total = len(images)
            self.update_caption_log(f"Starting AI captioning of {total} images...\n")

            for i, img_path in enumerate(images):
                if self.captioning_stop_flag:
                    self.master.after(0, lambda: self.update_caption_log("Captioning stopped by user\n"))
                    break

                # Update progress
                progress = ((i + 1) / total) * 100
                self.master.after(0, lambda p=progress, c=i+1, t=total: self.update_caption_progress(p, c, t))

                caption = self.generate_florence_caption(img_path)
                if caption:
                    self.save_caption_with_trigger(img_path, caption)
                    self.master.after(0, lambda f=os.path.basename(img_path): self.update_caption_log(f"✓ {f}\n"))
                else:
                    self.master.after(0, lambda f=os.path.basename(img_path): self.update_caption_log(f"✗ {f} (failed)\n"))

            self.master.after(0, lambda: self.update_caption_log(f"\nCaptioning complete!\n"))
            self.master.after(0, lambda: self.caption_stop_btn.configure(state=tk.DISABLED))
            self.master.after(0, self.refresh_caption_images)

        threading.Thread(target=caption_thread, daemon=True).start()

    def caption_selected_florence(self):
        """Caption only selected images"""
        if not self.selected_images:
            messagebox.showinfo("Info", "No images selected. Use checkboxes to select images.")
            return

        images = list(self.selected_images)
        self.captioning_stop_flag = False
        self.caption_stop_btn.configure(state=tk.NORMAL)

        def caption_thread():
            total = len(images)
            self.update_caption_log(f"Captioning {total} selected images...\n")

            for i, img_path in enumerate(images):
                if self.captioning_stop_flag:
                    self.master.after(0, lambda: self.update_caption_log("Captioning stopped by user\n"))
                    break

                progress = ((i + 1) / total) * 100
                self.master.after(0, lambda p=progress, c=i+1, t=total: self.update_caption_progress(p, c, t))

                caption = self.generate_florence_caption(img_path)
                if caption:
                    self.save_caption_with_trigger(img_path, caption)
                    self.master.after(0, lambda f=os.path.basename(img_path): self.update_caption_log(f"✓ {f}\n"))

            self.master.after(0, lambda: self.update_caption_log(f"\nCaptioning complete!\n"))
            self.master.after(0, lambda: self.caption_stop_btn.configure(state=tk.DISABLED))
            self.master.after(0, self.refresh_caption_images)

        threading.Thread(target=caption_thread, daemon=True).start()

    def caption_single_image(self, img_path):
        """Caption a single image (for regenerate button)"""
        self.update_caption_log(f"Captioning {os.path.basename(img_path)}...\n")

        def caption_thread():
            caption = self.generate_florence_caption(img_path)
            if caption:
                self.save_caption_with_trigger(img_path, caption)
                self.master.after(0, lambda: self.update_caption_log(f"✓ Done\n"))
                self.master.after(0, self.refresh_caption_images)
            else:
                self.master.after(0, lambda: self.update_caption_log(f"✗ Failed\n"))

        threading.Thread(target=caption_thread, daemon=True).start()

    def stop_captioning(self):
        """Stop the captioning process"""
        self.captioning_stop_flag = True
        self.caption_stop_btn.configure(state=tk.DISABLED)

    def update_caption_progress(self, progress, current, total):
        """Update caption progress bar and label"""
        self.caption_progress_var.set(progress)
        self.caption_progress_label.config(text=f"{current}/{total} images")

    def _smart_text_insert(self, widget, text):
        """Insert text and only auto-scroll if user was already at the bottom.
        Preserves manual scroll position so streaming output doesn't yank the view away."""
        try:
            at_bottom = widget.yview()[1] >= 0.999
        except Exception:
            at_bottom = True
        widget.configure(state="normal")
        widget.insert(tk.END, text)
        if at_bottom:
            widget.see(tk.END)
        widget.configure(state="disabled")

    def update_caption_log(self, text):
        """Update the caption log (preserves user scroll position)."""
        self._smart_text_insert(self.caption_log, text)
        self.master.update_idletasks()

    def find_replace_in_captions(self, preview_only=False):
        """Find and replace text in all caption files (case insensitive)"""
        import re

        folder = self.image_folder_var.get()
        find_text = self.find_text_var.get()
        replace_text = self.replace_text_var.get()

        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid folder")
            return []

        if not find_text:
            messagebox.showerror("Error", "Please enter text to find")
            return []

        results = []
        txt_files = glob.glob(os.path.join(folder, "*.txt"))

        # Compile case-insensitive pattern
        pattern = re.compile(re.escape(find_text), re.IGNORECASE)

        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                if pattern.search(content):
                    new_content = pattern.sub(replace_text, content)
                    results.append({
                        'file': txt_file,
                        'old': content,
                        'new': new_content
                    })

                    if not preview_only:
                        with open(txt_file, 'w', encoding='utf-8') as f:
                            f.write(new_content)
            except Exception as e:
                self.update_caption_log(f"Error processing {txt_file}: {e}\n")

        if not preview_only:
            self.update_caption_log(f"Replaced in {len(results)} files\n")
            self.refresh_caption_images()

        return results

    def preview_find_replace(self):
        """Preview find/replace changes"""
        results = self.find_replace_in_captions(preview_only=True)

        if not results:
            messagebox.showinfo("Preview", "No matches found")
            return

        # Show preview dialog
        dialog = tk.Toplevel(self.master)
        dialog.title("Find & Replace Preview")
        dialog.geometry("700x500")
        dialog.configure(bg=BG_COLOR)

        ttk.Label(dialog, text=f"Found {len(results)} files with matches:", font=("Arial", 11, "bold")).pack(pady=10)

        # Scrollable text area
        preview_text = scrolledtext.ScrolledText(dialog, height=20, width=80, bg=ENTRY_BG, fg=FG_COLOR, wrap="word")
        preview_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        for result in results:
            filename = os.path.basename(result['file'])
            preview_text.insert(tk.END, f"\n=== {filename} ===\n")
            preview_text.insert(tk.END, f"BEFORE: {result['old'][:200]}...\n" if len(result['old']) > 200 else f"BEFORE: {result['old']}\n")
            preview_text.insert(tk.END, f"AFTER:  {result['new'][:200]}...\n" if len(result['new']) > 200 else f"AFTER:  {result['new']}\n")

        preview_text.configure(state="disabled")

        # Apply button
        def apply_changes():
            self.find_replace_in_captions(preview_only=False)
            dialog.destroy()
            messagebox.showinfo("Done", f"Replaced text in {len(results)} files")

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Apply Changes", command=apply_changes).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=10)

    def browse_caption_folder(self):
        """Browse for caption folder"""
        folder = filedialog.askdirectory()
        if folder:
            self.image_folder_var.set(folder)

    def generate_captions(self):
        """Generate caption files for all images in the selected folder"""
        folder = self.image_folder_var.get()
        caption_text = self.caption_text_var.get()
        overwrite = self.overwrite_captions_var.get()

        if not folder:
            messagebox.showerror("Error", "Please select a folder.")
            return

        if not os.path.isdir(folder):
            messagebox.showerror("Error", "Selected folder does not exist.")
            return

        if not caption_text:
            messagebox.showerror("Error", "Please enter caption text.")
            return

        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

        # Clear log
        self.caption_log.configure(state="normal")
        self.caption_log.delete(1.0, tk.END)

        created = 0
        skipped = 0
        errors = 0

        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if not os.path.isfile(filepath):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext not in image_extensions:
                continue

            # Create caption file path
            caption_path = os.path.splitext(filepath)[0] + ".txt"

            try:
                if os.path.exists(caption_path) and not overwrite:
                    self.caption_log.insert(tk.END, f"⊘ Skipped (exists): {filename}\n")
                    skipped += 1
                else:
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(caption_text)
                    self.caption_log.insert(tk.END, f"✓ Created: {os.path.basename(caption_path)}\n")
                    created += 1
            except Exception as e:
                self.caption_log.insert(tk.END, f"✗ Error ({filename}): {str(e)}\n")
                errors += 1

        # Summary
        self.caption_log.insert(tk.END, f"\n--- Summary ---\n")
        self.caption_log.insert(tk.END, f"Created: {created}\n")
        self.caption_log.insert(tk.END, f"Skipped: {skipped}\n")
        self.caption_log.insert(tk.END, f"Errors: {errors}\n")
        self.caption_log.insert(tk.END, f"Total images found: {created + skipped + errors}\n")

        self.caption_log.configure(state="disabled")
        self.caption_log.see(tk.END)

    def create_samples_settings(self):
        """Create the Samples tab with sample generation settings (Start-tab styled)."""
        scrollable_frame, _ = self.create_scrollable_frame(self.samples_tab)

        # Outer bg_deep container so the card stack sits on a consistent background.
        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer,
            "Sample Previews",
            "Preview prompts rendered periodically during training (Distilled 4-step). "
            "Samples land in <output_dir>/sample/ and the Gallery button below opens the viewer.",
        )

        # Grid holder — video warning / master checkbox / settings block all row-managed
        # so update_samples_ui_for_architecture() can still .grid() / .grid_remove() them.
        grid_holder = tk.Frame(outer, bg=COLORS["bg_deep"])
        grid_holder.pack(fill=tk.X)
        grid_holder.grid_columnconfigure(0, weight=1)

        # --- Video model warning (hidden by default; grid_remove'd at the end) ---
        self.video_model_warning_frame = ttk.Frame(grid_holder)
        self.video_model_warning_frame.grid(row=0, column=0, sticky=tk.EW, padx=36, pady=(0, 16))
        ttk.Label(
            self.video_model_warning_frame,
            text="Sample generation is not available for video models (t2v, i2v).",
            font=("Arial", 10, "italic")
        ).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(
            self.video_model_warning_frame,
            text="Video sampling during training is too slow and memory-intensive.",
            font=("Arial", 10, "italic")
        ).pack(anchor=tk.W, pady=(0, 15))
        video_viewer_frame = ttk.Frame(self.video_model_warning_frame)
        video_viewer_frame.pack(anchor=tk.W, pady=10)
        ttk.Button(video_viewer_frame, text="View Samples Gallery", command=self.open_samples_gallery).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_viewer_frame, text="Open Samples Folder", command=self.open_samples_folder).pack(side=tk.LEFT, padx=5)
        self.video_model_warning_frame.grid_remove()

        # --- Master Enable card ---
        self.sample_enabled_var = tk.BooleanVar(value=self.settings["SAMPLE_ENABLED"])
        enable_card_outer = tk.Frame(grid_holder, bg=COLORS["bg_deep"])
        enable_card_outer.grid(row=1, column=0, sticky=tk.EW, padx=36, pady=(0, 16))
        enable_card = tk.Frame(enable_card_outer, bg=COLORS["bg_surface"],
                               highlightbackground=COLORS["border"], highlightthickness=1, bd=0)
        enable_card.pack(fill=tk.X)
        # Use ttk.Checkbutton — themed; sits on bg_surface via style inheritance
        self.sample_enabled_check = ttk.Checkbutton(
            enable_card, text="Enable Sample Generation", variable=self.sample_enabled_var,
            command=self.toggle_sample_settings,
        )
        self.sample_enabled_check.pack(anchor=tk.W, padx=20, pady=14)

        # --- Sample settings container (the 4 cards live inside this) ---
        self.sample_settings_frame = tk.Frame(grid_holder, bg=COLORS["bg_deep"])
        self.sample_settings_frame.grid(row=2, column=0, sticky=tk.EW)

        # Card 1: Prompt & Dimensions
        prompt_card = self._start_section_card(
            self.sample_settings_frame, "Prompt & Dimensions",
            "Multi-line prompt, output size, steps and seed for each preview render.",
        )
        prompt_card.grid_columnconfigure(1, weight=1)

        ttk.Label(prompt_card, text="Prompt:").grid(row=0, column=0, sticky=tk.NW, padx=(0, 10), pady=4)
        self.sample_prompt_text = tk.Text(
            prompt_card, height=3, width=50, bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"], font=(FONT_FAMILY, 10), wrap="word",
            relief="flat", highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["border_focus"],
        )
        self.sample_prompt_text.insert("1.0", self.last_used.get("sample_prompt", self.settings["SAMPLE_PROMPT"]))
        self.sample_prompt_text.grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=4)
        self.sample_prompt_text.bind("<KeyRelease>", lambda e: self._save_last_used_paths())

        ttk.Label(prompt_card, text="Width:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_width_var = tk.StringVar(value=str(self.settings["SAMPLE_WIDTH"]))
        self.sample_width_combo = ttk.Combobox(
            prompt_card, textvariable=self.sample_width_var,
            values=["512", "768", "1024", "1280"], state="readonly", width=10,
        )
        self.sample_width_combo.grid(row=1, column=1, sticky=tk.W, pady=4)

        ttk.Label(prompt_card, text="Height:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_height_var = tk.StringVar(value=str(self.settings["SAMPLE_HEIGHT"]))
        self.sample_height_combo = ttk.Combobox(
            prompt_card, textvariable=self.sample_height_var,
            values=["512", "768", "1024", "1280"], state="readonly", width=10,
        )
        self.sample_height_combo.grid(row=2, column=1, sticky=tk.W, pady=4)

        ttk.Label(prompt_card, text="Steps:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_steps_var = tk.StringVar(value=str(self.settings["SAMPLE_STEPS"]))
        self.sample_steps_entry = ttk.Entry(prompt_card, textvariable=self.sample_steps_var, width=10)
        self.sample_steps_entry.grid(row=3, column=1, sticky=tk.W, pady=4)

        ttk.Label(prompt_card, text="Seed:").grid(row=4, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_seed_var = tk.StringVar(value=str(self.settings["SAMPLE_SEED"]))
        self.sample_seed_entry = ttk.Entry(prompt_card, textvariable=self.sample_seed_var, width=10)
        self.sample_seed_entry.grid(row=4, column=1, sticky=tk.W, pady=4)
        tk.Label(prompt_card, text="(0 = random)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=4, column=2, sticky=tk.W, padx=(10, 0)
        )

        # Card 2: Generation Frequency
        freq_card = self._start_section_card(
            self.sample_settings_frame, "Generation Frequency",
            "How often preview renders fire during training. Set either value to 0 to disable that cadence.",
        )
        freq_card.grid_columnconfigure(1, weight=1)

        ttk.Label(freq_card, text="Every N Epochs:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_every_n_epochs_var = tk.StringVar(value=str(self.settings["SAMPLE_EVERY_N_EPOCHS"]))
        self.sample_every_n_epochs_entry = ttk.Entry(freq_card, textvariable=self.sample_every_n_epochs_var, width=10)
        self.sample_every_n_epochs_entry.grid(row=0, column=1, sticky=tk.W, pady=4)

        ttk.Label(freq_card, text="Every N Steps:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_every_n_steps_var = tk.StringVar(value=str(self.settings["SAMPLE_EVERY_N_STEPS"]))
        self.sample_every_n_steps_entry = ttk.Entry(freq_card, textvariable=self.sample_every_n_steps_var, width=10)
        self.sample_every_n_steps_entry.grid(row=1, column=1, sticky=tk.W, pady=4)
        tk.Label(freq_card, text="(0 = disabled)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=1, column=2, sticky=tk.W, padx=(10, 0)
        )

        self.sample_at_first_var = tk.BooleanVar(value=self.settings["SAMPLE_AT_FIRST"])
        ttk.Checkbutton(
            freq_card, text="Sample at Start", variable=self.sample_at_first_var
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))

        # Card 3: Architecture-Specific (Flow Shift / Guidance / Negative / CFG)
        arch_card = self._start_section_card(
            self.sample_settings_frame, "Advanced",
            "Architecture-specific knobs. Distilled models disable Negative Prompt; "
            "non-distilled models disable CFG Scale.",
        )
        arch_card.grid_columnconfigure(1, weight=1)

        self.sample_flow_shift_label = ttk.Label(arch_card, text="Flow Shift:")
        self.sample_flow_shift_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_flow_shift_var = tk.StringVar(value=str(self.settings["SAMPLE_FLOW_SHIFT"]))
        self.sample_flow_shift_entry = ttk.Entry(arch_card, textvariable=self.sample_flow_shift_var, width=10)
        self.sample_flow_shift_entry.grid(row=0, column=1, sticky=tk.W, pady=4)
        self.sample_flow_shift_row = 0

        self.sample_guidance_label = ttk.Label(arch_card, text="Guidance Scale:")
        self.sample_guidance_label.grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_guidance_var = tk.StringVar(value=str(self.settings["SAMPLE_GUIDANCE"]))
        self.sample_guidance_entry = ttk.Entry(arch_card, textvariable=self.sample_guidance_var, width=10)
        self.sample_guidance_entry.grid(row=1, column=1, sticky=tk.W, pady=4)
        self.sample_guidance_row = 1

        self.sample_negative_label = ttk.Label(arch_card, text="Negative Prompt:")
        self.sample_negative_label.grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_negative_var = tk.StringVar(value=self.settings["SAMPLE_NEGATIVE"])
        self.sample_negative_entry = ttk.Entry(arch_card, textvariable=self.sample_negative_var, width=50)
        self.sample_negative_entry.grid(row=2, column=1, columnspan=2, sticky=tk.EW, pady=4)
        self.sample_negative_row = 2

        self.sample_cfg_label = ttk.Label(arch_card, text="CFG Scale:")
        self.sample_cfg_label.grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.sample_cfg_scale_var = tk.StringVar(value=str(self.settings["SAMPLE_CFG_SCALE"]))
        self.sample_cfg_scale_entry = ttk.Entry(arch_card, textvariable=self.sample_cfg_scale_var, width=10)
        self.sample_cfg_scale_entry.grid(row=3, column=1, sticky=tk.W, pady=4)
        self.sample_cfg_row = 3

        # Card 4: Viewer
        viewer_card = self._start_section_card(
            self.sample_settings_frame, "Viewer",
            "Browse generated samples without leaving the app, or open the folder in Explorer.",
        )

        viewer_buttons = tk.Frame(viewer_card, bg=COLORS["bg_surface"])
        viewer_buttons.pack(anchor=tk.W)
        ttk.Button(viewer_buttons, text="View Samples Gallery", command=self.open_samples_gallery).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(viewer_buttons, text="Open Samples Folder", command=self.open_samples_folder).pack(side=tk.LEFT)

        self.sample_output_label = tk.Label(
            viewer_card, text="Sample output: <output_dir>/sample/",
            font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"],
        )
        self.sample_output_label.pack(anchor=tk.W, pady=(10, 0))

        # Store entries for saving/loading
        self.entries["SAMPLE_ENABLED"] = self.sample_enabled_var
        self.entries["SAMPLE_WIDTH"] = self.sample_width_combo
        self.entries["SAMPLE_HEIGHT"] = self.sample_height_combo
        self.entries["SAMPLE_STEPS"] = self.sample_steps_entry
        self.entries["SAMPLE_SEED"] = self.sample_seed_entry
        self.entries["SAMPLE_EVERY_N_EPOCHS"] = self.sample_every_n_epochs_entry
        self.entries["SAMPLE_EVERY_N_STEPS"] = self.sample_every_n_steps_entry
        self.entries["SAMPLE_AT_FIRST"] = self.sample_at_first_var
        self.entries["SAMPLE_FLOW_SHIFT"] = self.sample_flow_shift_entry
        self.entries["SAMPLE_GUIDANCE"] = self.sample_guidance_entry
        self.entries["SAMPLE_NEGATIVE"] = self.sample_negative_entry
        self.entries["SAMPLE_CFG_SCALE"] = self.sample_cfg_scale_entry

        # Initial UI state based on current architecture
        self.update_samples_ui_for_architecture()

        self._add_youtube_help_button(outer, "samples")

    def toggle_sample_settings(self):
        """Enable or disable sample settings based on the enable checkbox"""
        state = tk.NORMAL if self.sample_enabled_var.get() else tk.DISABLED

        def _apply(widget):
            try:
                if isinstance(widget, tk.Text):
                    widget.configure(state=state if state == tk.NORMAL else tk.DISABLED)
                else:
                    widget.configure(state=state)
            except tk.TclError:
                pass  # Some widgets don't support state

        def _walk(parent):
            for child in parent.winfo_children():
                _apply(child)
                if child.winfo_children():
                    _walk(child)

        _walk(self.sample_settings_frame)

    def update_samples_ui_for_architecture(self):
        """Update samples tab UI based on selected architecture"""
        arch = self.architecture_var.get()
        config = ARCHITECTURES.get(arch, ARCHITECTURES["Flux 2 Klein Base 9B"])

        supports_samples = config.get("supports_samples", False)

        if not supports_samples:
            # Show warning, hide settings
            self.video_model_warning_frame.grid()
            self.sample_enabled_check.grid_remove()
            self.sample_settings_frame.grid_remove()
        else:
            # Hide warning, show settings
            self.video_model_warning_frame.grid_remove()
            self.sample_enabled_check.grid()
            self.sample_settings_frame.grid()

            # Update default values for this architecture
            if config.get("sample_guidance_default") is not None:
                self.sample_guidance_var.set(str(config["sample_guidance_default"]))
            if config.get("sample_cfg_default") is not None:
                self.sample_cfg_scale_var.set(str(config["sample_cfg_default"]))
            if config.get("sample_flow_shift_default") is not None:
                self.sample_flow_shift_var.set(str(config["sample_flow_shift_default"]))
            if config.get("sample_steps_default") is not None:
                self.sample_steps_var.set(str(config["sample_steps_default"]))
            if config.get("sample_width_default") is not None:
                self.sample_width_var.set(str(config["sample_width_default"]))
            if config.get("sample_height_default") is not None:
                self.sample_height_var.set(str(config["sample_height_default"]))

            # Enable/disable flow shift based on architecture
            if config.get("sample_flow_shift_default") is None:
                self.sample_flow_shift_entry.configure(state=tk.DISABLED)
                self.sample_flow_shift_label.configure(foreground="gray")
            else:
                self.sample_flow_shift_entry.configure(state=tk.NORMAL)
                self.sample_flow_shift_label.configure(foreground=FG_COLOR)

            # Handle distilled models (no negative prompts)
            if config.get("sample_is_distilled", False):
                self.sample_negative_entry.configure(state=tk.DISABLED)
                self.sample_negative_label.configure(foreground="gray")
            else:
                self.sample_negative_entry.configure(state=tk.NORMAL)
                self.sample_negative_label.configure(foreground=FG_COLOR)

            # Handle fixed steps/cfg for distilled models
            if config.get("sample_steps_fixed", False):
                self.sample_steps_entry.configure(state=tk.DISABLED)
            else:
                self.sample_steps_entry.configure(state=tk.NORMAL)

            if config.get("sample_cfg_fixed", False):
                self.sample_cfg_scale_entry.configure(state=tk.DISABLED)
            else:
                self.sample_cfg_scale_entry.configure(state=tk.NORMAL)

        # Update sample output path label
        self.update_sample_output_label()

    def update_sample_output_label(self):
        """Update the sample output path label to show actual path"""
        samples_dir = self.get_samples_dir()
        self.sample_output_label.config(text=f"Sample Output: {samples_dir}")

    def generate_sample_prompt_file(self):
        """Generate prompt file for sample generation"""
        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)

        prompt_file = os.path.join(samples_dir, "prompts.txt")

        # Build prompt with options
        prompt = self.sample_prompt_text.get("1.0", tk.END).strip()

        # Add options if not already in prompt
        if "--w" not in prompt:
            prompt += f" --w {self.sample_width_var.get()}"
        if "--h" not in prompt:
            prompt += f" --h {self.sample_height_var.get()}"
        if "--f" not in prompt:
            prompt += " --f 1"  # Always 1 for images
        if "--s" not in prompt:
            prompt += f" --s {self.sample_steps_var.get()}"

        seed = self.sample_seed_var.get()
        if seed and seed != "0" and "--d" not in prompt:
            prompt += f" --d {seed}"

        # Add flow shift if set
        flow_shift = self.sample_flow_shift_var.get()
        if flow_shift and "--fs" not in prompt:
            prompt += f" --fs {flow_shift}"

        # Add guidance scale
        guidance = self.sample_guidance_var.get()
        if guidance and "--g" not in prompt:
            prompt += f" --g {guidance}"

        # Add negative prompt if set
        negative = self.sample_negative_var.get().strip()
        if negative and "--n" not in prompt:
            prompt += f" --n {negative}"

        # Always add CFG scale (omitting --l causes fallback to 4.0 in Z-Image)
        cfg = self.sample_cfg_scale_var.get()
        if cfg and "--l" not in prompt:
            prompt += f" --l {cfg}"

        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"# Auto-generated by Fizgig LoRA Trainer GUI\n{prompt}\n")

        return prompt_file

    def get_samples_dir(self):
        """Get the samples directory from output dir"""
        output_dir = self.settings.get("LORA_OUTPUT_DIR", "")
        if output_dir:
            return os.path.join(output_dir, "sample")
        # Fallback to local samples folder
        return os.path.join(os.path.dirname(__file__), "output_loras", "sample")

    def find_free_port(self):
        """Find an available port for the HTTP server"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def start_gallery_server(self):
        """Start HTTP server to serve samples directory (avoids CORS issues)"""
        if self.gallery_server is not None:
            return  # Already running

        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)

        # Find free port
        self.gallery_server_port = self.find_free_port()

        # Create handler that serves from samples directory
        class SamplesHandler(SimpleHTTPRequestHandler):
            def __init__(handler_self, *args, **kwargs):
                super().__init__(*args, directory=samples_dir, **kwargs)

            def log_message(handler_self, format, *args):
                pass  # Suppress logging

        try:
            self.gallery_server = HTTPServer(('127.0.0.1', self.gallery_server_port), SamplesHandler)

            # Run server in background thread
            def serve_forever():
                self.gallery_server.serve_forever()

            self.gallery_server_thread = threading.Thread(target=serve_forever, daemon=True)
            self.gallery_server_thread.start()

        except Exception as e:
            print(f"Failed to start gallery server: {e}")
            self.gallery_server = None
            self.gallery_server_port = None

    def stop_gallery_server(self):
        """Stop the HTTP server"""
        if self.gallery_server is not None:
            self.gallery_server.shutdown()
            self.gallery_server = None
            self.gallery_server_port = None

    def open_samples_gallery(self):
        """Open the samples gallery HTML viewer in browser via HTTP server"""
        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)

        gallery_path = os.path.join(samples_dir, "gallery.html")

        # Create gallery.html if it doesn't exist
        if not os.path.exists(gallery_path):
            self.create_gallery_html(gallery_path)

        # Generate/update the gallery HTML with current files
        self.update_gallery_html()

        # Start HTTP server if not running
        self.start_gallery_server()

        if self.gallery_server_port:
            # Open via HTTP (avoids CORS issues)
            webbrowser.open(f"http://127.0.0.1:{self.gallery_server_port}/gallery.html")
        else:
            # Fallback to file:// if server failed
            webbrowser.open(f"file://{os.path.abspath(gallery_path)}")

    def open_samples_folder(self):
        """Open the samples folder in file explorer"""
        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)
        self._open_in_file_manager(samples_dir)

    def create_gallery_html(self, gallery_path):
        """Create the gallery HTML file if it doesn't exist"""
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample Gallery - Fizgig</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1B2A38; color: #ECF0F1; min-height: 100vh; }
        header { background-color: #2C3E50; padding: 20px; border-bottom: 2px solid #2980B9; position: sticky; top: 0; z-index: 100; }
        header h1 { color: #ECF0F1; font-size: 24px; margin-bottom: 15px; display: flex; align-items: center; gap: 15px; }
        .live-indicator { width: 12px; height: 12px; background-color: #27AE60; border-radius: 50%; animation: pulse 2s infinite; }
        .live-indicator.paused { background-color: #95A5A6; animation: none; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .controls { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
        .controls label { display: flex; align-items: center; gap: 8px; }
        .controls select { padding: 5px 8px; border: 1px solid #2980B9; border-radius: 4px; background-color: #1B2A38; color: #ECF0F1; }
        .controls button { padding: 8px 16px; background-color: #2980B9; color: #ECF0F1; border: none; border-radius: 4px; cursor: pointer; }
        .controls button:hover { background-color: #3498DB; }
        .status { color: #95A5A6; font-size: 14px; }
        main { padding: 20px; }
        #gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .gallery-item { background-color: #2C3E50; border-radius: 8px; overflow: hidden; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }
        .gallery-item:hover { transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); }
        .gallery-item.new { animation: highlight 2s ease-out; }
        @keyframes highlight { 0%, 30% { box-shadow: 0 0 30px #27AE60; } 100% { box-shadow: none; } }
        .image-container { position: relative; }
        .gallery-item img { width: 100%; height: 280px; object-fit: cover; display: block; background-color: #1B2A38; }
        .badge { position: absolute; top: 10px; left: 10px; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
        .epoch-badge { background-color: #27AE60; color: white; }
        .new-badge { position: absolute; top: 10px; right: 10px; background-color: #E74C3C; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
        .image-info { padding: 12px; }
        .lora-name { color: #9B59B6; font-weight: 600; font-size: 14px; margin-bottom: 6px; }
        .meta-row { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
        .meta-item { font-size: 13px; color: #BDC3C7; }
        .meta-item.seed { color: #3498DB; font-family: monospace; }
        .meta-item.time { color: #95A5A6; }
        .no-images { grid-column: 1 / -1; text-align: center; padding: 60px 20px; color: #95A5A6; }
        .no-images h2 { margin-bottom: 15px; color: #ECF0F1; }
        .stats { background-color: #2C3E50; padding: 8px 15px; border-radius: 4px; font-size: 14px; }
        #lightbox { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.95); z-index: 1000; justify-content: center; align-items: center; flex-direction: column; }
        #lightbox.active { display: flex; }
        #lightbox img { max-width: 90%; max-height: 80%; object-fit: contain; }
        #lightbox .close-btn { position: absolute; top: 20px; right: 30px; font-size: 40px; color: #ECF0F1; cursor: pointer; }
        #lightbox .close-btn:hover { color: #E74C3C; }
        #lightbox .nav-btn { position: absolute; top: 50%; transform: translateY(-50%); font-size: 50px; color: #ECF0F1; cursor: pointer; padding: 20px; user-select: none; }
        #lightbox .nav-btn:hover { color: #2980B9; }
        #lightbox .prev-btn { left: 20px; }
        #lightbox .next-btn { right: 20px; }
        #lightbox .image-details { margin-top: 15px; text-align: center; }
        #lightbox .image-name { color: #ECF0F1; font-size: 16px; }
        #lightbox .image-meta { color: #95A5A6; font-size: 14px; margin-top: 5px; }
    </style>
</head>
<body>
    <header>
        <h1><span class="live-indicator" id="live-dot"></span> Fizgig Sample Gallery</h1>
        <div class="controls">
            <label>Sort: <select id="sort-select">
                <option value="newest">Newest First</option>
                <option value="oldest">Oldest First</option>
                <option value="epoch-desc">Epoch (High-Low)</option>
                <option value="epoch-asc">Epoch (Low-High)</option>
            </select></label>
            <label>Refresh: <select id="refresh-select">
                <option value="3">3 sec</option>
                <option value="5">5 sec</option>
                <option value="10" selected>10 sec</option>
                <option value="30">30 sec</option>
                <option value="0">Off</option>
            </select></label>
            <button onclick="loadImages()">Refresh Now</button>
            <span class="stats" id="stats">0 images</span>
            <span class="status" id="status">Ready</span>
        </div>
    </header>
    <main>
        <div id="gallery">
            <div class="no-images">
                <h2>Loading...</h2>
            </div>
        </div>
    </main>
    <div id="lightbox">
        <span class="close-btn" onclick="closeLightbox()">&times;</span>
        <span class="nav-btn prev-btn" onclick="navigateLightbox(-1)">&#10094;</span>
        <img id="lightbox-img" src="" alt="">
        <span class="nav-btn next-btn" onclick="navigateLightbox(1)">&#10095;</span>
        <div class="image-details">
            <div class="image-name" id="lightbox-name"></div>
            <div class="image-meta" id="lightbox-meta"></div>
        </div>
    </div>
    <!-- EMBEDDED_FILES_START -->
    <script id="files-data" type="application/json">[]</script>
    <!-- EMBEDDED_FILES_END -->
    <script>
        let images = [];
        let currentLightboxIndex = 0;
        let refreshTimer = null;

        document.getElementById('sort-select').value = localStorage.getItem('fizgig-sort') || 'newest';
        document.getElementById('refresh-select').value = localStorage.getItem('fizgig-refresh') || '10';

        document.getElementById('sort-select').addEventListener('change', (e) => {
            localStorage.setItem('fizgig-sort', e.target.value);
            renderGallery();
        });

        document.getElementById('refresh-select').addEventListener('change', (e) => {
            localStorage.setItem('fizgig-refresh', e.target.value);
            setupTimer();
        });

        function setupTimer() {
            if (refreshTimer) clearInterval(refreshTimer);
            const sec = parseInt(document.getElementById('refresh-select').value);
            const dot = document.getElementById('live-dot');
            if (sec > 0) {
                refreshTimer = setInterval(loadImages, sec * 1000);
                dot.classList.remove('paused');
            } else {
                dot.classList.add('paused');
            }
        }

        function parseFilename(filename) {
            const match = filename.match(/^(.+)_e(\\d{6})_(\\d{2})_(\\d{14})_(\\d+)\\.png$/i);
            if (match) {
                const ts = match[4];
                return {
                    filename,
                    loraName: match[1],
                    epoch: parseInt(match[2]),
                    idx: parseInt(match[3]),
                    timestamp: ts,
                    seed: match[5],
                    time: `${ts.slice(8,10)}:${ts.slice(10,12)}:${ts.slice(12,14)}`
                };
            }
            return { filename, loraName: 'Unknown', epoch: 0, idx: 0, timestamp: '', seed: '', time: '' };
        }

        async function loadImages() {
            document.getElementById('status').textContent = 'Loading...';

            // Try fetch first (works with HTTP server), fallback to embedded
            try {
                const resp = await fetch('files.json?t=' + Date.now());
                if (resp.ok) {
                    const files = await resp.json();
                    images = files.map(f => parseFilename(f));
                    renderGallery();
                    document.getElementById('stats').textContent = `${images.length} image${images.length !== 1 ? 's' : ''}`;
                    document.getElementById('status').textContent = `Updated: ${new Date().toLocaleTimeString()}`;
                    return;
                }
            } catch (e) {
                // Fetch failed, try embedded data
            }

            // Fallback to embedded data
            const filesData = document.getElementById('files-data');
            if (filesData) {
                try {
                    const files = JSON.parse(filesData.textContent);
                    images = files.map(f => parseFilename(f));
                    renderGallery();
                    document.getElementById('stats').textContent = `${images.length} image${images.length !== 1 ? 's' : ''}`;
                    document.getElementById('status').textContent = `Loaded: ${new Date().toLocaleTimeString()}`;
                } catch (e) {
                    document.getElementById('status').textContent = 'Error loading files';
                }
            }
        }

        function renderGallery() {
            const gallery = document.getElementById('gallery');
            const sortBy = document.getElementById('sort-select').value;

            if (images.length === 0) {
                gallery.innerHTML = `<div class="no-images"><h2>No samples yet</h2><p>Start training with sample generation enabled.</p></div>`;
                return;
            }

            let sorted = [...images];
            switch (sortBy) {
                case 'newest': sorted.sort((a, b) => b.timestamp.localeCompare(a.timestamp)); break;
                case 'oldest': sorted.sort((a, b) => a.timestamp.localeCompare(b.timestamp)); break;
                case 'epoch-desc': sorted.sort((a, b) => b.epoch - a.epoch || b.timestamp.localeCompare(a.timestamp)); break;
                case 'epoch-asc': sorted.sort((a, b) => a.epoch - b.epoch || a.timestamp.localeCompare(b.timestamp)); break;
            }

            gallery.innerHTML = sorted.map(img => `
                <div class="gallery-item" onclick="openLightbox('${img.filename}')">
                    <div class="image-container">
                        <img src="${img.filename}" alt="${img.filename}" loading="lazy">
                        <span class="badge epoch-badge">Epoch ${img.epoch}</span>
                    </div>
                    <div class="image-info">
                        <div class="lora-name">${img.loraName}</div>
                        <div class="meta-row">
                            <span class="meta-item seed">Seed: ${img.seed}</span>
                            <span class="meta-item time">${img.time}</span>
                        </div>
                    </div>
                </div>`).join('');
        }

        function openLightbox(filename) {
            const idx = images.findIndex(img => img.filename === filename);
            if (idx >= 0) { currentLightboxIndex = idx; showLightbox(images[idx]); }
        }

        function showLightbox(img) {
            document.getElementById('lightbox-img').src = img.filename;
            document.getElementById('lightbox-name').textContent = img.filename;
            document.getElementById('lightbox-meta').textContent = `${img.loraName} | Epoch ${img.epoch} | Seed: ${img.seed} | ${img.time}`;
            document.getElementById('lightbox').classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function closeLightbox() {
            document.getElementById('lightbox').classList.remove('active');
            document.body.style.overflow = '';
        }

        function navigateLightbox(dir) {
            if (images.length === 0) return;
            currentLightboxIndex = (currentLightboxIndex + dir + images.length) % images.length;
            showLightbox(images[currentLightboxIndex]);
        }

        document.addEventListener('keydown', (e) => {
            if (!document.getElementById('lightbox').classList.contains('active')) return;
            if (e.key === 'Escape') closeLightbox();
            if (e.key === 'ArrowLeft') navigateLightbox(-1);
            if (e.key === 'ArrowRight') navigateLightbox(1);
        });

        document.getElementById('lightbox').addEventListener('click', (e) => {
            if (e.target.id === 'lightbox') closeLightbox();
        });

        setupTimer();
        loadImages();
    </script>
</body>
</html>'''

        with open(gallery_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def update_gallery_html(self):
        """Update gallery HTML and files.json for live updates"""
        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)

        # Find all images
        images = []
        if os.path.exists(samples_dir):
            for f in os.listdir(samples_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    images.append(f)

        # Sort alphabetically (which sorts by epoch due to naming convention)
        images.sort()

        # Write files.json for HTTP fetch (live updates)
        files_json_path = os.path.join(samples_dir, "files.json")
        try:
            with open(files_json_path, 'w', encoding='utf-8') as f:
                json.dump(images, f)
        except Exception:
            pass

        # Also update embedded data in gallery.html (for fallback)
        gallery_path = os.path.join(samples_dir, "gallery.html")
        if os.path.exists(gallery_path):
            try:
                with open(gallery_path, 'r', encoding='utf-8') as f:
                    html = f.read()

                # Find and replace the embedded JSON
                import re
                new_json = json.dumps(images)
                # Replace content between the script tags
                pattern = r'(<script id="files-data" type="application/json">).*?(</script>)'
                replacement = rf'\1{new_json}\2'
                new_html = re.sub(pattern, replacement, html, flags=re.DOTALL)

                if new_html != html:
                    with open(gallery_path, 'w', encoding='utf-8') as f:
                        f.write(new_html)
            except Exception:
                pass  # Don't fail if gallery update fails

    def start_samples_watcher(self):
        """Start background thread to update files.json for live gallery"""
        if self.samples_watcher_running:
            return

        self.samples_watcher_running = True

        def watcher_loop():
            while self.samples_watcher_running:
                try:
                    self.update_gallery_html()
                except Exception:
                    pass
                # Update every 5 seconds
                for _ in range(50):  # 5 seconds in 0.1s increments
                    if not self.samples_watcher_running:
                        break
                    time.sleep(0.1)

        self.samples_watcher_thread = threading.Thread(target=watcher_loop, daemon=True)
        self.samples_watcher_thread.start()

    def stop_samples_watcher(self):
        """Stop the samples watcher thread"""
        self.samples_watcher_running = False

    def parse_sample_filename(self, filename):
        """Parse epoch, step, seed from sample filename"""
        import re
        info = {"epoch": None, "step": None, "seed": None, "prompt_idx": None}

        # Pattern: {name}_e{epoch:06d}_{promptIdx}_{timestamp}_{seed}.png
        # or: {name}_{step:06d}_{promptIdx}_{timestamp}_{seed}.png
        epoch_match = re.search(r'_e(\d{6})_', filename)
        if epoch_match:
            info["epoch"] = int(epoch_match.group(1))

        step_match = re.search(r'_(\d{6})_(\d{2})_\d{14}', filename)
        if step_match and not epoch_match:
            info["step"] = int(step_match.group(1))
            info["prompt_idx"] = int(step_match.group(2))
        elif epoch_match:
            prompt_match = re.search(r'_e\d{6}_(\d{2})_', filename)
            if prompt_match:
                info["prompt_idx"] = int(prompt_match.group(1))

        seed_match = re.search(r'_(\d+)\.\w+$', filename)
        if seed_match:
            info["seed"] = int(seed_match.group(1))

        return info

    def generate_gallery_html(self, images):
        """Generate the gallery HTML content"""
        import time

        # Parse image info and build items
        image_data = []
        for filename, mtime in images:
            info = self.parse_sample_filename(filename)
            info["filename"] = filename
            info["mtime"] = mtime
            info["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
            image_data.append(info)

        image_items = ""
        for img in image_data:
            filename = img["filename"]
            timestamp = img["timestamp"]

            # Build epoch/step badge
            if img["epoch"] is not None:
                badge = f'<span class="epoch-badge">Epoch {img["epoch"]}</span>'
            elif img["step"] is not None:
                badge = f'<span class="step-badge">Step {img["step"]}</span>'
            else:
                badge = ''

            # Build seed info
            seed_info = f'<span class="seed">Seed: {img["seed"]}</span>' if img["seed"] else ''

            image_items += f'''
            <div class="gallery-item" onclick="openLightbox('{filename}')">
                <div class="image-container">
                    <img src="{filename}" alt="{filename}" loading="lazy">
                    {badge}
                </div>
                <div class="image-info">
                    <span class="filename">{filename}</span>
                    <div class="meta-row">
                        <span class="timestamp">{timestamp}</span>
                        {seed_info}
                    </div>
                </div>
            </div>'''

        if not image_items:
            image_items = '<div class="no-images">No sample images found yet. Start training to generate samples.</div>'

        # Build image data for JavaScript
        js_image_data = []
        for img in image_data:
            js_image_data.append({
                "filename": img["filename"],
                "epoch": img["epoch"],
                "step": img["step"],
                "seed": img["seed"]
            })

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample Gallery - Fizgig</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1B2A38; color: #ECF0F1; min-height: 100vh; }}
        header {{ background-color: #2C3E50; padding: 20px; border-bottom: 2px solid #2980B9; position: sticky; top: 0; z-index: 100; }}
        header h1 {{ color: #ECF0F1; font-size: 24px; margin-bottom: 15px; }}
        .controls {{ display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
        .controls label {{ display: flex; align-items: center; gap: 8px; }}
        .controls input[type="number"], .controls select {{ padding: 5px 8px; border: 1px solid #2980B9; border-radius: 4px; background-color: #1B2A38; color: #ECF0F1; }}
        .controls button {{ padding: 8px 16px; background-color: #2980B9; color: #ECF0F1; border: none; border-radius: 4px; cursor: pointer; }}
        .controls button:hover {{ background-color: #3498DB; }}
        #last-update {{ color: #95A5A6; font-size: 14px; }}
        main {{ padding: 20px; }}
        #gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .gallery-item {{ background-color: #2C3E50; border-radius: 8px; overflow: hidden; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }}
        .gallery-item:hover {{ transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); }}
        .image-container {{ position: relative; }}
        .gallery-item img {{ width: 100%; height: 250px; object-fit: cover; display: block; }}
        .epoch-badge, .step-badge {{ position: absolute; top: 10px; left: 10px; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
        .epoch-badge {{ background-color: #27AE60; color: white; }}
        .step-badge {{ background-color: #E67E22; color: white; }}
        .image-info {{ padding: 12px; display: flex; flex-direction: column; gap: 6px; }}
        .filename {{ font-weight: 500; font-size: 13px; word-break: break-all; color: #BDC3C7; }}
        .meta-row {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
        .timestamp {{ color: #95A5A6; font-size: 12px; }}
        .seed {{ color: #3498DB; font-size: 12px; font-family: monospace; }}
        .no-images {{ grid-column: 1 / -1; text-align: center; padding: 60px 20px; color: #95A5A6; font-size: 18px; }}
        #lightbox {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.95); z-index: 1000; justify-content: center; align-items: center; flex-direction: column; }}
        #lightbox.active {{ display: flex; }}
        #lightbox img {{ max-width: 90%; max-height: 80%; object-fit: contain; }}
        #lightbox .close-btn {{ position: absolute; top: 20px; right: 30px; font-size: 40px; color: #ECF0F1; cursor: pointer; }}
        #lightbox .close-btn:hover {{ color: #E74C3C; }}
        #lightbox .nav-btn {{ position: absolute; top: 50%; transform: translateY(-50%); font-size: 50px; color: #ECF0F1; cursor: pointer; padding: 20px; user-select: none; }}
        #lightbox .nav-btn:hover {{ color: #2980B9; }}
        #lightbox .prev-btn {{ left: 20px; }}
        #lightbox .next-btn {{ right: 20px; }}
        #lightbox .image-details {{ margin-top: 15px; text-align: center; }}
        #lightbox .image-name {{ color: #ECF0F1; font-size: 16px; }}
        #lightbox .image-meta {{ color: #95A5A6; font-size: 14px; margin-top: 5px; }}
    </style>
</head>
<body>
    <header>
        <h1>Fizgig Sample Gallery</h1>
        <div class="controls">
            <label>Sort by: <select id="sort-select" onchange="sortGallery()">
                <option value="newest">Newest First</option>
                <option value="oldest">Oldest First</option>
                <option value="epoch-desc">Epoch (High to Low)</option>
                <option value="epoch-asc">Epoch (Low to High)</option>
            </select></label>
            <label>Auto-refresh: <input type="number" id="refresh-interval" value="30" min="5" max="300"> sec</label>
            <button onclick="refreshGallery()">Refresh Now</button>
            <span id="last-update"></span>
        </div>
    </header>
    <main><div id="gallery">{image_items}</div></main>
    <div id="lightbox">
        <span class="close-btn" onclick="closeLightbox()">&times;</span>
        <span class="nav-btn prev-btn" onclick="navigateLightbox(-1)">&#10094;</span>
        <img id="lightbox-img" src="" alt="">
        <span class="nav-btn next-btn" onclick="navigateLightbox(1)">&#10095;</span>
        <div class="image-details">
            <div class="image-name" id="lightbox-name"></div>
            <div class="image-meta" id="lightbox-meta"></div>
        </div>
    </div>
    <script>
        const imageData = {json.dumps(js_image_data)};
        let refreshInterval = localStorage.getItem('fizgig-refresh') || 30;
        let refreshTimer = null;
        let currentImageIndex = 0;
        const images = imageData.map(d => d.filename);
        document.getElementById('refresh-interval').value = refreshInterval;
        updateLastRefresh();
        startRefreshTimer();
        const savedSort = localStorage.getItem('fizgig-sort') || 'newest';
        document.getElementById('sort-select').value = savedSort;
        document.getElementById('refresh-interval').addEventListener('change', (e) => {{
            refreshInterval = Math.max(5, Math.min(300, parseInt(e.target.value) || 30));
            e.target.value = refreshInterval;
            localStorage.setItem('fizgig-refresh', refreshInterval);
            startRefreshTimer();
        }});
        function startRefreshTimer() {{ if (refreshTimer) clearInterval(refreshTimer); refreshTimer = setInterval(refreshGallery, refreshInterval * 1000); }}
        function refreshGallery() {{ location.reload(); }}
        function updateLastRefresh() {{ document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString(); }}
        function sortGallery() {{
            const sortBy = document.getElementById('sort-select').value;
            localStorage.setItem('fizgig-sort', sortBy);
            const gallery = document.getElementById('gallery');
            const items = Array.from(gallery.querySelectorAll('.gallery-item'));
            items.sort((a, b) => {{
                const aFile = a.querySelector('img').alt;
                const bFile = b.querySelector('img').alt;
                const aData = imageData.find(d => d.filename === aFile) || {{}};
                const bData = imageData.find(d => d.filename === bFile) || {{}};
                switch(sortBy) {{
                    case 'newest': return images.indexOf(aFile) - images.indexOf(bFile);
                    case 'oldest': return images.indexOf(bFile) - images.indexOf(aFile);
                    case 'epoch-desc': return (bData.epoch || 0) - (aData.epoch || 0);
                    case 'epoch-asc': return (aData.epoch || 0) - (bData.epoch || 0);
                    default: return 0;
                }}
            }});
            items.forEach(item => gallery.appendChild(item));
        }}
        function getImageMeta(filename) {{
            const data = imageData.find(d => d.filename === filename);
            if (!data) return '';
            const parts = [];
            if (data.epoch !== null) parts.push('Epoch ' + data.epoch);
            if (data.step !== null) parts.push('Step ' + data.step);
            if (data.seed !== null) parts.push('Seed: ' + data.seed);
            return parts.join(' | ');
        }}
        function openLightbox(filename) {{
            currentImageIndex = images.indexOf(filename);
            document.getElementById('lightbox-img').src = filename;
            document.getElementById('lightbox-name').textContent = filename;
            document.getElementById('lightbox-meta').textContent = getImageMeta(filename);
            document.getElementById('lightbox').classList.add('active');
            document.body.style.overflow = 'hidden';
        }}
        function closeLightbox() {{
            document.getElementById('lightbox').classList.remove('active');
            document.body.style.overflow = '';
        }}
        function navigateLightbox(direction) {{
            if (images.length === 0) return;
            currentImageIndex = (currentImageIndex + direction + images.length) % images.length;
            const filename = images[currentImageIndex];
            document.getElementById('lightbox-img').src = filename;
            document.getElementById('lightbox-name').textContent = filename;
            document.getElementById('lightbox-meta').textContent = getImageMeta(filename);
        }}
        document.addEventListener('keydown', (e) => {{
            if (!document.getElementById('lightbox').classList.contains('active')) return;
            if (e.key === 'Escape') closeLightbox();
            if (e.key === 'ArrowLeft') navigateLightbox(-1);
            if (e.key === 'ArrowRight') navigateLightbox(1);
        }});
        document.getElementById('lightbox').addEventListener('click', (e) => {{ if (e.target.id === 'lightbox') closeLightbox(); }});
        sortGallery();
    </script>
</body>
</html>'''
        return html

    def create_image_converter(self):
        """Create the Image Prep tab (Start-tab styled)."""
        scrollable_frame, _ = self.create_scrollable_frame(self.image_converter_tab)

        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer,
            "Image Prep",
            "Resize, convert to PNG, and optionally face-crop your training images. "
            "Optional — skip straight to Captions if your images are already prepared.",
        )

        # Card 1: Folders
        folders_card = self._start_section_card(
            outer, "Folders",
            "Source is the Training image folder from the Start tab. Output is where "
            "prepared images land — leave blank to write next to the originals.",
        )
        folders_card.grid_columnconfigure(1, weight=1)

        ttk.Label(folders_card, text="Source Folder:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        tk.Label(folders_card, textvariable=self.image_folder_var,
                 font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
                 anchor="w").grid(row=0, column=1, columnspan=2, sticky=tk.W, pady=4)
        tk.Label(folders_card, text="(set on the Start tab)",
                 font=(FONT_FAMILY, 9, "italic"),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=1, column=1, columnspan=2, sticky=tk.W, pady=(0, 8)
        )

        ttk.Label(folders_card, text="Output Folder:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        ttk.Entry(folders_card, textvariable=self.convert_output_var, width=40).grid(
            row=2, column=1, sticky=tk.EW, pady=4
        )
        ttk.Button(folders_card, text="Browse", command=self.browse_convert_output).grid(
            row=2, column=2, sticky=tk.W, padx=(8, 0), pady=4
        )
        tk.Label(folders_card, text="(leave empty to save in source folder)",
                 font=(FONT_FAMILY, 9, "italic"),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=3, column=1, columnspan=2, sticky=tk.W
        )

        # Card 2: Resize
        resize_card = self._start_section_card(
            outer, "Resize",
            "Images larger than Max Size are downscaled on the longer edge; smaller images are left untouched.",
        )
        resize_card.grid_columnconfigure(1, weight=1)

        ttk.Label(resize_card, text="Max Size (px):").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        ttk.Combobox(resize_card, textvariable=self.max_size_var,
                     values=["256", "512", "640", "768", "1024", "1256"],
                     state="readonly", width=10).grid(row=0, column=1, sticky=tk.W, pady=4)

        # Card 3: Prep Mode (face-related controls live here; _on_prep_mode_changed grid_removes them)
        prep_card = self._start_section_card(
            outer, "Prep Mode",
            "Auto Prep generates face-cropped derivatives alongside resized originals. "
            "Resize Only skips face detection; Face Crop Only replaces the original with the detected crop.",
        )
        prep_card.grid_columnconfigure(1, weight=1)

        ttk.Label(prep_card, text="Mode:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.prep_mode_combo = ttk.Combobox(
            prep_card, textvariable=self.prep_mode_var,
            values=["Auto Prep (Face Crops)", "Resize Only", "Face Crop Only"],
            state="readonly", width=24,
        )
        self.prep_mode_combo.grid(row=0, column=1, sticky=tk.W, pady=4)
        self.prep_mode_combo.bind("<<ComboboxSelected>>", self._on_prep_mode_changed)

        # Face Target (row 1 — hidden when Resize Only)
        self._face_target_label = ttk.Label(prep_card, text="Face Target:")
        self._face_target_label.grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self._face_target_combo = ttk.Combobox(
            prep_card, textvariable=self.face_selection_var,
            values=["Largest Face", "Largest Male Face", "Largest Female Face"],
            state="readonly" if FACE_DETECTION_AVAILABLE else "disabled", width=20,
        )
        self._face_target_combo.grid(row=1, column=1, sticky=tk.W, pady=4)
        self._face_target_row = 1

        if not FACE_DETECTION_AVAILABLE:
            self._face_unavail_label = ttk.Label(
                prep_card, text="(Run install_fizgig.py to enable)",
                foreground=COLORS["warning"],
            )
            self._face_unavail_label.grid(row=1, column=2, sticky=tk.W, padx=(8, 0))
        else:
            self._face_unavail_label = None

        # Face Padding (row 2 — hidden when Resize Only)
        self._face_padding_label = ttk.Label(prep_card, text="Face Padding (%):")
        self._face_padding_label.grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self._face_padding_frame = tk.Frame(prep_card, bg=COLORS["bg_surface"])
        self._face_padding_frame.grid(row=2, column=1, sticky=tk.W, pady=4)
        ttk.Entry(self._face_padding_frame, textvariable=self.face_padding_var, width=8).pack(side=tk.LEFT)
        tk.Label(self._face_padding_frame, text="(extra space around face)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(8, 0))
        self._face_padding_row = 2

        # Replace originals (row 3)
        replace_frame = tk.Frame(prep_card, bg=COLORS["bg_surface"])
        replace_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        ttk.Checkbutton(
            replace_frame, text="Replace originals", variable=self.delete_originals_var,
            command=self._update_prep_note,
        ).pack(side=tk.LEFT)
        tk.Label(replace_frame, text="(untick to keep originals in a subfolder)",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(8, 0))

        # Dynamic note (row 4)
        self._prep_note_var = tk.StringVar()
        self._prep_note_label = tk.Label(
            prep_card, textvariable=self._prep_note_var,
            font=(FONT_FAMILY, 9, "italic"),
            fg=COLORS["accent_hover"], bg=COLORS["bg_surface"],
            wraplength=700, justify=tk.LEFT,
        )
        self._prep_note_label.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))

        # Card 4: Actions
        action_card = self._start_section_card(outer, "Run", None)

        button_frame = tk.Frame(action_card, bg=COLORS["bg_surface"])
        button_frame.pack(anchor=tk.W)
        self.preview_faces_btn = ttk.Button(
            button_frame, text="Preview Faces", command=self.preview_faces,
            state="normal" if FACE_DETECTION_AVAILABLE else "disabled",
        )
        self.preview_faces_btn.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_frame, text="Prepare Images", command=self.convert_images).pack(side=tk.LEFT)

        # Apply initial visibility (may grid_remove face widgets in prep_card)
        self._on_prep_mode_changed()

        # Card 5: Output Log
        log_card = self._start_section_card(outer, "Output Log", None)

        self.convert_log = scrolledtext.ScrolledText(
            log_card, height=12, width=80,
            bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
            wrap="word", state="disabled",
            relief="flat", highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["border_focus"],
        )
        self.convert_log.pack(fill=tk.BOTH, expand=True)

        self._add_youtube_help_button(outer, "image_prep")

    def browse_convert_output(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory()
        if folder:
            self.convert_output_var.set(folder)

    @property
    def face_detector(self):
        """Lazy-loaded face detector instance"""
        if self._face_detector is None and FACE_DETECTION_AVAILABLE:
            self._face_detector = FaceDetector()
        return self._face_detector

    def _on_prep_mode_changed(self, *args):
        """Show/hide face-related controls based on prep mode."""
        mode = self.prep_mode_var.get()
        show_face = mode != "Resize Only"

        if show_face:
            self._face_target_label.grid()
            self._face_target_combo.grid()
            self._face_padding_label.grid()
            self._face_padding_frame.grid()
            if self._face_unavail_label:
                self._face_unavail_label.grid()
            self.preview_faces_btn.configure(state="normal" if FACE_DETECTION_AVAILABLE else "disabled")
        else:
            self._face_target_label.grid_remove()
            self._face_target_combo.grid_remove()
            self._face_padding_label.grid_remove()
            self._face_padding_frame.grid_remove()
            if self._face_unavail_label:
                self._face_unavail_label.grid_remove()
            self.preview_faces_btn.configure(state="disabled")

        self._update_prep_note()

    def _update_prep_note(self):
        """Update the contextual note based on prep mode and replace originals setting."""
        if not hasattr(self, '_prep_note_var'):
            return
        mode = self.prep_mode_var.get()
        replace = self.delete_originals_var.get()

        if mode == "Auto Prep (Face Crops)":
            if replace:
                note = "Result: Your folder will contain resized originals + face crop derivatives. Images larger than max size will be shrunk in place."
            else:
                note = "Result: Your folder will contain resized copies + face crop derivatives. Original files will be moved to an 'originals' subfolder."
        elif mode == "Resize Only":
            if replace:
                note = "Result: Images larger than max size will be shrunk in place and converted to PNG. Smaller images converted to PNG only."
            else:
                note = "Result: Resized PNG copies in your folder. Original files will be moved to an 'originals' subfolder."
        elif mode == "Face Crop Only":
            if replace:
                note = "Result: Your folder will contain ONLY face crops. Original images will be replaced."
            else:
                note = "Result: Face crops in your folder. Original files will be moved to an 'originals' subfolder."
        else:
            note = ""

        self._prep_note_var.set(note)

    def _get_face_selection_mode(self):
        """Parse face selection mode from Face Target dropdown."""
        mode_text = self.face_selection_var.get()
        if "Male" in mode_text:
            return "largest_male"
        elif "Female" in mode_text:
            return "largest_female"
        return "largest_face"

    def preview_faces(self):
        """Preview face detection on a single image"""
        if not FACE_DETECTION_AVAILABLE:
            messagebox.showerror("Error", "Face detection not available. Run install_fizgig.py first.")
            return

        filepath = filedialog.askopenfilename(
            title="Select image to preview faces",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif")]
        )
        if not filepath:
            return

        try:
            # Detect faces
            faces = self.face_detector.detect_all(filepath)

            if not faces:
                messagebox.showinfo("No Faces", f"No faces detected in:\n{os.path.basename(filepath)}")
                return

            # Find the largest face (or by gender based on current mode)
            crop_mode = self._get_face_selection_mode()
            if crop_mode == "largest_male":
                selected = self.face_detector.get_largest_by_gender(faces, "male", fallback_to_any=True)
            elif crop_mode == "largest_female":
                selected = self.face_detector.get_largest_by_gender(faces, "female", fallback_to_any=True)
            else:
                selected = self.face_detector.get_largest(faces)

            # Get highlight index
            highlight_idx = faces.index(selected) if selected in faces else None

            # Load image and draw boxes
            with Image.open(filepath) as img:
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')

                preview_img = draw_face_boxes(img, faces, highlight_index=highlight_idx)

                # Create preview window
                self._show_face_preview_window(preview_img, faces, filepath, highlight_idx)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect faces:\n{str(e)}")

    def _show_face_preview_window(self, preview_img, faces, filepath, highlight_idx):
        """Show a popup window with the face detection preview"""
        preview_window = tk.Toplevel(self.master)
        preview_window.title(f"Face Preview - {os.path.basename(filepath)}")
        preview_window.configure(bg=BG_COLOR)

        # Resize for display if too large
        display_img = preview_img.copy()
        max_display = 800
        if display_img.width > max_display or display_img.height > max_display:
            ratio = min(max_display / display_img.width, max_display / display_img.height)
            new_size = (int(display_img.width * ratio), int(display_img.height * ratio))
            display_img = display_img.resize(new_size, Image.LANCZOS)

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(display_img)

        # Image label
        img_label = ttk.Label(preview_window, image=photo)
        img_label.image = photo  # Keep reference
        img_label.pack(padx=10, pady=10)

        # Info frame
        info_frame = ttk.Frame(preview_window)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        # Face count
        ttk.Label(info_frame, text=f"Faces detected: {len(faces)}", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        # Face details
        for i, face in enumerate(faces):
            marker = " [SELECTED]" if i == highlight_idx else ""
            gender = face.gender.capitalize() if face.gender != 'unknown' else '?'
            ttk.Label(
                info_frame,
                text=f"  Face {i+1}: {gender}, {face.area:,} px{marker}"
            ).pack(anchor=tk.W)

        # Legend
        legend_frame = ttk.Frame(preview_window)
        legend_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(legend_frame, text="Green = Selected for cropping", foreground="green").pack(side=tk.LEFT, padx=10)
        ttk.Label(legend_frame, text="Yellow = Other faces", foreground="yellow").pack(side=tk.LEFT, padx=10)

        # Close button
        ttk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=10)

    # region Image Prep Helpers

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

    def _handle_original(self, filepath, output_path, output_folder, replace_originals):
        """Handle the original file: delete if replacing, move to subfolder if preserving."""
        if filepath == output_path:
            return  # Output overwrote original, nothing to do
        if replace_originals:
            os.remove(filepath)
        else:
            if not hasattr(self, '_originals_dir_cache'):
                self._originals_dir_cache = {}
            if output_folder not in self._originals_dir_cache:
                self._originals_dir_cache[output_folder] = self._get_originals_dir(output_folder)
            originals_dir = self._originals_dir_cache[output_folder]
            os.makedirs(originals_dir, exist_ok=True)
            import shutil
            dest = os.path.join(originals_dir, os.path.basename(filepath))
            shutil.move(filepath, dest)

    def _get_originals_dir(self, output_folder):
        """Find the next available originals folder (originals, originals_2, originals_3, etc.)."""
        candidate = os.path.join(output_folder, "originals")
        if not os.path.isdir(candidate):
            return candidate
        # Check if it has any images
        has_images = any(
            os.path.splitext(f)[1].lower() in self.IMAGE_EXTENSIONS
            for f in os.listdir(candidate) if os.path.isfile(os.path.join(candidate, f))
        )
        if not has_images:
            return candidate
        # Find next numbered folder
        n = 2
        while True:
            candidate = os.path.join(output_folder, f"originals_{n}")
            if not os.path.isdir(candidate):
                return candidate
            has_images = any(
                os.path.splitext(f)[1].lower() in self.IMAGE_EXTENSIONS
                for f in os.listdir(candidate) if os.path.isfile(os.path.join(candidate, f))
            )
            if not has_images:
                return candidate
            n += 1

    def _get_image_files(self, folder):
        """Scan folder for image files, return sorted list of full paths."""
        files = []
        for filename in sorted(os.listdir(folder)):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath) and os.path.splitext(filename)[1].lower() in self.IMAGE_EXTENSIONS:
                files.append(filepath)
        return files

    def _load_image(self, filepath):
        """Load an image and convert to RGB/RGBA."""
        img = Image.open(filepath)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGBA')
        else:
            img = img.convert('RGB')
        return img

    def _resize_image(self, img, max_size):
        """Resize image maintaining aspect ratio. Never upscales. Returns (img, resized_bool)."""
        width, height = img.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            return img.resize((new_width, new_height), Image.LANCZOS), True
        return img, False

    def _select_face(self, faces, face_mode):
        """Select a face from detected faces based on face_mode. Returns (FaceInfo, note_str) or (None, note_str)."""
        if not faces:
            return None, ""
        if face_mode == "largest_male":
            selected = self.face_detector.get_largest_by_gender(faces, "male", fallback_to_any=True)
            note = "  Note: No male face, using largest face\n" if (selected and selected.gender != "male") else ""
        elif face_mode == "largest_female":
            selected = self.face_detector.get_largest_by_gender(faces, "female", fallback_to_any=True)
            note = "  Note: No female face, using largest face\n" if (selected and selected.gender != "female") else ""
        else:
            selected = self.face_detector.get_largest(faces)
            note = ""
        return selected, note

    def _get_next_facecrop_index(self, folder):
        """Find next available FaceCrop_NNN index in a folder."""
        import glob as glob_module
        existing = glob_module.glob(os.path.join(folder, "FaceCrop_*.png"))
        max_idx = 0
        for f in existing:
            basename = os.path.splitext(os.path.basename(f))[0]
            parts = basename.split("_")
            if len(parts) >= 2:
                try:
                    max_idx = max(max_idx, int(parts[1]))
                except ValueError:
                    pass
        return max_idx + 1

    def _log(self, text):
        """Append text to the convert log (preserves user scroll position)."""
        try:
            at_bottom = self.convert_log.yview()[1] >= 0.999
        except Exception:
            at_bottom = True
        self.convert_log.insert(tk.END, text)
        if at_bottom:
            self.convert_log.see(tk.END)
        self.convert_log.see(tk.END)
        self.master.update_idletasks()

    # endregion

    # region Dataset Analysis & Smart Defaults

    def _analyze_dataset(self, folder):
        """Analyze a dataset folder: count images, detect face crops, check captions."""
        if not folder or not os.path.isdir(folder):
            return None

        files = self._get_image_files(folder)
        face_crops = 0
        full_shots = 0
        for f in files:
            basename = os.path.splitext(os.path.basename(f))[0]
            if basename.startswith("FaceCrop_"):
                face_crops += 1
            else:
                full_shots += 1

        # Count caption files
        caption_count = 0
        for f in os.listdir(folder):
            if f.endswith(".txt") and os.path.isfile(os.path.join(folder, f)):
                caption_count += 1

        return {
            "total_images": len(files),
            "face_crops": face_crops,
            "full_shots": full_shots,
            "has_captions": caption_count > 0,
            "caption_count": caption_count,
        }

    def _recommend_training_settings(self, analysis):
        """Recommend rank, LR, and epochs based on dataset analysis.
        Based on empirical findings from the Fizgig Expansion Vision document."""
        if analysis is None:
            return None

        total = analysis["total_images"]
        face_crops = analysis["face_crops"]

        if total >= 80 and face_crops >= 30:
            return {
                "rank": 4, "alpha": 4, "lr": 0.0004, "epochs": 12,
                "tier": "optimal",
                "summary": f"{total} images ({face_crops} face crops, {analysis['full_shots']} full shots)",
                "message": "Strong dataset for rank 4:4. Fast convergence expected.",
            }
        elif total >= 40 and face_crops >= 15:
            return {
                "rank": 4, "alpha": 4, "lr": 0.0003, "epochs": 16,
                "tier": "good",
                "summary": f"{total} images ({face_crops} face crops, {analysis['full_shots']} full shots)",
                "message": "Good dataset for rank 4:4. Slightly conservative LR recommended.",
            }
        elif total >= 20:
            warnings = []
            if face_crops < 15:
                warnings.append("Few face crops — use Auto Prep on the Image Prep tab to generate more.")
            return {
                "rank": 8, "alpha": 8, "lr": 0.0002, "epochs": 25,
                "tier": "caution",
                "summary": f"{total} images ({face_crops} face crops, {analysis['full_shots']} full shots)",
                "message": "Small dataset. Rank 8 recommended over rank 4.",
                "warnings": warnings,
            }
        else:
            return {
                "rank": 16, "alpha": 16, "lr": 0.0001, "epochs": 40,
                "tier": "limited",
                "summary": f"{total} images ({face_crops} face crops, {analysis['full_shots']} full shots)",
                "message": "Very small dataset. Higher rank needed to avoid underfitting.",
                "warnings": ["Very small dataset. Results may be inconsistent. Add more training images if possible."],
            }

    def _update_dataset_recommendation(self, *args):
        """Analyze dataset and update the recommendation panel on the Training tab."""
        if not hasattr(self, '_rec_summary_var'):
            return  # UI not built yet

        # Authoritative training folder lives on the Start tab (self.image_folder_var).
        folder = self.image_folder_var.get()
        analysis = self._analyze_dataset(folder)
        rec = self._recommend_training_settings(analysis)

        if rec is None:
            self._rec_summary_var.set("")
            self._rec_detail_var.set("")
            self._rec_warning_var.set("")
            self._last_recommendation = None
            return

        self._last_recommendation = rec

        # Tier colors
        tier_prefix = {"optimal": "Optimal", "good": "Good", "caution": "Caution", "limited": "Limited"}
        self._rec_summary_var.set(f"Dataset: {rec['summary']}")
        self._rec_detail_var.set(
            f"Recommended: rank {rec['rank']}:{rec['alpha']}, LR {rec['lr']}, ~{rec['epochs']} epochs  [{tier_prefix[rec['tier']]}]"
        )

        # Warnings
        warnings = rec.get("warnings", [])
        # Also check current rank vs recommendation
        try:
            current_rank = int(self.entries.get("NETWORK_DIM", tk.Entry()).get())
        except (ValueError, AttributeError):
            current_rank = 0
        if current_rank > 0 and current_rank <= 4 and rec["rank"] > 4:
            warnings.append(f"Current rank {current_rank} may be too low for this dataset size. Recommended: {rec['rank']}.")
        if analysis and not analysis["has_captions"]:
            warnings.append("No caption files (.txt) found — captions are required for training.")

        self._rec_warning_var.set("\n".join(warnings) if warnings else "")

    def _apply_recommendation(self):
        """Apply recommended settings to the training fields."""
        rec = getattr(self, '_last_recommendation', None)
        if rec is None:
            return

        if "NETWORK_DIM" in self.entries:
            self.entries["NETWORK_DIM"].delete(0, tk.END)
            self.entries["NETWORK_DIM"].insert(0, str(rec["rank"]))
        if "NETWORK_ALPHA" in self.entries:
            self.entries["NETWORK_ALPHA"].delete(0, tk.END)
            self.entries["NETWORK_ALPHA"].insert(0, str(rec["alpha"]))
        if "LEARNING_RATE" in self.entries:
            self.entries["LEARNING_RATE"].delete(0, tk.END)
            self.entries["LEARNING_RATE"].insert(0, str(rec["lr"]))
        if "MAX_TRAIN_EPOCHS" in self.entries:
            self.entries["MAX_TRAIN_EPOCHS"].delete(0, tk.END)
            self.entries["MAX_TRAIN_EPOCHS"].insert(0, str(rec["epochs"]))

        self._update_dataset_recommendation()  # Refresh warnings

    # endregion

    def convert_images(self):
        """Prepare images based on selected prep mode."""
        self._originals_dir_cache = {}  # Reset per run
        source_folder = self.image_folder_var.get()
        output_folder = self.convert_output_var.get() or source_folder
        max_size = int(self.max_size_var.get())
        replace_originals = self.delete_originals_var.get()
        prep_mode = self.prep_mode_var.get()
        face_mode = self._get_face_selection_mode()

        try:
            face_padding = float(self.face_padding_var.get())
        except ValueError:
            face_padding = 20.0

        if not source_folder:
            messagebox.showerror("Error", "Please select a source folder.")
            return
        if not os.path.isdir(source_folder):
            messagebox.showerror("Error", "Source folder does not exist.")
            return

        # Check face detection for modes that need it
        if prep_mode != "Resize Only" and not FACE_DETECTION_AVAILABLE:
            messagebox.showerror("Error", "Face detection not available.\nRun install_fizgig.py to enable.")
            return

        os.makedirs(output_folder, exist_ok=True)

        # Clear log
        self.convert_log.configure(state="normal")
        self.convert_log.delete(1.0, tk.END)

        if prep_mode == "Auto Prep (Face Crops)":
            self._auto_prep_images(source_folder, output_folder, max_size, face_mode, face_padding, replace_originals)
        elif prep_mode == "Resize Only":
            self._resize_only_images(source_folder, output_folder, max_size, replace_originals)
        elif prep_mode == "Face Crop Only":
            self._face_crop_only_images(source_folder, output_folder, max_size, face_mode, face_padding, replace_originals)

        self.convert_log.configure(state="disabled")
        self.convert_log.see(tk.END)

    def _resize_only_images(self, source_folder, output_folder, max_size, replace_originals):
        """Resize Only mode: convert/resize images, no face detection."""
        self._log("Mode: Resize Only\n\n")
        files = self._get_image_files(source_folder)
        converted, skipped, errors = 0, 0, 0

        for filepath in files:
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            try:
                img = self._load_image(filepath)
                original_size = img.size
                img, resized = self._resize_image(img, max_size)
                w, h = img.size

                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, base_name + ".png")

                if filepath == output_path and ext == '.png' and not resized:
                    self._log(f"Skipped (no changes): {filename}\n")
                    skipped += 1
                    img.close()
                    continue

                img.save(output_path, "PNG")
                size_info = f"{original_size[0]}x{original_size[1]} -> {w}x{h}" if resized else f"{w}x{h}"
                self._log(f"Converted: {filename} [{size_info}]\n")
                converted += 1
                img.close()

                self._handle_original(filepath, output_path, output_folder, replace_originals)

            except Exception as e:
                self._log(f"Error ({filename}): {e}\n")
                errors += 1

        self._log(f"\n--- Summary ---\nConverted: {converted} | Skipped: {skipped} | Errors: {errors}\n")

    def _face_crop_only_images(self, source_folder, output_folder, max_size, face_mode, face_padding, replace_originals):
        """Face Crop Only mode: face crop replaces the output."""
        self._log(f"Mode: Face Crop Only ({face_mode}, padding {face_padding}%)\n\n")
        files = self._get_image_files(source_folder)
        converted, skipped, errors, face_crops, no_face = 0, 0, 0, 0, 0

        for filepath in files:
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            try:
                img = self._load_image(filepath)
                original_size = img.size
                cropped = False
                crop_info = ""

                try:
                    faces = self.face_detector.detect_from_pil(img)
                    if faces:
                        selected, note = self._select_face(faces, face_mode)
                        if note:
                            self._log(note)
                        if selected:
                            img = crop_to_face(img, selected, face_padding)
                            cropped = True
                            face_crops += 1
                            crop_info = f" [face: {selected.gender}]"
                    else:
                        self._log(f"  No face in {filename}, skipping crop\n")
                        no_face += 1
                except Exception as fe:
                    self._log(f"  Face error ({filename}): {fe}\n")

                img, resized = self._resize_image(img, max_size)
                w, h = img.size

                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, base_name + ".png")

                if filepath == output_path and ext == '.png' and not resized and not cropped:
                    self._log(f"Skipped (no changes): {filename}\n")
                    skipped += 1
                    img.close()
                    continue

                img.save(output_path, "PNG")
                size_info = f"{original_size[0]}x{original_size[1]} -> {w}x{h}" if (resized or cropped) else f"{w}x{h}"
                self._log(f"Converted: {filename} [{size_info}]{crop_info}\n")
                converted += 1
                img.close()

                self._handle_original(filepath, output_path, output_folder, replace_originals)

            except Exception as e:
                self._log(f"Error ({filename}): {e}\n")
                errors += 1

        self._log(f"\n--- Summary ---\nConverted: {converted} | Skipped: {skipped} | Errors: {errors}\n")
        self._log(f"Face crops: {face_crops} | No face: {no_face}\n")

    def _auto_prep_images(self, source_folder, output_folder, max_size, face_mode, face_padding, replace_originals):
        """Auto Prep mode: resize originals in place, then generate FaceCrop derivatives."""
        self._log(f"Mode: Auto Prep (Face Crops)\n")
        self._log(f"Face target: {face_mode}, padding: {face_padding}%\n")
        self._log(f"Output: {output_folder}\n\n")

        files = self._get_image_files(source_folder)
        converted, skipped, errors = 0, 0, 0
        face_crops_created, no_face = 0, 0

        # Phase A: Resize/convert originals
        self._log("--- Phase 1: Resize/Convert Originals ---\n")
        for filepath in files:
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            base_name = os.path.splitext(filename)[0]

            # Skip existing FaceCrop derivatives
            if base_name.startswith("FaceCrop_"):
                self._log(f"Skipped (derivative): {filename}\n")
                skipped += 1
                continue

            try:
                img = self._load_image(filepath)
                original_size = img.size
                img, resized = self._resize_image(img, max_size)
                w, h = img.size

                output_path = os.path.join(output_folder, base_name + ".png")

                if filepath == output_path and ext == '.png' and not resized:
                    self._log(f"OK (no changes): {filename}\n")
                    skipped += 1
                    img.close()
                    continue

                img.save(output_path, "PNG")
                size_info = f"{original_size[0]}x{original_size[1]} -> {w}x{h}" if resized else f"{w}x{h}"
                self._log(f"Converted: {filename} [{size_info}]\n")
                converted += 1
                img.close()

                self._handle_original(filepath, output_path, output_folder, replace_originals)

            except Exception as e:
                self._log(f"Error ({filename}): {e}\n")
                errors += 1

        # Phase B: Generate face crop derivatives
        self._log(f"\n--- Phase 2: Generate Face Crop Derivatives ---\n")
        crop_index = self._get_next_facecrop_index(output_folder)

        # Re-scan output folder for the resized originals (they're now in output_folder)
        output_files = self._get_image_files(output_folder)

        for filepath in output_files:
            filename = os.path.basename(filepath)
            base_name = os.path.splitext(filename)[0]

            # Skip existing FaceCrop derivatives
            if base_name.startswith("FaceCrop_"):
                continue

            try:
                img = self._load_image(filepath)

                faces = self.face_detector.detect_from_pil(img)
                if not faces:
                    self._log(f"No face: {filename}\n")
                    no_face += 1
                    img.close()
                    continue

                selected, note = self._select_face(faces, face_mode)
                if note:
                    self._log(note)

                if not selected:
                    no_face += 1
                    img.close()
                    continue

                cropped = crop_to_face(img, selected, face_padding)
                cropped, _ = self._resize_image(cropped, max_size)

                crop_name = f"FaceCrop_{crop_index:03d}.png"
                crop_path = os.path.join(output_folder, crop_name)
                cropped.save(crop_path, "PNG")
                w, h = cropped.size
                self._log(f"Created: {crop_name} ({w}x{h}) from {filename} [{selected.gender}]\n")
                face_crops_created += 1
                crop_index += 1

                cropped.close()
                img.close()

            except Exception as e:
                self._log(f"Error ({filename}): {e}\n")
                errors += 1

        self._log(f"\n--- Summary ---\n")
        self._log(f"Originals converted: {converted} | Skipped: {skipped} | Errors: {errors}\n")
        self._log(f"Face crops created: {face_crops_created} | No face: {no_face}\n")
        self._log(f"Total files in output: {len(self._get_image_files(output_folder))}\n")


    # region LoRA the Explorer Tab

    def create_explorer_tab(self):
        """LoRA the Explorer — evolutionary LoRA discovery via human-guided selection."""
        scrollable_frame, _ = self.create_scrollable_frame(self.explorer_tab)
        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer, "LoRA the Explorer",
            "The computer randomly adjusts blocks and shows you 4 variants — pick your favourite and it evolves. "
            "Find a direction you like? Reduce Structure to stabilise composition, then Freeze to lock tweaked blocks in place.",
        )

        # State
        self._explorer_engine = None
        self._explorer_baseline_state = None
        self._explorer_baseline_image = None
        self._explorer_history = []  # stack of (SliderState, PIL.Image) for undo
        self._explorer_variant_states = []  # 4 SliderState objects
        self._explorer_variant_images = []  # 4 PIL.Image objects
        self._explorer_generating = False
        self._explorer_thumbnails = {}  # keep refs to prevent GC
        self._explorer_locked_blocks = set()  # Freeze: blocks locked at their current value
        self._explorer_last_pick_blocks = set()  # blocks changed in the most recent pick

        # Card 1: Setup
        setup_card = self._start_section_card(
            outer, "Setup",
            "Load a LoRA and configure the exploration parameters.",
        )
        setup_card.columnconfigure(1, weight=1)

        r = 0
        ttk.Label(setup_card, text="DiT:").grid(row=r, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        self.explorer_dit_var = tk.StringVar(value="distilled")
        dit_frame = ttk.Frame(setup_card)
        dit_frame.grid(row=r, column=1, sticky=tk.W, pady=2)
        ttk.Radiobutton(dit_frame, text="Distilled (4-step, fast)",
                        variable=self.explorer_dit_var, value="distilled",
                        style="Surface.TRadiobutton").pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(dit_frame, text="Base (20-step, precise)",
                        variable=self.explorer_dit_var, value="base",
                        style="Surface.TRadiobutton").pack(side=tk.LEFT)
        r += 1

        ttk.Label(setup_card, text="LoRA:").grid(row=r, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        self.explorer_lora_var = tk.StringVar()
        ttk.Entry(setup_card, textvariable=self.explorer_lora_var).grid(
            row=r, column=1, sticky=tk.EW, padx=4, pady=2)
        btn_frame = ttk.Frame(setup_card)
        btn_frame.grid(row=r, column=2, pady=2)
        ttk.Button(btn_frame, text="Browse",
                   command=lambda: self._browse_repair_lora(self.explorer_lora_var)).pack(side=tk.LEFT, padx=2)
        ttk.Label(btn_frame, text="Strength:").pack(side=tk.LEFT, padx=(12, 4))
        self.explorer_strength_var = tk.StringVar(value="1.0")
        ttk.Entry(btn_frame, textvariable=self.explorer_strength_var, width=5).pack(side=tk.LEFT)
        r += 1

        ttk.Label(setup_card, text="Prompt:").grid(row=r, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        self.explorer_prompt_var = tk.StringVar()
        ttk.Entry(setup_card, textvariable=self.explorer_prompt_var).grid(
            row=r, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=2)
        r += 1

        params_frame = ttk.Frame(setup_card)
        params_frame.grid(row=r, column=0, columnspan=3, sticky=tk.W, pady=(6, 2))
        ttk.Label(params_frame, text="Seed:").pack(side=tk.LEFT, padx=(0, 4))
        self.explorer_seed_var = tk.StringVar(value="42")
        ttk.Entry(params_frame, textvariable=self.explorer_seed_var, width=8).pack(side=tk.LEFT, padx=(0, 2))
        tk.Button(params_frame, text="\u21bb", font=(FONT_FAMILY, 9),
                  bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
                  activebackground=COLORS["bg_surface"], activeforeground=COLORS["text_primary"],
                  relief="flat", bd=0, padx=4, pady=0, cursor="hand2",
                  command=self._explorer_randomize_seed
                  ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(params_frame, text="Res:").pack(side=tk.LEFT, padx=(0, 4))
        self.explorer_res_var = tk.StringVar(value="512")
        ttk.Combobox(params_frame, textvariable=self.explorer_res_var,
                     values=["256", "384", "512", "768"], state="readonly", width=5).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(params_frame, text="Intensity:").pack(side=tk.LEFT, padx=(0, 4))
        self.explorer_intensity_var = tk.DoubleVar(value=0.964)
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.explorer_intensity_var,
                  orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=(0, 4))
        self._explorer_intensity_lbl = ttk.Label(params_frame, text="\u00b12.9", width=5)
        self._explorer_intensity_lbl.pack(side=tk.LEFT, padx=(0, 16))
        self._explorer_intensity_debounce_id = None
        def _update_intensity_lbl(*_):
            mag = 0.2 + self.explorer_intensity_var.get() * 2.8
            self._explorer_intensity_lbl.configure(text=f"\u00b1{mag:.1f}")
            # Debounced re-roll when intensity changes
            if self._explorer_baseline_state is not None and not self._explorer_generating:
                if self._explorer_intensity_debounce_id is not None:
                    try:
                        self.master.after_cancel(self._explorer_intensity_debounce_id)
                    except Exception:
                        pass
                self._explorer_intensity_debounce_id = self.master.after(
                    750, self._explorer_reroll)
        self.explorer_intensity_var.trace_add("write", _update_intensity_lbl)
        ttk.Label(params_frame, text="Mutations:").pack(side=tk.LEFT, padx=(0, 4))
        self.explorer_mutations_var = tk.StringVar(value="8")
        ttk.Combobox(params_frame, textvariable=self.explorer_mutations_var,
                     values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "14", "16"], state="readonly", width=3).pack(side=tk.LEFT, padx=(0, 16))
        # Hold Mode removed — replaced by Freeze Tweaked Blocks button
        r += 1

        struct_frame = ttk.Frame(setup_card)
        struct_frame.grid(row=r, column=0, columnspan=3, sticky=tk.W, pady=(2, 0))
        ttk.Label(struct_frame, text="Structure change:").pack(side=tk.LEFT, padx=(0, 4))
        self.explorer_structure_var = tk.DoubleVar(value=1.0)
        ttk.Scale(struct_frame, from_=0.0, to=1.0, variable=self.explorer_structure_var,
                  orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=(0, 4))
        self._explorer_structure_lbl = ttk.Label(struct_frame, text="100%", width=5)
        self._explorer_structure_lbl.pack(side=tk.LEFT, padx=(0, 8))
        self._explorer_structure_debounce_id = None
        def _update_structure_lbl(*_):
            val = self.explorer_structure_var.get()
            self._explorer_structure_lbl.configure(text=f"{int(val * 100)}%")
            if self._explorer_baseline_state is not None and not self._explorer_generating:
                if self._explorer_structure_debounce_id is not None:
                    try:
                        self.master.after_cancel(self._explorer_structure_debounce_id)
                    except Exception:
                        pass
                self._explorer_structure_debounce_id = self.master.after(
                    750, self._explorer_reroll)
        self.explorer_structure_var.trace_add("write", _update_structure_lbl)
        r += 1

        status_row = tk.Frame(setup_card, bg=COLORS["bg_surface"])
        status_row.grid(row=r, column=0, columnspan=3, sticky=tk.EW, pady=(4, 0))
        tk.Label(status_row,
                 text="Increase Structure if variants look too similar to baseline.",
                 font=(FONT_FAMILY, 8, "italic"),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
        self._explorer_start_btn = tk.Button(
            status_row, text="Start", font=(FONT_FAMILY, 11, "bold"),
            fg="#FFFFFF", bg="#2E8B57", activeforeground="#FFFFFF", activebackground="#256F46",
            relief="flat", bd=0, padx=24, pady=6, cursor="hand2",
            command=self._explorer_start)
        self._explorer_start_btn.pack(side=tk.RIGHT, padx=(8, 0))
        self.explorer_status_var = tk.StringVar(value="Set a LoRA path and prompt, then click Start.")
        tk.Label(status_row, textvariable=self.explorer_status_var,
                 font=(FONT_FAMILY, 10, "italic"),
                 fg=COLORS["accent"], bg=COLORS["bg_surface"]).pack(side=tk.RIGHT)

        # Card 2: Baseline
        baseline_card = self._start_section_card(
            outer, "Current Baseline",
            "Your current best. Pick a favourite below to evolve it, or save it as a LoRA.",
        )
        baseline_inner = tk.Frame(baseline_card, bg=COLORS["bg_surface"])
        baseline_inner.pack(fill=tk.X)

        # Baseline image
        self._explorer_baseline_holder = tk.Frame(baseline_inner, bg="#000000",
                                                   width=512, height=512)
        self._explorer_baseline_holder.pack(side=tk.LEFT, padx=(0, 16), pady=4)
        self._explorer_baseline_holder.pack_propagate(False)
        self._explorer_baseline_label = tk.Label(self._explorer_baseline_holder,
                                                  text="(no baseline yet)",
                                                  fg=COLORS["text_muted"], bg="#000000")
        self._explorer_baseline_label.pack(expand=True)

        # Baseline info + buttons
        baseline_right = tk.Frame(baseline_inner, bg=COLORS["bg_surface"])
        baseline_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        btn_row = tk.Frame(baseline_right, bg=COLORS["bg_surface"])
        btn_row.pack(fill=tk.X, pady=(4, 8))
        btn_row.columnconfigure(0, weight=2)
        btn_row.columnconfigure(1, weight=1)
        btn_row.columnconfigure(2, weight=1)
        self._explorer_save_btn = ttk.Button(btn_row, text="Save Baseline as LoRA...",
                                              command=self._explorer_save, state="disabled")
        self._explorer_save_btn.grid(row=0, column=0, sticky=tk.EW, padx=(0, 4))
        self._explorer_undo_btn = ttk.Button(btn_row, text="Undo",
                                              command=self._explorer_undo, state="disabled")
        self._explorer_undo_btn.grid(row=0, column=1, sticky=tk.EW, padx=4)
        ttk.Button(btn_row, text="Restart",
                   command=self._explorer_restart).grid(row=0, column=2, sticky=tk.EW, padx=(4, 0))

        handoff_row = tk.Frame(baseline_right, bg=COLORS["bg_surface"])
        handoff_row.pack(fill=tk.X, pady=(0, 8))
        handoff_row.columnconfigure(0, weight=1)
        handoff_row.columnconfigure(1, weight=1)
        self._explorer_freeze_btn = tk.Button(
            handoff_row, text="Freeze tweaked blocks",
            font=(FONT_FAMILY, 10, "bold"),
            fg="#FFFFFF", bg=COLORS["accent"], activeforeground="#FFFFFF",
            activebackground=COLORS["accent_hover"],
            relief="flat", bd=0, padx=16, pady=6, cursor="hand2", state="disabled",
            command=self._explorer_freeze_tweaked)
        self._explorer_freeze_btn.grid(row=0, column=0, sticky=tk.EW, padx=(0, 4))
        self._explorer_refine_btn = tk.Button(
            handoff_row, text="Refine this baseline in Repair Studio \u2192",
            font=(FONT_FAMILY, 10, "bold"),
            fg="#FFFFFF", bg="#2E8B57", activeforeground="#FFFFFF", activebackground="#256F46",
            relief="flat", bd=0, padx=16, pady=6, cursor="hand2", state="disabled",
            command=self._explorer_refine_in_repair)
        self._explorer_refine_btn.grid(row=0, column=1, sticky=tk.EW, padx=(4, 0))

        # Collapsed slider state display
        state_frame = tk.Frame(baseline_right, bg=COLORS["bg_deep"])
        state_frame.pack(anchor=tk.W, fill=tk.BOTH, expand=True, pady=(0, 4))
        self._explorer_state_text = tk.Text(state_frame, height=14, width=60,
                                             bg=COLORS["bg_deep"], fg=COLORS["text_secondary"],
                                             font=(FONT_FAMILY, 8), wrap="word",
                                             state="disabled", relief="flat",
                                             highlightthickness=1,
                                             highlightbackground=COLORS["border"])
        state_scroll = ttk.Scrollbar(state_frame, orient="vertical",
                                      command=self._explorer_state_text.yview)
        self._explorer_state_text.configure(yscrollcommand=state_scroll.set)
        self._explorer_state_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        state_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Card 3: Gallery
        gallery_card = self._start_section_card(
            outer, "Variants",
            "4 random mutations of the current baseline. Click your favourite to evolve.",
        )

        gallery_btn_row = tk.Frame(gallery_card, bg=COLORS["bg_surface"])
        gallery_btn_row.pack(anchor=tk.W, pady=(0, 8))
        self._explorer_roll_btn = ttk.Button(gallery_btn_row, text="Re-roll",
                                              command=self._explorer_reroll, state="disabled")
        self._explorer_roll_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._explorer_progress_var = tk.StringVar(value="")
        tk.Label(gallery_btn_row, textvariable=self._explorer_progress_var,
                 font=(FONT_FAMILY, 10, "bold"),
                 fg=COLORS["accent_hover"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)

        self._explorer_gallery_frame = tk.Frame(gallery_card, bg=COLORS["bg_surface"])
        self._explorer_gallery_frame.pack(fill=tk.X)

        # 2x2 grid of clickable images
        self._explorer_gallery_labels = []
        for row_idx in range(2):
            for col_idx in range(2):
                idx = row_idx * 2 + col_idx
                holder = tk.Frame(self._explorer_gallery_frame, bg="#000000",
                                  highlightbackground=COLORS["border"], highlightthickness=2)
                holder.grid(row=row_idx, column=col_idx, padx=6, pady=6, sticky=tk.NSEW)
                lbl = tk.Label(holder, text=f"(variant {idx + 1})",
                               fg=COLORS["text_muted"], bg="#000000", cursor="hand2")
                lbl.pack(expand=True, fill=tk.BOTH)
                lbl.bind("<Button-1>", lambda e, i=idx: self._explorer_pick(i))
                lbl.bind("<Enter>", lambda e, h=holder: h.configure(highlightbackground=COLORS["accent"]))
                lbl.bind("<Leave>", lambda e, h=holder: h.configure(highlightbackground=COLORS["border"]))
                # Seed cycle button overlaid in top-right corner
                seed_btn = tk.Button(holder, text="\u21bb", font=(FONT_FAMILY, 10, "bold"),
                                     bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
                                     activebackground=COLORS["bg_surface"],
                                     activeforeground=COLORS["text_primary"],
                                     relief="flat", bd=0, padx=4, pady=2, cursor="hand2",
                                     command=lambda: self._explorer_cycle_seed())
                seed_btn.place(relx=1.0, x=-4, y=4, anchor="ne")
                self._explorer_gallery_labels.append(lbl)
        self._explorer_gallery_frame.columnconfigure(0, weight=1)
        self._explorer_gallery_frame.columnconfigure(1, weight=1)

        self._add_youtube_help_button(outer, "explorer")

    # ------------------------------------------------------------------
    # Explorer actions
    # ------------------------------------------------------------------

    def _explorer_ensure_engine(self):
        """Lazy-load engine + pipeline for the Explorer. Returns True on success."""
        if self._explorer_engine is not None and self._explorer_engine.pipeline is not None and self._explorer_engine.pipeline.is_loaded:
            return True

        dit_choice = self.explorer_dit_var.get()
        dit_pref_key = "base_dit" if dit_choice == "base" else "distilled_dit"
        dit_path = self.prefs_vars[dit_pref_key].get() if dit_pref_key in self.prefs_vars else ""
        vae_path = self._get_path("VAE_MODEL")
        te_path = self._get_path("TEXT_ENCODER")

        for path, name in [(dit_path, "DiT"), (vae_path, "VAE"), (te_path, "Text Encoder")]:
            if not path or not os.path.exists(path):
                messagebox.showerror("Error", f"{name} not found:\n{path}\n\nCheck Preferences tab.")
                return False

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        from fizgig.repair_studio.engine import RepairEngine

        if self._explorer_engine is None:
            self._explorer_engine = RepairEngine()
        self._explorer_engine._turbo_enabled = True  # always use Turbo for Explorer

        dit_basename = os.path.basename(dit_path).lower()
        model_version = "klein-base-9b" if "base" in dit_basename else "klein-9b"
        is_fp8_model = "fp8" in dit_basename
        try:
            self.explorer_status_var.set(f"Loading models ({model_version})...")
            self.master.update_idletasks()
            self._explorer_engine.ensure_pipeline(
                dit_path=dit_path, vae_path=vae_path, text_encoder_path=te_path,
                model_version=model_version, device="cuda",
                fp8_scaled=False if is_fp8_model else True,
                blocks_to_swap=self._get_inference_blocks_to_swap(),
            )
            self.explorer_status_var.set("Models loaded.")
            return True
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Failed to load models:\n{traceback.format_exc()}")
            self.explorer_status_var.set("Error loading models.")
            return False

    def _explorer_load_lora(self):
        path = self.explorer_lora_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Pick a valid LoRA file first.")
            return
        if not self._explorer_ensure_engine():
            return
        try:
            self.explorer_status_var.set("Loading LoRA...")
            self.master.update_idletasks()
            # Reset if something was already loaded
            if self._explorer_engine.primary_network is not None:
                self._explorer_engine.reset()
                self._explorer_engine = None
                if not self._explorer_ensure_engine():
                    return
            self._explorer_engine.load_primary(path)
            n_active = len(self._explorer_engine.primary_block_ids)
            # Detect format for user info
            from safetensors.torch import load_file as _lf
            from fizgig.networks.lora import ensure_kohya_lora_state_dict as _ek, detect_lora_format as _df
            _fmt = _df(_ek(_lf(path)))
            if _fmt in ("lokr", "loha"):
                messagebox.showinfo("LyCORIS LoRA loaded",
                    f"This is a {_fmt.upper()} LoRA (LyCORIS format). "
                    f"Preview and profiling work normally.\n\n"
                    f"If you save, each block will be converted to standard LoRA via SVD. "
                    f"This may take a minute or two for large LoRAs (GPU-accelerated when available). "
                    f"The result is a slight approximation of the original.")
            self.explorer_status_var.set(
                f"Loaded: {os.path.basename(path)} ({n_active}/32 blocks). Click Re-roll to start exploring.")
            # Initialize baseline state with user-specified LoRA strength
            from fizgig.repair_studio.state import SliderState
            self._explorer_baseline_state = SliderState.default_klein9b()
            try:
                base_strength = float(self.explorer_strength_var.get())
            except ValueError:
                base_strength = 1.0
            for bid, bs in self._explorer_baseline_state.blocks.items():
                bs.primary_strength = base_strength
            self._explorer_baseline_state.prompt = self.explorer_prompt_var.get()
            self._explorer_baseline_state.seed = int(self.explorer_seed_var.get() or 42)
            res = int(self.explorer_res_var.get() or 512)
            self._explorer_baseline_state.preview_width = res
            self._explorer_baseline_state.preview_height = res
            self._explorer_history.clear()
            self._explorer_locked_blocks.clear()
            self._explorer_last_pick_blocks.clear()

            self._explorer_baseline_image = None
            self._explorer_undo_btn.configure(state="disabled")
            self._explorer_save_btn.configure(state="disabled")
            self._explorer_refine_btn.configure(state="disabled")
            self._explorer_freeze_btn.configure(state="disabled")
            self._explorer_roll_btn.configure(state="normal")
            # Generate initial baseline image
            self._explorer_generate_baseline_and_roll()
        except Exception as ex:
            from fizgig.networks.lora import UnsupportedLoRAFormat
            if isinstance(ex, UnsupportedLoRAFormat):
                messagebox.showerror("Unsupported LoRA format", str(ex))
            else:
                import traceback
                messagebox.showerror("Error", f"Failed to load LoRA:\n{traceback.format_exc()}")
            self.explorer_status_var.set("Error loading LoRA.")

    def _explorer_generate_baseline_and_roll(self):
        """Generate baseline image then 4 variants in a background thread."""
        if self._explorer_generating or self._explorer_engine is None:
            return
        self._explorer_generating = True
        self._explorer_roll_btn.configure(state="disabled")
        self._explorer_progress_var.set("Generating baseline...")
        self.master.update_idletasks()

        # Sync prompt/seed/res into baseline state
        state = self._explorer_baseline_state
        state.prompt = self.explorer_prompt_var.get()
        state.seed = int(self.explorer_seed_var.get() or 42)
        res = int(self.explorer_res_var.get() or 512)
        state.preview_width = res
        state.preview_height = res

        # Generate mutations (exclude locked blocks)
        active = self._explorer_engine.primary_block_ids - self._explorer_locked_blocks
        # double_0 as structural anchor — only if not explicitly frozen
        if "double_0" in self._explorer_engine.primary_block_ids and "double_0" not in self._explorer_locked_blocks:
            active.add("double_0")
        intensity = self.explorer_intensity_var.get()
        structure = self.explorer_structure_var.get()
        n_muts = int(self.explorer_mutations_var.get() or 5)
        # Variant 1 & 2: structural (double_0 anchor)
        # Variant 3: pure random
        # Variant 4: random but avoids last pick's blocks (protects recent changes)
        variant_states = []
        for vi in range(4):
            vs_structure = structure if vi < 2 else 0.0
            if vi == 3 and self._explorer_last_pick_blocks:
                # Variant 4: exclude last pick's changed blocks
                protected_active = active - self._explorer_last_pick_blocks
                if len(protected_active) < 2:
                    protected_active = active  # fallback if too few left
                vs = state.mutate(protected_active, num_mutations=n_muts,
                                  intensity=intensity, structure=vs_structure)
            else:
                vs = state.mutate(active, num_mutations=n_muts, intensity=intensity,
                                  structure=vs_structure)
            variant_states.append(vs)

        import threading
        thread = threading.Thread(
            target=self._explorer_worker,
            args=(state, variant_states),
            daemon=True,
        )
        thread.start()

    def _explorer_worker(self, baseline_state, variant_states):
        """Background: generate baseline + 4 variants."""
        try:
            engine = self._explorer_engine
            if engine is None:
                return

            # Generate baseline (full forward, populates activation cache)
            engine._changed_blocks = set(baseline_state.blocks.keys())
            baseline_img = engine.generate_preview(baseline_state)

            # Generate 4 variants — each runs a full forward (invalidate activation
            # cache between variants so they don't contaminate each other).
            # The prompt cache is still shared, saving ~300-500ms per variant.
            variant_images = []
            for i, vs in enumerate(variant_states):
                self.master.after(0, lambda i=i: self._explorer_progress_var.set(
                    f"Generating variant {i + 1}/4..."))
                engine._invalidate_activation_cache()
                engine._changed_blocks = set(vs.blocks.keys())
                img = engine.generate_preview(vs)
                variant_images.append(img)

            self.master.after(0, lambda: self._explorer_on_results(
                baseline_state, baseline_img, variant_states, variant_images))
        except Exception:
            import traceback
            err = traceback.format_exc()
            self.master.after(0, lambda: self._explorer_on_error(err))

    def _explorer_on_results(self, baseline_state, baseline_img, variant_states, variant_images):
        """Main-thread callback: update UI with results."""
        self._explorer_baseline_state = baseline_state
        self._explorer_baseline_image = baseline_img
        self._explorer_variant_states = variant_states
        self._explorer_variant_images = variant_images
        self._explorer_generating = False

        # Update baseline display
        self._explorer_show_baseline(baseline_img)
        self._explorer_update_state_text(baseline_state)

        # Update gallery
        for i, img in enumerate(variant_images):
            self._explorer_show_variant(i, img)

        self._explorer_save_btn.configure(state="normal")
        self._explorer_refine_btn.configure(state="normal")
        self._explorer_freeze_btn.configure(state="normal")
        # Update freeze button appearance based on locked state
        if self._explorer_locked_blocks:
            self._explorer_freeze_btn.configure(bg=COLORS["accent_hover"])
        else:
            self._explorer_freeze_btn.configure(bg=COLORS["accent"])
        self._explorer_progress_var.set("")

        # Check if all blocks are frozen
        active = self._explorer_engine.primary_block_ids if self._explorer_engine else set()
        unlocked = active - self._explorer_locked_blocks
        if not unlocked:
            self._explorer_roll_btn.configure(state="disabled")
            self.explorer_status_var.set(
                f"All {len(active)} blocks frozen! Save your LoRA, or click Freeze to unlock.")
        else:
            self._explorer_roll_btn.configure(state="normal")
            locked_msg = f" ({len(self._explorer_locked_blocks)} frozen)" if self._explorer_locked_blocks else ""
            self.explorer_status_var.set(f"Pick a favourite or re-roll.{locked_msg}")

    def _explorer_on_error(self, err):
        self._explorer_generating = False
        self._explorer_roll_btn.configure(state="normal")
        self._explorer_progress_var.set("")
        self.explorer_status_var.set("Error — see console.")
        print(err)

    def _explorer_show_baseline(self, pil_img):
        """Display a PIL image in the baseline holder."""
        from PIL import ImageTk
        holder = self._explorer_baseline_holder
        w, h = holder.winfo_width(), holder.winfo_height()
        if w < 10:
            w, h = 512, 512
        resized = pil_img.copy()
        resized.thumbnail((w, h))
        tk_img = ImageTk.PhotoImage(resized)
        self._explorer_thumbnails["baseline"] = tk_img
        self._explorer_baseline_label.configure(image=tk_img, text="")

    def _explorer_show_variant(self, idx, pil_img):
        """Display a PIL image in gallery slot idx (0-3)."""
        from PIL import ImageTk
        lbl = self._explorer_gallery_labels[idx]
        parent = lbl.master
        w = parent.winfo_width()
        if w < 10:
            w = 256
        resized = pil_img.copy()
        resized.thumbnail((w, w))
        tk_img = ImageTk.PhotoImage(resized)
        self._explorer_thumbnails[f"variant_{idx}"] = tk_img
        lbl.configure(image=tk_img, text="")

    def _explorer_update_state_text(self, state):
        """Show the baseline's slider state as read-only text, with lock indicators."""
        lines = []
        for bid in sorted(state.blocks.keys(),
                          key=lambda b: (b.split("_")[0], int(b.split("_")[1]))):
            bs = state.blocks[bid]
            if not bs.primary_enabled or bs.primary_strength != 1.0:
                en = "ON" if bs.primary_enabled else "OFF"
                lock = " [LOCKED]" if bid in self._explorer_locked_blocks else ""
                lines.append(f"{bid}: {en} @ {bs.primary_strength:+.2f}{lock}")
        if not lines:
            lines = ["All blocks at default (1.0)"]
        self._explorer_state_text.configure(state="normal")
        self._explorer_state_text.delete("1.0", tk.END)
        self._explorer_state_text.insert("1.0", "\n".join(lines))
        self._explorer_state_text.configure(state="disabled")

    def _explorer_pick(self, idx):
        """User picked variant idx as the new baseline."""
        if self._explorer_generating or idx >= len(self._explorer_variant_images):
            return
        # Push current baseline to undo stack
        if self._explorer_baseline_state is not None and self._explorer_baseline_image is not None:
            self._explorer_history.append(
                (self._explorer_baseline_state.copy(), self._explorer_baseline_image,
                 set(self._explorer_locked_blocks)))
            self._explorer_undo_btn.configure(state="normal")

        picked_state = self._explorer_variant_states[idx]

        # Track which blocks changed in this pick (for variant 4 protection)
        if self._explorer_baseline_state is not None:
            self._explorer_last_pick_blocks = set(picked_state.diff_blocks(self._explorer_baseline_state))
        else:
            self._explorer_last_pick_blocks = set()

        # New baseline = the picked variant
        self._explorer_baseline_state = picked_state
        self._explorer_baseline_image = self._explorer_variant_images[idx]

        locked_msg = f" ({len(self._explorer_locked_blocks)} blocks locked)" if self._explorer_locked_blocks else ""
        self._explorer_show_baseline(self._explorer_baseline_image)
        self._explorer_update_state_text(self._explorer_baseline_state)
        self.explorer_status_var.set(
            f"Variant {idx + 1} selected as new baseline{locked_msg}. Generating new mutations...")

        # Roll new variants from the new baseline
        self._explorer_generate_baseline_and_roll()

    def _explorer_cycle_seed(self):
        """New random seed, regenerate all variants + baseline with current slider states."""
        if self._explorer_generating or self._explorer_baseline_state is None:
            return
        import random
        new_seed = random.randint(1, 99999)
        self.explorer_seed_var.set(str(new_seed))
        # Update baseline and all variant states to the new seed
        self._explorer_baseline_state.seed = new_seed
        for vs in self._explorer_variant_states:
            vs.seed = new_seed
        # Regenerate with same slider states but new seed
        self._explorer_generating = True
        self._explorer_roll_btn.configure(state="disabled")
        self._explorer_progress_var.set("Cycling seed...")
        self.master.update_idletasks()

        import threading
        thread = threading.Thread(
            target=self._explorer_seed_cycle_worker,
            args=(self._explorer_baseline_state, list(self._explorer_variant_states)),
            daemon=True,
        )
        thread.start()

    def _explorer_seed_cycle_worker(self, baseline_state, variant_states):
        """Background: regenerate baseline + existing variants at a new seed."""
        try:
            engine = self._explorer_engine
            if engine is None:
                return
            # Generate baseline at new seed
            engine._invalidate_activation_cache()
            engine._changed_blocks = set(baseline_state.blocks.keys())
            baseline_img = engine.generate_preview(baseline_state)
            # Generate each variant at new seed (same slider states)
            variant_images = []
            for i, vs in enumerate(variant_states):
                self.master.after(0, lambda i=i: self._explorer_progress_var.set(
                    f"Seed cycling variant {i + 1}/4..."))
                engine._invalidate_activation_cache()
                engine._changed_blocks = set(vs.blocks.keys())
                img = engine.generate_preview(vs)
                variant_images.append(img)
            self.master.after(0, lambda: self._explorer_on_results(
                baseline_state, baseline_img, variant_states, variant_images))
        except Exception:
            import traceback
            self.master.after(0, lambda: self._explorer_on_error(traceback.format_exc()))

    def _explorer_randomize_seed(self):
        """Randomize seed and regenerate (same as Apply but for seed)."""
        import random
        self.explorer_seed_var.set(str(random.randint(1, 99999)))
        if self._explorer_baseline_state is not None and not self._explorer_generating:
            self._explorer_generate_baseline_and_roll()

    def _explorer_start(self):
        """Start button: load LoRA if not loaded, or regenerate with current settings."""
        if self._explorer_generating:
            return
        if self._explorer_engine is None or self._explorer_engine.primary_network is None:
            # Not loaded yet — load the LoRA
            self._explorer_load_lora()
        else:
            # Already loaded — invalidate prompt cache and regenerate
            self._explorer_apply_prompt()

    def _explorer_apply_prompt(self):
        """Apply a new prompt — invalidates prompt cache and regenerates."""
        if self._explorer_generating or self._explorer_baseline_state is None:
            return
        if self._explorer_engine is not None:
            self._explorer_engine._prompt_cache_key = None
            self._explorer_engine._prompt_cache = None
        self._explorer_generate_baseline_and_roll()

    def _explorer_reroll(self):
        """Re-roll: generate 4 new mutations from the same baseline."""
        if self._explorer_generating or self._explorer_baseline_state is None:
            return
        self._explorer_generate_baseline_and_roll()

    def _explorer_undo(self):
        """Pop the history stack and restore the previous baseline + locked blocks."""
        if not self._explorer_history or self._explorer_generating:
            return
        prev_state, prev_img, prev_locked = self._explorer_history.pop()
        self._explorer_baseline_state = prev_state
        self._explorer_baseline_image = prev_img
        self._explorer_locked_blocks = prev_locked
        self._explorer_show_baseline(prev_img)
        self._explorer_update_state_text(prev_state)
        self._explorer_roll_btn.configure(state="normal")
        self._explorer_freeze_btn.configure(
            bg=COLORS["accent_hover"] if self._explorer_locked_blocks else COLORS["accent"])
        if not self._explorer_history:
            self._explorer_undo_btn.configure(state="disabled")
        locked_msg = f" ({len(self._explorer_locked_blocks)} frozen)" if self._explorer_locked_blocks else ""
        self.explorer_status_var.set(f"Undone{locked_msg}. Re-rolling...")
        self._explorer_generate_baseline_and_roll()

    def _explorer_restart(self):
        """Unlock all blocks and restart exploration — ask whether from defaults or current baseline."""
        if self._explorer_generating or self._explorer_baseline_state is None:
            return
        choice = messagebox.askyesnocancel(
            "Restart Exploration",
            "Unlock all blocks and restart.\n\n"
            "Yes = restart from default values\n"
            "No = restart from current baseline (keep slider positions)\n"
            "Cancel = don't restart",
        )
        if choice is None:
            return  # Cancel

        # Push current state to undo stack
        self._explorer_history.append(
            (self._explorer_baseline_state.copy(), self._explorer_baseline_image,
             set(self._explorer_locked_blocks)))
        self._explorer_undo_btn.configure(state="normal")
        self._explorer_locked_blocks.clear()
        self._explorer_walk_index = 0

        if choice:
            # Yes = reset to default values
            from fizgig.repair_studio.state import SliderState
            self._explorer_baseline_state = SliderState.default_klein9b()
            try:
                base_strength = float(self.explorer_strength_var.get())
            except ValueError:
                base_strength = 1.0
            for bid, bs in self._explorer_baseline_state.blocks.items():
                bs.primary_strength = base_strength
            self._explorer_baseline_state.prompt = self.explorer_prompt_var.get()
            self._explorer_baseline_state.seed = int(self.explorer_seed_var.get() or 42)
            res = int(self.explorer_res_var.get() or 512)
            self._explorer_baseline_state.preview_width = res
            self._explorer_baseline_state.preview_height = res
            self.explorer_status_var.set("Restarted from defaults — all blocks unlocked.")
        else:
            # No = keep current baseline, just unlock
            self.explorer_status_var.set("All blocks unlocked — continuing from current baseline.")

        self._explorer_roll_btn.configure(state="normal")
        self._explorer_generate_baseline_and_roll()

    def _explorer_full_reset(self):
        """Unload LoRA and pipeline, return to initial state."""
        if self._explorer_generating:
            return
        if self._explorer_engine is not None:
            try:
                self._explorer_engine.reset()
            except Exception:
                pass
            self._explorer_engine = None
        self._explorer_baseline_state = None
        self._explorer_baseline_image = None
        self._explorer_history.clear()
        self._explorer_locked_blocks.clear()
        self._explorer_walk_index = 0
        self._explorer_variant_states.clear()
        self._explorer_variant_images.clear()
        self._explorer_thumbnails.clear()
        self._explorer_baseline_label.configure(image="", text="(no baseline yet)")
        for lbl in self._explorer_gallery_labels:
            lbl.configure(image="", text="(variant)")
        self._explorer_roll_btn.configure(state="disabled")
        self._explorer_save_btn.configure(state="disabled")
        self._explorer_undo_btn.configure(state="disabled")
        self._explorer_progress_var.set("")
        self.explorer_status_var.set("Load a LoRA to begin exploring.")

    def _explorer_freeze_tweaked(self):
        """Freeze all blocks that differ from default — they won't be mutated on re-roll."""
        if self._explorer_baseline_state is None or self._explorer_generating:
            return

        # Find blocks that differ from default (strength != starting strength)
        try:
            base_strength = float(self.explorer_strength_var.get())
        except ValueError:
            base_strength = 1.0

        tweaked = set()
        for bid, bs in self._explorer_baseline_state.blocks.items():
            if not bs.primary_enabled or abs(bs.primary_strength - base_strength) > 0.01:
                tweaked.add(bid)
        # When explicitly freezing, include double_0 — stops the structural anchor behaviour

        if self._explorer_locked_blocks:
            # Already have frozen blocks — ask what to do
            active = self._explorer_engine.primary_block_ids if self._explorer_engine else set()
            unlocked = active - self._explorer_locked_blocks - {"double_0"}

            if not unlocked:
                # All locked — offer unlock all or undo last
                choice = messagebox.askyesnocancel(
                    "All blocks frozen",
                    f"All {len(active)} blocks are already frozen.\n\n"
                    "Yes = Unlock all blocks\n"
                    "No = Undo last freeze (restore previous unlocked set)\n"
                    "Cancel = Keep as is",
                )
                if choice is True:
                    self._explorer_locked_blocks.clear()
                    self._explorer_freeze_btn.configure(bg=COLORS["accent"])
                    self._explorer_roll_btn.configure(state="normal")
                    self._explorer_update_state_text(self._explorer_baseline_state)
                    self.explorer_status_var.set("All blocks unlocked. Re-rolling...")
                    self._explorer_generate_baseline_and_roll()
                elif choice is False and hasattr(self, "_explorer_prev_locked"):
                    self._explorer_locked_blocks = self._explorer_prev_locked
                    if hasattr(self, "_explorer_prev_baseline") and self._explorer_prev_baseline is not None:
                        self._explorer_baseline_state = self._explorer_prev_baseline
                    self._explorer_freeze_btn.configure(bg=COLORS["accent_hover"] if self._explorer_locked_blocks else COLORS["accent"])
                    self._explorer_roll_btn.configure(state="normal")
                    self._explorer_show_baseline(self._explorer_baseline_image)
                    self._explorer_update_state_text(self._explorer_baseline_state)
                    n = len(self._explorer_locked_blocks)
                    self.explorer_status_var.set(f"Last freeze undone. {n} blocks frozen. Re-rolling...")
                    self._explorer_generate_baseline_and_roll()
                return
            else:
                # Some locked, some not — ask to lock additions or unlock all
                choice = messagebox.askyesnocancel(
                    "Freeze tweaked blocks",
                    f"Currently {len(self._explorer_locked_blocks)} blocks frozen.\n"
                    f"{len(tweaked - self._explorer_locked_blocks)} new tweaked blocks to add.\n\n"
                    "Yes = Lock the additions too\n"
                    "No = Unlock all\n"
                    "Cancel = Keep as is",
                )
                if choice is True:
                    self._explorer_prev_locked = set(self._explorer_locked_blocks)
                    self._explorer_prev_baseline = self._explorer_baseline_state.copy()
                    self._explorer_locked_blocks |= tweaked
                elif choice is False:
                    self._explorer_prev_locked = set(self._explorer_locked_blocks)
                    self._explorer_prev_baseline = self._explorer_baseline_state.copy()
                    self._explorer_locked_blocks.clear()
                    self._explorer_freeze_btn.configure(bg=COLORS["accent"])
                else:
                    return
        else:
            # No existing frozen blocks — freeze the tweaked ones
            if not tweaked:
                messagebox.showinfo("Nothing to freeze",
                    "No blocks have been changed from their starting values yet.")
                return
            self._explorer_prev_locked = set()
            self._explorer_prev_baseline = self._explorer_baseline_state.copy()
            self._explorer_locked_blocks = tweaked

        # Update UI
        if self._explorer_locked_blocks:
            self._explorer_freeze_btn.configure(bg=COLORS["accent_hover"])
        else:
            self._explorer_freeze_btn.configure(bg=COLORS["accent"])

        # Check if all blocks now locked (Freeze can lock double_0 too)
        active = self._explorer_engine.primary_block_ids if self._explorer_engine else set()
        unlocked = active - self._explorer_locked_blocks
        if not unlocked:
            self._explorer_roll_btn.configure(state="disabled")
            self.explorer_status_var.set(
                f"All {len(active)} blocks frozen! Save your LoRA, Restart, or click Freeze to unlock.")
        else:
            n = len(self._explorer_locked_blocks)
            self.explorer_status_var.set(f"{n} blocks frozen. {len(unlocked)} still explorable.")

        self._explorer_update_state_text(self._explorer_baseline_state)

        # Re-roll variants to respect the new freeze state
        if unlocked:
            self._explorer_generate_baseline_and_roll()

    def _explorer_refine_in_repair(self):
        """Send the current Explorer baseline to the Repair Studio for manual editing."""
        if self._explorer_engine is None or self._explorer_baseline_state is None:
            return
        lora_path = self._explorer_engine.primary_path
        if not lora_path:
            return

        # Warn if LyCORIS — saving will require SVD materialization
        try:
            from safetensors.torch import load_file as _lf
            from fizgig.networks.lora import ensure_kohya_lora_state_dict as _ek, detect_lora_format as _df
            _fmt = _df(_ek(_lf(lora_path)))
            if _fmt in ("lokr", "loha"):
                proceed = messagebox.askyesno(
                    "LyCORIS LoRA",
                    f"This is a {_fmt.upper()} LoRA. Preview and editing work normally, "
                    f"but saving will require SVD conversion (may take a minute).\n\n"
                    f"Consider using the Extract tab to convert to standard LoRA first "
                    f"for faster saves.\n\nContinue anyway?")
                if not proceed:
                    return
        except Exception:
            pass

        baseline = self._explorer_baseline_state

        # Set the LoRA path in Repair Studio
        self.repair_primary_var.set(lora_path)

        # Unload Explorer engine to free VRAM
        self._unload_explorer_models()

        # Switch to Repair Studio tab
        self.notebook.select(self.repair_studio_tab)

        # Load the LoRA in Repair Studio
        if not self._ensure_repair_engine():
            return
        try:
            self.repair_status_var.set("Loading LoRA from Explorer baseline...")
            self.master.update_idletasks()
            self.repair_engine.load_primary(lora_path)
            self._refresh_block_slider_activity()

            # Push the Explorer baseline slider values into Repair Studio
            self._repair_master_mutating = True
            try:
                for bid, bs in baseline.blocks.items():
                    if bid in self.repair_block_vars:
                        self.repair_block_vars[bid]["primary_enabled"].set(bs.primary_enabled)
                        self.repair_block_vars[bid]["primary_strength"].set(bs.primary_strength)
            finally:
                self._repair_master_mutating = False

            # Set the prompt, seed, and resolution to match Explorer
            self.repair_prompt_var.set(baseline.prompt)
            self.repair_seed_var.set(str(baseline.seed))
            self.repair_res_var.set(str(baseline.preview_width))

            n_active = len(self.repair_engine.primary_block_ids)
            self._find_repair_profile_match()
            self.repair_status_var.set(
                f"Loaded from Explorer: {os.path.basename(lora_path)} ({n_active}/32 blocks). "
                f"Sliders set to Explorer baseline. Generating preview...")
            self._schedule_preview(force=True)
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Failed to load in Repair Studio:\n{traceback.format_exc()}")

    def _explorer_save(self):
        """Save the current baseline as a baked LoRA."""
        if self._explorer_engine is None or self._explorer_baseline_state is None:
            return
        primary_path = self._explorer_engine.primary_path
        if not primary_path:
            return
        stem = os.path.splitext(os.path.basename(primary_path))[0]
        default_name = f"{stem}_explored.safetensors"
        out = filedialog.asksaveasfilename(
            title="Save Explored LoRA",
            defaultextension=".safetensors",
            filetypes=[("SafeTensors", "*.safetensors")],
            initialfile=default_name,
        )
        if not out:
            return
        from fizgig.repair_studio.bake import save_repaired_lora
        from fizgig.networks.lora import UnsupportedLoRAFormat
        try:
            summary = save_repaired_lora(primary_path, self._explorer_baseline_state, out)
            messagebox.showinfo("Explored LoRA saved",
                                f"Saved: {out}\n\nKeys: {summary['keys_in']} -> {summary['keys_out']}")
        except UnsupportedLoRAFormat as ex:
            messagebox.showerror("Unsupported LoRA format", str(ex))
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Save failed:\n{traceback.format_exc()}")

    def _unload_explorer_models(self):
        """Unload Explorer pipeline when leaving the tab."""
        if self._explorer_engine is not None and self._explorer_engine.pipeline is not None:
            try:
                self._explorer_engine.reset()
            except Exception:
                pass
            self._explorer_engine = None
            self.explorer_status_var.set("Models unloaded (tab switch). Load a LoRA to resume.")

    # endregion

    # region Extract Tab

    def create_extract_tab(self):
        """Create the Extract tab (Start-tab styled) — activation-based LoRA extraction."""
        scrollable_frame, _ = self.create_scrollable_frame(self.extract_tab)

        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer,
            "Extract",
            "Distill an existing LoRA down to a lower rank — optionally targeting specific blocks "
            "or timestep ranges. SVD runs per-block and typically takes a few minutes.",
        )

        # Card 1: Source & Output
        io_card = self._start_section_card(
            outer, "Source & Output",
            "Choose the source LoRA and name the extraction — it will land in your LoRA output folder.",
        )
        io_card.grid_columnconfigure(1, weight=1)

        ttk.Label(io_card, text="Source LoRA:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.extract_source_var = tk.StringVar()
        ttk.Entry(io_card, textvariable=self.extract_source_var, width=50).grid(row=0, column=1, sticky=tk.EW, pady=4)
        ttk.Button(io_card, text="Browse", command=self._browse_extract_source).grid(row=0, column=2, sticky=tk.W, padx=(8, 0), pady=4)

        ttk.Label(io_card, text="Output Name:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.extract_output_var = tk.StringVar()
        ttk.Entry(io_card, textvariable=self.extract_output_var, width=50).grid(row=1, column=1, sticky=tk.EW, pady=4)
        tk.Label(io_card, text="(will be saved in your LoRA output folder)",
                 font=(FONT_FAMILY, 9, "italic"),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=2, column=1, sticky=tk.W, pady=(0, 4)
        )

        # Card 2: Preset
        preset_card = self._start_section_card(
            outer, "Preset",
            "Fast * = pure weight SVD (samples=0, no GPU probes).  |  Identity = single 1-16.  |  "
            "Style = style+comp @ late timesteps.  |  Style+Composition = double 0-7 + single 0-1 + single 2 @ 0.5.  |  "
            "Details = single 12-23.",
        )
        preset_row = tk.Frame(preset_card, bg=COLORS["bg_surface"])
        preset_row.pack(anchor=tk.W)
        ttk.Label(preset_row, text="Extract Preset:").pack(side=tk.LEFT, padx=(0, 10))
        self.extract_preset_var = tk.StringVar(value="Identity")
        preset_combo = ttk.Combobox(
            preset_row, textvariable=self.extract_preset_var,
            values=["All Blocks", "Fast SVD", "Identity", "Fast Identity", "Style",
                    "Style+Composition", "Fast Style+Composition", "Details", "Fast Details", "Custom"],
            state="readonly", width=24,
        )
        preset_combo.pack(side=tk.LEFT)
        preset_combo.bind("<<ComboboxSelected>>", self._on_extract_preset_changed)

        # Card 3: Custom Blocks — pack-managed so pack_forget/pack drives visibility
        self._extract_custom_frame = tk.Frame(outer, bg=COLORS["bg_deep"])
        self._extract_custom_frame.pack(fill=tk.X, padx=36, pady=(0, 16))
        custom_card = tk.Frame(self._extract_custom_frame, bg=COLORS["bg_surface"],
                               highlightbackground=COLORS["border"], highlightthickness=1, bd=0)
        custom_card.pack(fill=tk.X)
        tk.Label(custom_card, text="Custom Blocks",
                 font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(anchor=tk.W, padx=20, pady=(16, 2))
        tk.Label(custom_card,
                 text="Pick individual blocks to target. Only shown when preset = Custom.",
                 font=(FONT_FAMILY, 9),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))

        custom_content = tk.Frame(custom_card, bg=COLORS["bg_surface"])
        custom_content.pack(fill=tk.X, padx=20, pady=(0, 16))

        self.extract_block_vars = {}

        header_row = tk.Frame(custom_content, bg=COLORS["bg_surface"])
        header_row.pack(anchor=tk.W, pady=(0, 6))
        tk.Label(header_row, text="Select individual blocks:",
                 font=(FONT_FAMILY, 10, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(header_row, text="All", width=5,
                   command=lambda: self._set_all_extract_blocks(True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(header_row, text="None", width=5,
                   command=lambda: self._set_all_extract_blocks(False)).pack(side=tk.LEFT, padx=2)
        ttk.Button(header_row, text="Identity", width=8,
                   command=lambda: self._set_category_extract_blocks("identity")).pack(side=tk.LEFT, padx=2)
        ttk.Button(header_row, text="Style+Comp", width=10,
                   command=lambda: self._set_category_extract_blocks("style_composition")).pack(side=tk.LEFT, padx=2)
        ttk.Button(header_row, text="Details", width=8,
                   command=lambda: self._set_category_extract_blocks("details")).pack(side=tk.LEFT, padx=2)

        double_row = tk.Frame(custom_content, bg=COLORS["bg_surface"])
        double_row.pack(anchor=tk.W, pady=2)
        tk.Label(double_row, text="Double:", width=8, fg="#5B9BD5", bg=COLORS["bg_surface"],
                 font=(FONT_FAMILY, 10)).pack(side=tk.LEFT)
        for i in range(8):
            key = f"double_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.extract_block_vars[key] = var
            ttk.Checkbutton(double_row, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        single_row1 = tk.Frame(custom_content, bg=COLORS["bg_surface"])
        single_row1.pack(anchor=tk.W, pady=2)
        tk.Label(single_row1, text="Single:", width=8, fg="#70AD47", bg=COLORS["bg_surface"],
                 font=(FONT_FAMILY, 10)).pack(side=tk.LEFT)
        for i in range(12):
            key = f"single_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.extract_block_vars[key] = var
            ttk.Checkbutton(single_row1, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        single_row2 = tk.Frame(custom_content, bg=COLORS["bg_surface"])
        single_row2.pack(anchor=tk.W, pady=2)
        tk.Label(single_row2, text="Single:", width=8, fg="#ED7D31", bg=COLORS["bg_surface"],
                 font=(FONT_FAMILY, 10)).pack(side=tk.LEFT)
        for i in range(12, 24):
            key = f"single_blocks.{i}"
            var = tk.BooleanVar(value=False)
            self.extract_block_vars[key] = var
            ttk.Checkbutton(single_row2, text=str(i), variable=var).pack(side=tk.LEFT, padx=2)

        tk.Label(custom_content,
                 text="double + single 0-1 = style+composition  |  single 1-16 = identity (overlaps at 1 and 12-16)  |  single 12-23 = details",
                 font=(FONT_FAMILY, 9, "italic"),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(anchor=tk.W, pady=(4, 0))

        self._extract_custom_frame.pack_forget()  # hidden until preset = Custom

        # Card 4: Options (rank / timesteps / forward passes)
        options_card = self._start_section_card(
            outer, "Options",
            "Timesteps: 'all' for general, 'late' for style, 'early' for composition.  |  "
            "Forward Passes: 0 = pure weight SVD (fastest, timestep-agnostic). Higher = better activation-weighted accuracy.",
        )
        # Anchor for the Custom Blocks card's re-pack so it reappears between Preset and Options.
        self._extract_options_anchor = options_card.master.master
        options_row = tk.Frame(options_card, bg=COLORS["bg_surface"])
        options_row.pack(anchor=tk.W)

        ttk.Label(options_row, text="Target Rank:").pack(side=tk.LEFT, padx=(0, 6))
        self.extract_rank_var = tk.StringVar(value="4")
        rank_combo = ttk.Combobox(options_row, textvariable=self.extract_rank_var,
                     values=["1", "2", "4", "8", "16"], state="readonly", width=4)
        rank_combo.pack(side=tk.LEFT, padx=(0, 20))
        rank_combo.bind("<<ComboboxSelected>>", lambda e: self._update_extract_output_name())

        ttk.Label(options_row, text="Timesteps:").pack(side=tk.LEFT, padx=(0, 6))
        self.extract_timesteps_var = tk.StringVar(value="all")
        self._extract_timesteps_combo = ttk.Combobox(
            options_row, textvariable=self.extract_timesteps_var,
            values=["all", "early", "mid", "late"], state="readonly", width=8,
        )
        self._extract_timesteps_combo.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(options_row, text="Forward Passes:").pack(side=tk.LEFT, padx=(0, 6))
        self.extract_samples_var = tk.StringVar(value="16")
        self._extract_samples_combo = ttk.Combobox(
            options_row, textvariable=self.extract_samples_var,
            values=["0", "8", "16", "32"], state="readonly", width=4,
        )
        self._extract_samples_combo.pack(side=tk.LEFT)
        self._extract_samples_combo.bind("<<ComboboxSelected>>", self._on_extract_samples_changed)

        # Card 5: Prompt
        prompt_card = self._start_section_card(
            outer, "Prompt",
            "Used during the GPU probe forward passes. Include the source LoRA's trigger word for best results.",
        )
        prompt_card.grid_columnconfigure(1, weight=1)
        ttk.Label(prompt_card, text="Prompt:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.extract_prompt_var = tk.StringVar(value="")
        ttk.Entry(prompt_card, textvariable=self.extract_prompt_var, width=50).grid(row=0, column=1, sticky=tk.EW, pady=4)

        # Card 6: Run
        run_card = self._start_section_card(
            outer, "Run",
            "Extraction runs SVD on each block and can take several minutes depending on rank and block count.",
        )
        run_row = tk.Frame(run_card, bg=COLORS["bg_surface"])
        run_row.pack(anchor=tk.W)
        self.extract_run_btn = ttk.Button(run_row, text="Extract LoRA", command=self._run_extract, style="Primary.TButton")
        self.extract_run_btn.pack(side=tk.LEFT, padx=(0, 12))
        self.extract_open_btn = ttk.Button(run_row, text="Open Output Folder", command=self._open_extract_folder, state="disabled")
        self.extract_open_btn.pack(side=tk.LEFT)

        self.extract_progress_var = tk.StringVar(value="")
        tk.Label(run_card, textvariable=self.extract_progress_var,
                 font=(FONT_FAMILY, 10, "bold"),
                 fg=COLORS["accent_hover"], bg=COLORS["bg_surface"]).pack(anchor=tk.W, pady=(10, 0))

        # Card 7: Output Log
        log_card = self._start_section_card(outer, "Output Log", None)
        self.extract_log = scrolledtext.ScrolledText(
            log_card, height=14, width=80,
            bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
            wrap="word", state="disabled",
            relief="flat", highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["border_focus"],
        )
        self.extract_log.pack(fill=tk.BOTH, expand=True)

        self._extract_output_path = None

        self._add_youtube_help_button(outer, "extract")

    def _on_training_preset_changed(self, *args):
        """Auto-fill MIN/MAX_TIMESTEP and show/hide custom block picker based on training preset."""
        preset = self.training_preset_var.get()

        # Show/hide custom block panel
        if preset == "Custom":
            self._training_custom_frame.grid()
        else:
            self._training_custom_frame.grid_remove()

        # Skip timestep auto-fill for Custom (user-driven)
        if preset == "Custom":
            return

        # Auto-fill MIN/MAX_TIMESTEP entries
        min_entry = self.entries.get("MIN_TIMESTEP")
        max_entry = self.entries.get("MAX_TIMESTEP")
        if min_entry is None or max_entry is None:
            return

        if preset == "Style":
            min_val, max_val = "0", "400"
        else:  # Full Model, Identity, Style+Composition, Details
            min_val, max_val = "", ""

        min_entry.delete(0, tk.END)
        if min_val:
            min_entry.insert(0, min_val)
        max_entry.delete(0, tk.END)
        if max_val:
            max_entry.insert(0, max_val)
        self.settings["MIN_TIMESTEP"] = min_val
        self.settings["MAX_TIMESTEP"] = max_val
        if hasattr(self, "_update_noise_range_label"):
            try:
                self._update_noise_range_label()
            except Exception:
                pass

    def _set_all_training_blocks(self, value: bool):
        """Tick/untick all per-block checkboxes in Custom training mode."""
        for var in self.training_block_vars.values():
            var.set(value)

    def _set_category_training_blocks(self, category: str):
        """Select all blocks in a category (identity/style_composition/details). Clears others."""
        for key, var in self.training_block_vars.items():
            var.set(False)
        if category == "style_composition":
            for i in range(8):
                self.training_block_vars[f"double_blocks.{i}"].set(True)
            for i in (0, 1):
                self.training_block_vars[f"single_blocks.{i}"].set(True)
        elif category == "identity":
            for i in range(1, 17):
                self.training_block_vars[f"single_blocks.{i}"].set(True)
        elif category == "details":
            for i in range(12, 24):
                self.training_block_vars[f"single_blocks.{i}"].set(True)

    def _build_custom_training_patterns(self):
        """Build a list of include_patterns regexes from the Custom block checkboxes."""
        selected = [key for key, var in self.training_block_vars.items() if var.get()]
        if not selected:
            return None
        patterns = []
        for key in selected:
            # key is "double_blocks.N" or "single_blocks.N"
            kind, idx = key.split(".")
            patterns.append(rf".*{kind}\.{idx}\..*")
        return patterns

    def _on_extract_preset_changed(self, *args):
        """Show/hide custom block checkboxes and auto-switch timesteps/samples based on preset."""
        preset = self.extract_preset_var.get()
        if preset == "Custom":
            # Pack before Options so the custom card appears between Preset and Options.
            self._extract_custom_frame.pack(fill=tk.X, padx=36, pady=(0, 16),
                                             before=self._extract_options_anchor)
        else:
            self._extract_custom_frame.pack_forget()

        # Fast * presets force samples=0 (pure weight SVD)
        is_fast = preset in ("Fast SVD", "Fast Identity", "Fast Style+Composition", "Fast Details")
        if is_fast:
            self.extract_samples_var.set("0")
        elif preset in ("All Blocks", "Identity", "Style", "Style+Composition", "Details"):
            # Activation-weighted presets need samples>0; bump off 0 if user is coming from a Fast variant
            if self.extract_samples_var.get() == "0":
                self.extract_samples_var.set("16")

        # Timestep auto-fill
        if preset == "Style":
            self.extract_timesteps_var.set("late")
        elif preset in ("All Blocks", "Fast SVD", "Identity", "Fast Identity",
                        "Style+Composition", "Fast Style+Composition", "Details", "Fast Details"):
            self.extract_timesteps_var.set("all")

        # Reflect timestep combo state (samples may have just changed above)
        self._apply_extract_samples_state()

        # Update suggested output filename to match new preset
        self._update_extract_output_name()

    def _on_extract_samples_changed(self, *args):
        """Grey out timesteps when samples=0; map presets to their Fast variant where applicable."""
        if self.extract_samples_var.get() == "0":
            preset = self.extract_preset_var.get()
            # Map activation-weighted presets to their Fast equivalent so the UI reflects reality
            fast_map = {
                "All Blocks": "Fast SVD",
                "Identity": "Fast Identity",
                "Style": "Fast Style+Composition",   # Style is inherently timestep-based; Fast loses that meaning
                "Style+Composition": "Fast Style+Composition",
                "Details": "Fast Details",
            }
            if preset in fast_map:
                self.extract_preset_var.set(fast_map[preset])
                self.extract_timesteps_var.set("all")
        self._apply_extract_samples_state()

    def _apply_extract_samples_state(self):
        """Timesteps dropdown is meaningful only when forward passes > 0."""
        if self.extract_samples_var.get() == "0":
            self._extract_timesteps_combo.configure(state="disabled")
        else:
            self._extract_timesteps_combo.configure(state="readonly")

    def _set_all_extract_blocks(self, value: bool):
        """Tick/untick all per-block checkboxes in Custom mode."""
        for var in self.extract_block_vars.values():
            var.set(value)

    def _set_category_extract_blocks(self, category: str):
        """Select all blocks in a category (identity/style_composition/details). Clears others."""
        for key, var in self.extract_block_vars.items():
            var.set(False)
        if category == "style_composition":
            # double 0-7 + single 0-1 + single 2 (at full strength in Custom mode)
            for i in range(8):
                self.extract_block_vars[f"double_blocks.{i}"].set(True)
            for i in (0, 1, 2):
                self.extract_block_vars[f"single_blocks.{i}"].set(True)
        elif category == "identity":
            for i in range(1, 17):
                self.extract_block_vars[f"single_blocks.{i}"].set(True)
        elif category == "details":
            for i in range(12, 24):
                self.extract_block_vars[f"single_blocks.{i}"].set(True)

    def _update_extract_output_name(self):
        """Regenerate the suggested output filename from source + preset + rank."""
        source = self.extract_source_var.get().strip()
        if not source:
            return
        base = os.path.splitext(os.path.basename(source))[0]
        preset_slug = self.extract_preset_var.get().lower().replace("+", "_").replace(" ", "_")
        self.extract_output_var.set(f"{base}_{preset_slug}_r{self.extract_rank_var.get()}.safetensors")

    def _browse_extract_source(self):
        filepath = filedialog.askopenfilename(
            title="Select source LoRA",
            filetypes=[("SafeTensors", "*.safetensors")]
        )
        if filepath:
            self.extract_source_var.set(filepath)
            self._update_extract_output_name()

    def _extract_log(self, text):
        """Append to extract log (preserves user scroll position)."""
        self._smart_text_insert(self.extract_log, text)

    def _open_extract_folder(self):
        if self._extract_output_path and os.path.exists(self._extract_output_path):
            folder = os.path.dirname(self._extract_output_path)
            self._open_in_file_manager(folder)

    def _run_extract(self):
        """Start extraction in a background thread."""
        source = self.extract_source_var.get()
        if not source or not os.path.exists(source):
            messagebox.showerror("Error", "Please select a valid source LoRA.")
            return

        output_name = self.extract_output_var.get().strip()
        if not output_name:
            messagebox.showerror("Error", "Please enter an output name.")
            return
        if not output_name.endswith(".safetensors"):
            output_name += ".safetensors"

        prompt = self.extract_prompt_var.get().strip()
        if not prompt:
            messagebox.showerror("Error", "Please enter a prompt (trigger word recommended).")
            return

        # Get paths from prefs
        dit_path = self.prefs_vars["distilled_dit"].get()
        vae_path = self.prefs_vars["vae"].get()
        te_path = self.prefs_vars["text_encoder"].get()

        for path, name in [(dit_path, "DiT"), (vae_path, "VAE"), (te_path, "Text Encoder")]:
            if not path or not os.path.exists(path):
                messagebox.showerror("Error", f"{name} not found:\n{path}\n\nCheck Preferences tab.")
                return

        # Build block list from preset (or custom individual blocks)
        preset = self.extract_preset_var.get()
        custom_blocks = None  # Per-block list for Custom mode
        if preset in ("Identity", "Fast Identity"):
            blocks = ["identity"]
        elif preset in ("Style", "Style+Composition", "Fast Style+Composition"):
            blocks = ["style_composition"]
        elif preset in ("Details", "Fast Details"):
            blocks = ["details"]
        elif preset in ("All Blocks", "Fast SVD"):
            blocks = ["all"]
        else:  # Custom
            selected = [key for key, var in self.extract_block_vars.items() if var.get()]
            if not selected:
                messagebox.showerror("Error", "Custom mode: please select at least one block.")
                return
            blocks = ["custom"]
            custom_blocks = selected

        output_dir = self.prefs_vars["lora_output_dir"].get()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)

        # Disable button, clear log
        self.extract_run_btn.configure(state="disabled")
        self.extract_open_btn.configure(state="disabled")
        self.extract_log.configure(state="normal")
        self.extract_log.delete(1.0, tk.END)
        self.extract_log.configure(state="disabled")
        self.extract_progress_var.set("Loading models...")

        import threading
        thread = threading.Thread(
            target=self._extract_worker,
            args=(source, output_path, dit_path, vae_path, te_path, blocks, prompt, custom_blocks),
            daemon=True,
        )
        thread.start()

    def _extract_worker(self, source, output_path, dit_path, vae_path, te_path, blocks, prompt, custom_blocks=None):
        """Background worker for extraction."""
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

            from fizgig.extraction.extractor import LoRAExtractor, ExtractionConfig

            timestep_presets = {
                "all": (0.0, 1.0),
                "early": (0.6, 1.0),
                "mid": (0.3, 0.7),
                "late": (0.0, 0.4),
            }

            rank = int(self.extract_rank_var.get())
            samples = int(self.extract_samples_var.get())
            timesteps = timestep_presets[self.extract_timesteps_var.get()]

            config = ExtractionConfig(
                source_lora_path=source,
                output_lora_path=output_path,
                target_rank=rank,
                timestep_range=timesteps,
                include_blocks=blocks,
                custom_blocks=custom_blocks,
                num_samples=samples,
                prompt=prompt,
                width=1024,
                height=1024,
                seed=42,
            )

            def progress(stage, current, total):
                def _update():
                    self.extract_progress_var.set(f"{stage}: {current+1}/{total}")
                self.master.after(0, _update)

            pipeline = None
            if samples == 0:
                # Pure weight SVD — no pipeline needed, no GPU models loaded
                self.master.after(0, lambda: self._extract_log(f"Pure weight SVD: blocks={blocks}, rank={rank}\n"))
                result = LoRAExtractor.extract_weight_only(config, progress_callback=progress)
            else:
                # Activation-weighted SVD — needs full pipeline for forward passes
                from fizgig.klein.inference import KleinInferencePipeline

                # Auto-detect model version and fp8 from filename
                dit_basename = os.path.basename(dit_path).lower()
                model_version = "klein-base-9b" if "base" in dit_basename else "klein-9b"
                is_fp8_model = "fp8" in dit_basename

                self.master.after(0, lambda: self._extract_log("Loading models...\n"))

                pipeline = KleinInferencePipeline()
                pipeline.load_models(
                    dit_path=dit_path,
                    vae_path=vae_path,
                    text_encoder_path=te_path,
                    model_version=model_version,
                    device="cuda",
                    fp8_scaled=not is_fp8_model,
                    fp8_text_encoder=True,
                    blocks_to_swap=self._get_inference_blocks_to_swap(),
                )

                self.master.after(0, lambda: self._extract_log(
                    f"Starting extraction: blocks={blocks}, rank={rank}, samples={samples}\n"))

                extractor = LoRAExtractor(pipeline)
                result = extractor.extract(config, progress_callback=progress)

            if pipeline is not None:
                pipeline.unload_models()

            summary = f"\nExtraction complete!\n"
            summary += f"  Output: {result.output_path}\n"
            summary += f"  Layers extracted: {result.num_layers_extracted}\n"
            summary += f"  Target rank: {result.target_rank}\n"
            summary += f"  Total params: {result.total_params:,}\n"
            summary += f"  Time: {result.elapsed_seconds:.1f}s\n"

            self._extract_output_path = output_path

            def _update_ui():
                self._extract_log(summary)
                self.extract_progress_var.set("Done!")
                self.extract_run_btn.configure(state="normal")
                self.extract_open_btn.configure(state="normal")

            self.master.after(0, _update_ui)

        except Exception as e:
            import traceback
            error_msg = f"Extraction failed:\n{traceback.format_exc()}"
            def _show_error():
                self._extract_log(error_msg)
                self.extract_progress_var.set("Error")
                self.extract_run_btn.configure(state="normal")
            self.master.after(0, _show_error)

    # endregion

    # region Preferences Tab

    def create_prefs_tab(self):
        """Create the Preferences tab (Start-tab styled)."""
        scrollable_frame, _ = self.create_scrollable_frame(self.prefs_tab)

        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer,
            "Preferences",
            "Centralised paths + inference performance knobs. Changes here propagate to every tab automatically "
            "and persist to prefs.json.",
        )

        # Card 1: Model Paths
        models_card = self._start_section_card(
            outer, "Model Paths (Klein 9B)",
            "Absolute paths to the four model files. Each row has a Download link that opens the HuggingFace page "
            "in your browser.",
        )
        models_card.columnconfigure(1, weight=1)
        next_row = 0
        next_row = self._add_pref_row(
            models_card, next_row, "Base DiT:", "base_dit",
            "Klein 9B Base model (for training & precise profiling). "
            "An fp8 version (~9.5GB) is also available — see the GitHub README for the link.",
            download_url="https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/tree/main",
            download_note="~17GB bf16 — Black Forest Labs on HuggingFace (flux-2-klein-base-9b.safetensors)",
        )
        next_row = self._add_pref_row(
            models_card, next_row, "Distilled DiT:", "distilled_dit",
            "Klein 9B Distilled model (for Repair Studio previews, fast profiling & diagnostics)",
            download_url="https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/tree/main",
            download_note="~9GB fp8 quantised — Black Forest Labs (flux-2-klein-9b-fp8.safetensors)",
        )
        next_row = self._add_pref_row(
            models_card, next_row, "VAE / AE:", "vae",
            "Flux 2 AutoEncoder — use ae.safetensors from FLUX.2-dev root (NOT the vae/ subfolder Diffusers file)",
            download_url="https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors",
            download_note="~320MB  ·  get ae.safetensors from FLUX.2-dev root  ·  NOT vae/diffusion_pytorch_model.safetensors (Diffusers format, incompatible)",
        )
        next_row = self._add_pref_row(
            models_card, next_row, "Text Encoder:", "text_encoder",
            "Qwen3-8B text encoder (used by Klein 9B)",
            download_url="https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors",
            download_note="~15GB single-file safetensors — Qwen3-8B packaged for Klein 9B (Comfy-Org)",
        )

        # Card 2: Inference Performance
        inf_card = self._start_section_card(
            outer, "Inference Performance",
            "DiT Block Swap moves transformer blocks to CPU during forward passes to cut VRAM, at the cost of PCIe "
            "latency per step. Affects Repair Studio / Profiler / Extract — Training has its own setting.",
        )
        inf_card.columnconfigure(1, weight=1)

        ttk.Label(inf_card, text="DiT Block Swap (inference):").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        inference_swap_options = [
            "0  (24 GB VRAM)",
            "4  (20 GB VRAM)",
            "8  (16 GB VRAM)",
            "12 (14 GB VRAM)",
            "16 (12 GB VRAM)",
        ]
        _inf_combo = ttk.Combobox(
            inf_card, textvariable=self.prefs_vars["inference_blocks_to_swap"],
            values=inference_swap_options, width=26, state="readonly",
        )
        _inf_combo.grid(row=0, column=1, sticky=tk.W, pady=4)
        # Snap whatever's saved to the matching new label by extracting the leading integer.
        import re as _re_snap
        _current = str(self.prefs_vars["inference_blocks_to_swap"].get()).strip()
        _m = _re_snap.match(r'\d+', _current)
        if _m:
            _leading_int = _m.group()
            for _opt in inference_swap_options:
                if _opt.lstrip().startswith(_leading_int + " ") or _opt.lstrip().startswith(_leading_int + "  "):
                    self.prefs_vars["inference_blocks_to_swap"].set(_opt)
                    break

        # Card 3: Output Directories
        out_card = self._start_section_card(
            outer, "Output Directories",
            "Paths stored as relative-to-repo when they live inside FizgigIndependent/ (portable across clones/moves), "
            "absolute otherwise. Dataset TOMLs always live in FizgigIndependent/dataset/ — not configurable.",
        )
        out_card.columnconfigure(1, weight=1)
        next_row = 0
        next_row = self._add_pref_row(out_card, next_row, "LoRA output:", "lora_output_dir", "Where trained LoRAs are saved", is_dir=True)
        next_row = self._add_pref_row(out_card, next_row, "Profiles:", "profiles_dir", "Where profiler HTML reports are saved", is_dir=True)
        next_row = self._add_pref_row(out_card, next_row, "Cache:", "cache_dir", "Cached latents and text encodings", is_dir=True)

        # Card 4: Actions
        actions_card = self._start_section_card(outer, "Actions", None)
        action_row = tk.Frame(actions_card, bg=COLORS["bg_surface"])
        action_row.pack(anchor=tk.W)
        ttk.Button(action_row, text="Reset to Defaults", command=self._reset_prefs).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(action_row, text="Open prefs.json", command=self._open_prefs_file).pack(side=tk.LEFT)

        self._add_youtube_help_button(outer, "preferences")

    def _add_pref_row(self, frame, row, label, pref_key, hint, is_dir=False, download_url=None, download_note=None):
        """Add a labeled pref entry with Browse button and hint text. Returns next row index.

        If download_url is set, a "Download" link is added next to Browse that opens the URL in
        the user's default browser. download_note (optional) is appended to the hint line.
        """
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        ttk.Entry(frame, textvariable=self.prefs_vars[pref_key], width=60).grid(row=row, column=1, sticky=tk.EW, pady=4)

        btn_frame = tk.Frame(frame, bg=COLORS["bg_surface"])
        btn_frame.grid(row=row, column=2, sticky=tk.W, padx=(8, 0))
        browse_cmd = (lambda: self._browse_pref_dir(pref_key)) if is_dir else (lambda: self._browse_pref_file(pref_key))
        ttk.Button(btn_frame, text="Browse", command=browse_cmd).pack(side=tk.LEFT)
        if download_url:
            dl_link = tk.Label(btn_frame, text="Download",
                               fg=COLORS["accent_hover"], cursor="hand2",
                               font=(FONT_FAMILY, 9, "underline"),
                               bg=COLORS["bg_surface"])
            dl_link.pack(side=tk.LEFT, padx=(8, 0))
            dl_link.bind("<Button-1>", lambda e, url=download_url: webbrowser.open(url))
            ToolTip(dl_link, f"Open download page in browser:\n{download_url}")
        row += 1
        hint_text = hint
        if download_note:
            hint_text = f"{hint}  ·  {download_note}"
        tk.Label(frame, text=hint_text,
                 font=(FONT_FAMILY, 9, "italic"),
                 fg=COLORS["text_muted"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).grid(
            row=row, column=1, columnspan=2, sticky=tk.W, pady=(0, 6)
        )
        row += 1
        return row

    def _browse_pref_file(self, pref_key):
        filepath = filedialog.askopenfilename(
            title="Select file",
            filetypes=[("SafeTensors", "*.safetensors"), ("All files", "*.*")]
        )
        if filepath:
            self.prefs_vars[pref_key].set(filepath)

    def _browse_pref_dir(self, pref_key):
        dirpath = filedialog.askdirectory(title="Select directory")
        if dirpath:
            self.prefs_vars[pref_key].set(dirpath)

    def _reset_prefs(self):
        if messagebox.askyesno("Reset Preferences", "Restore all paths to defaults?"):
            for key, default in DEFAULT_PREFS.items():
                if key in self.prefs_vars:
                    self.prefs_vars[key].set(default)

    def _open_prefs_file(self):
        if os.path.exists(PREFS_FILE):
            self._open_in_file_manager(PREFS_FILE)
        else:
            messagebox.showinfo("Info", "prefs.json doesn't exist yet — change a path to create it.")

    # endregion

    # region Profiler Tab

    def create_profiler_tab(self):
        """Create the LoRA Profiler tab (Start-tab styled)."""
        scrollable_frame, _ = self.create_scrollable_frame(self.profiler_tab)

        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer,
            "Profiler",
            "Analyze a LoRA to see which blocks are active at each denoising stage. "
            "Writes a full HTML report plus a `<name>_profile.json` sidecar that the Repair Studio reads inline.",
        )

        # Card 1: Model selection
        model_card = self._start_section_card(
            outer, "Model",
            "Paths are set on the Preferences tab. Distilled is a few seconds per probe and fine for most scans; "
            "Base produces the authoritative report but is slower.",
        )
        self.profiler_dit_choice_var = tk.StringVar(value="distilled")
        dit_choice_frame = tk.Frame(model_card, bg=COLORS["bg_surface"])
        dit_choice_frame.pack(anchor=tk.W)
        ttk.Radiobutton(dit_choice_frame, text="Distilled (fast, ~4-step probes)",
                        variable=self.profiler_dit_choice_var, value="distilled").pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(dit_choice_frame, text="Base (precise)",
                        variable=self.profiler_dit_choice_var, value="base").pack(side=tk.LEFT)

        # Card 2: LoRA
        lora_card = self._start_section_card(
            outer, "LoRA File",
            "Select the LoRA you want to profile. PEFT and LyCORIS (LoKR / LoHa) are auto-converted on load.",
        )
        lora_card.grid_columnconfigure(1, weight=1)
        ttk.Label(lora_card, text="LoRA File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.profiler_lora_var = tk.StringVar()
        ttk.Entry(lora_card, textvariable=self.profiler_lora_var, width=50).grid(row=0, column=1, sticky=tk.EW, pady=4)
        ttk.Button(lora_card, text="Browse", command=self._browse_profiler_lora).grid(row=0, column=2, sticky=tk.W, padx=(8, 0), pady=4)

        # Card 3: Prompt
        prompt_card = self._start_section_card(
            outer, "Prompt",
            "Include the LoRA's trigger word so the profile captures its active pathways, e.g.: "
            "`zwxem, a portrait photo of a woman`.",
        )
        prompt_card.grid_columnconfigure(1, weight=1)
        ttk.Label(prompt_card, text="Prompt:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.profiler_prompt_var = tk.StringVar(value="")
        ttk.Entry(prompt_card, textvariable=self.profiler_prompt_var, width=50).grid(row=0, column=1, sticky=tk.EW, pady=4)

        # Card 4: Options
        options_card = self._start_section_card(
            outer, "Options",
            "Resolution controls the probe render size; Stages is the number of denoising buckets the profiler measures.",
        )
        options_row = tk.Frame(options_card, bg=COLORS["bg_surface"])
        options_row.pack(anchor=tk.W)

        ttk.Label(options_row, text="Resolution:").pack(side=tk.LEFT, padx=(0, 6))
        self.profiler_res_var = tk.StringVar(value="1024")
        ttk.Combobox(options_row, textvariable=self.profiler_res_var,
                     values=["512", "768", "1024"], state="readonly", width=6).pack(side=tk.LEFT, padx=(0, 24))

        ttk.Label(options_row, text="Stages:").pack(side=tk.LEFT, padx=(0, 6))
        self.profiler_stages_var = tk.StringVar(value="5")
        ttk.Combobox(options_row, textvariable=self.profiler_stages_var,
                     values=["3", "5", "8", "10"], state="readonly", width=4).pack(side=tk.LEFT)

        # Card 5: Run
        run_card = self._start_section_card(outer, "Run", None)
        run_row = tk.Frame(run_card, bg=COLORS["bg_surface"])
        run_row.pack(anchor=tk.W)
        self.profiler_run_btn = ttk.Button(run_row, text="Profile LoRA", command=self._run_profiler, style="Primary.TButton")
        self.profiler_run_btn.pack(side=tk.LEFT, padx=(0, 12))
        self.profiler_open_btn = ttk.Button(run_row, text="Open Report", command=self._open_profiler_report, state="disabled")
        self.profiler_open_btn.pack(side=tk.LEFT)

        self.profiler_progress_var = tk.StringVar(value="")
        tk.Label(run_card, textvariable=self.profiler_progress_var,
                 font=(FONT_FAMILY, 10, "bold"),
                 fg=COLORS["accent_hover"], bg=COLORS["bg_surface"]).pack(anchor=tk.W, pady=(10, 0))

        # Card 6: Results
        results_card = self._start_section_card(
            outer, "Results",
            "Summary text lands here during profiling; the full heat-mapped report opens in your browser via Open Report.",
        )
        self.profiler_results = scrolledtext.ScrolledText(
            results_card, height=18, width=80,
            bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
            wrap="word", state="disabled",
            relief="flat", highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["border_focus"],
        )
        self.profiler_results.pack(fill=tk.BOTH, expand=True)

        self._profiler_report_path = None

        self._add_youtube_help_button(outer, "profiler")

    def _browse_profiler_lora(self):
        filepath = filedialog.askopenfilename(
            title="Select LoRA file",
            filetypes=[("SafeTensors", "*.safetensors")]
        )
        if filepath:
            self.profiler_lora_var.set(filepath)

    def _browse_profiler_file(self, var):
        filepath = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("SafeTensors", "*.safetensors")]
        )
        if filepath:
            var.set(filepath)

    def _open_profiler_report(self):
        if self._profiler_report_path and os.path.exists(self._profiler_report_path):
            import webbrowser
            webbrowser.open(self._profiler_report_path)

    def _profiler_log(self, text):
        """Append to profiler log (preserves user scroll position)."""
        self._smart_text_insert(self.profiler_results, text)

    def _run_profiler(self):
        """Start profiling in a background thread."""
        lora_path = self.profiler_lora_var.get()
        if not lora_path or not os.path.exists(lora_path):
            messagebox.showerror("Error", "Please select a valid LoRA file.")
            return

        prompt = self.profiler_prompt_var.get().strip()
        if not prompt:
            messagebox.showerror("Error", "Please enter a prompt (include the LoRA trigger word).")
            return

        # Resolve model paths from Preferences (single source of truth)
        dit_choice = self.profiler_dit_choice_var.get()
        dit_pref_key = "base_dit" if dit_choice == "base" else "distilled_dit"
        dit_path = self.prefs_vars[dit_pref_key].get() if dit_pref_key in self.prefs_vars else ""
        vae_path = self._get_path("VAE_MODEL")
        te_path = self._get_path("TEXT_ENCODER")

        if not dit_path:
            messagebox.showerror(
                "Error",
                f"{dit_choice.capitalize()} DiT path not set.\nConfigure it on the Preferences tab.",
            )
            return
        if not vae_path or not te_path:
            messagebox.showerror("Error", "VAE and Text Encoder paths not set.\nConfigure them on the Preferences tab.")
            return

        for path, name in [(dit_path, "DiT"), (vae_path, "VAE"), (te_path, "Text Encoder")]:
            if not os.path.exists(path):
                messagebox.showerror("Error", f"{name} file not found:\n{path}")
                return

        res = int(self.profiler_res_var.get())
        stages = int(self.profiler_stages_var.get())

        # Disable button during profiling
        self.profiler_run_btn.configure(state="disabled")
        self.profiler_open_btn.configure(state="disabled")
        self.profiler_results.configure(state="normal")
        self.profiler_results.delete(1.0, tk.END)
        self.profiler_results.configure(state="disabled")
        self.profiler_progress_var.set("Loading models...")

        import threading
        thread = threading.Thread(
            target=self._profiler_worker,
            args=(lora_path, prompt, dit_path, vae_path, te_path, res, stages),
            daemon=True,
        )
        thread.start()

    def _profiler_worker(self, lora_path, prompt, dit_path, vae_path, te_path, res, stages):
        """Background worker for profiling."""
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

            from fizgig.klein.inference import KleinInferencePipeline
            from fizgig.profiler.profiler import LoRAProfiler
            from fizgig.profiler.visualize import plot_profile_heatmap, print_profile_summary

            self.master.after(0, lambda: self._profiler_log("Loading models...\n"))

            # Auto-detect model version and fp8 from filename
            dit_basename = os.path.basename(dit_path).lower()
            model_version = "klein-base-9b" if "base" in dit_basename else "klein-9b"
            is_fp8_model = "fp8" in dit_basename

            pipeline = KleinInferencePipeline()
            pipeline.load_models(
                dit_path=dit_path,
                vae_path=vae_path,
                text_encoder_path=te_path,
                model_version=model_version,
                device="cuda",
                fp8_scaled=False if is_fp8_model else True,  # Don't apply fp8_scaled to already-fp8 models
                fp8_text_encoder=self.settings.get("FP8_TEXT_ENCODER", True),
                blocks_to_swap=self._get_inference_blocks_to_swap(),
            )

            self.master.after(0, lambda: self._profiler_log("Models loaded. Starting profiling...\n"))
            self.master.after(0, lambda: self.profiler_progress_var.set(f"Profiling: 0 of {stages} stages..."))

            profiler = LoRAProfiler(pipeline)

            # Patch the profiler to report progress
            original_profile = profiler.profile
            _self = self

            result = profiler.profile(
                lora_path=lora_path,
                num_samples=4,
                num_bins=stages,
                width=res,
                height=res,
                prompt=prompt,
                seed=42,
            )

            self.master.after(0, lambda: self.profiler_progress_var.set("Generating report..."))

            # Save report into the dedicated Profiles directory from prefs
            # (defaults to FizgigIndependent/profiles/).
            lora_name = os.path.splitext(os.path.basename(lora_path))[0]
            profiles_dir = self.prefs_vars["profiles_dir"].get() if "profiles_dir" in self.prefs_vars else os.path.join(OUTPUT_LORAS_DIR, "profiles")
            os.makedirs(profiles_dir, exist_ok=True)
            report_path = os.path.join(profiles_dir, f"{lora_name}_profile.html")

            plot_profile_heatmap(result, report_path)
            self._profiler_report_path = report_path

            # Build summary text
            from fizgig.profiler.visualize import _category_totals, _short_name, _get_category
            cat_totals = _category_totals(result)
            grand_total = sum(cat_totals.values()) or 1.0

            summary = f"LoRA Profile: {os.path.basename(lora_path)}\n"
            summary += f"Prompt: {prompt}\n"
            summary += f"Resolution: {res}x{res}\n\n"
            summary += f"Category Breakdown:\n"
            summary += f"  Style+Composition:        {cat_totals['style_composition']/grand_total*100:5.1f}%  (double 0-7 + single 0)\n"
            summary += f"  ↔ style↔identity:         {cat_totals['style_ident_overlap']/grand_total*100:5.1f}%  (single 1)\n"
            summary += f"  Identity:                 {cat_totals['identity']/grand_total*100:5.1f}%  (single 2-11)\n"
            summary += f"  ↔ identity↔details:       {cat_totals['ident_details_overlap']/grand_total*100:5.1f}%  (single 12-16)\n"
            summary += f"  Details:                  {cat_totals['details']/grand_total*100:5.1f}%  (single 17-23)\n\n"
            summary += f"Most Active Blocks:\n"
            for name, total in result.get_top_blocks(10):
                cat = _get_category(name)
                pct = total / grand_total * 100
                summary += f"  {_short_name(name):14s}  {pct:5.1f}%  [{cat}]\n"
            summary += f"\nReport saved: {report_path}\n"

            pipeline.unload_models()

            def _update_ui():
                self._profiler_log(summary)
                self.profiler_progress_var.set("Done!")
                self.profiler_run_btn.configure(state="normal")
                self.profiler_open_btn.configure(state="normal")

            self.master.after(0, _update_ui)

        except Exception as e:
            import traceback
            error_msg = f"Profiling failed:\n{traceback.format_exc()}"
            def _show_error():
                self._profiler_log(error_msg)
                self.profiler_progress_var.set("Error")
                self.profiler_run_btn.configure(state="normal")
            self.master.after(0, _show_error)

    # endregion

    # region Repair Studio Tab

    # Color palette mirrors src/fizgig/profiler/visualize.py 5-bucket scheme
    _REPAIR_CAT_COLOR = {
        "style_composition": "#5B9BD5",
        "style_ident_overlap": "#5BB3A6",
        "identity": "#70AD47",
        "ident_details_overlap": "#B8A547",
        "details": "#ED7D31",
    }
    _REPAIR_CAT_SHORT = {
        "style_composition": "Style+Comp",
        "style_ident_overlap": "Style/ID",
        "identity": "Identity",
        "ident_details_overlap": "ID/Detail",
        "details": "Details",
    }

    @staticmethod
    def _repair_category_for_block(block_id: str) -> str:
        kind, idx_s = block_id.split("_")
        idx = int(idx_s)
        if kind == "double":
            return "style_composition"
        if idx == 0:
            return "style_composition"
        if idx == 1:
            return "style_ident_overlap"
        if 2 <= idx <= 11:
            return "identity"
        if 12 <= idx <= 16:
            return "ident_details_overlap"
        return "details"

    def create_repair_studio_tab(self):
        """Per-block LoRA repair with side-by-side preview (Start-tab styled)."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        from fizgig.repair_studio.state import SliderState

        # Outer master scroll: the whole tab scrolls vertically when its
        # content exceeds the window height (e.g. when Res=768 blows up the
        # preview panel). The inner sliders panel has its own scroll too —
        # mousewheel hand-off is handled by <Enter>/<Leave> bind_all swapping
        # on each scrollable canvas independently.
        frame = self._build_repair_outer_scroll(self.repair_studio_tab)

        # Bg_deep container — all cards pack into this.
        outer = tk.Frame(frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        # Engine + state — lazy
        self.repair_engine = None
        self.repair_state = SliderState.default_klein9b()
        self.repair_block_vars = {}   # block_id -> dict(primary_chk, primary_scale, primary_lbl, donor_chk, donor_scale, donor_lbl)
        self.repair_thumbnails = {}   # GC-safe ImageTk.PhotoImage refs
        self.repair_pil_images = {"baseline": None, "tweaked": None}  # raw PIL for resize-to-fit
        self._repair_preview_redraw_after = {"baseline": None, "tweaked": None}
        self.repair_profile_match = None  # dict payload from profile sidecar, or None
        self._repair_preview_after_id = None
        self._repair_preview_in_flight = False
        self._repair_preview_dirty = False
        self._repair_donor_loaded = False

        self._add_tab_banner(
            outer,
            "Repair Studio",
            "Tweak each block's contribution live with side-by-side preview. "
            "Optional donor LoRA blends in via rank concatenation. Save the repaired result as a new .safetensors. "
            "Turbo Preview is on by default for faster updates — turn it off if VRAM is tight.",
        )

        # Card 1: Setup (DiT, Primary, Donor, Preview params, Preset)
        setup_card = self._start_section_card(
            outer, "Setup",
            "Paths come from Preferences. Load the primary LoRA first; donor is optional. "
            "Changing prompt / seed / resolution triggers a fresh baseline render.",
        )
        setup_card.columnconfigure(1, weight=1)
        self._build_repair_top_controls(setup_card)

        # Profile-match info panel (populated when a matching Profiler sidecar
        # exists for the primary's content hash). Packs directly into outer so
        # pack_forget/pack(before=…) slots it cleanly between Status and Preview.
        self.repair_profile_frame = tk.Frame(outer, bg=COLORS["bg_surface"],
                                             highlightbackground=COLORS["accent"],
                                             highlightthickness=1, bd=0)
        # Deliberately not packed yet — shown only when a match is found.

        # Card 2: Preview
        preview_card = self._start_section_card(
            outer, "Preview",
            "Left side is the baseline (LoRA at its original strengths); right side is the tweaked render using your slider state.",
        )
        # Anchor so _render_repair_profile_panel can pack(before=…) into the right slot.
        self._repair_profile_anchor = preview_card.master.master
        self._build_repair_preview_panel(preview_card)

        # Card 3: Master Controls
        master_card = self._start_section_card(
            outer, "Master Controls",
            "Bulk-tune by category. Flip the target radio to switch between primary and donor. "
            "Category toggles next to the donor ones bulk on/off the donor's contribution per bucket.",
        )
        self._build_repair_master_controls(master_card)

        # Card 4: Per-Block Sliders
        sliders_card = self._start_section_card(
            outer, "Per-Block Sliders",
            "Range ±3.0. Greyed-out rows are blocks the LoRA doesn't touch. "
            "Colour bands match the Profiler's 5-bucket scheme: blue Style+Comp, teal Style/ID, green Identity, olive ID/Detail, orange Details.",
        )
        self._build_repair_slider_panel(sliders_card)

        # Card 5: Actions
        actions_card = self._start_section_card(outer, "Actions", None)
        action_row = tk.Frame(actions_card, bg=COLORS["bg_surface"])
        action_row.pack(fill=tk.X)
        ttk.Button(action_row, text="Save Repaired LoRA…",
                   command=self._save_repaired_lora_action, style="Primary.TButton").pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(action_row, text="Reset All Sliders",
                   command=self._reset_repair_sliders).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(action_row, text="Reset Session (unload models)",
                   command=self._reset_repair_session).pack(side=tk.LEFT)
        tk.Button(action_row, text="Explore this in LoRA the Explorer \u2192",
                  font=(FONT_FAMILY, 10, "bold"),
                  fg="#FFFFFF", bg="#2E8B57", activeforeground="#FFFFFF", activebackground="#256F46",
                  relief="flat", bd=0, padx=16, pady=6, cursor="hand2",
                  command=self._repair_explore_in_explorer).pack(side=tk.RIGHT)

        self._add_youtube_help_button(outer, "repair_studio")

    def _build_repair_outer_scroll(self, tab):
        """Wrap the Repair Studio tab in a vertical scrolling canvas. Returns the
        inner Frame into which all tab content should be placed."""
        outer_canvas = tk.Canvas(tab, highlightthickness=0, bg=COLORS["bg_deep"])
        outer_scroll = ttk.Scrollbar(tab, orient="vertical", command=outer_canvas.yview)
        outer_canvas.configure(yscrollcommand=outer_scroll.set)
        outer_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        outer_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        inner = tk.Frame(outer_canvas, bg=COLORS["bg_deep"])
        inner_id = outer_canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_config(_e):
            outer_canvas.configure(scrollregion=outer_canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_config)

        def _on_canvas_config(e):
            outer_canvas.itemconfigure(inner_id, width=e.width)
            # Window resize also drives preview re-render (preview scales with width).
            self._schedule_repair_preview_redraws()
        outer_canvas.bind("<Configure>", _on_canvas_config)

        def _on_wheel(e):
            outer_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        outer_canvas.bind("<Enter>", lambda e: outer_canvas.bind_all("<MouseWheel>", _on_wheel))
        outer_canvas.bind("<Leave>", lambda e: outer_canvas.unbind_all("<MouseWheel>"))

        self.repair_outer_canvas = outer_canvas
        return inner

    def _schedule_repair_preview_redraws(self):
        """Debounced redraw trigger for both preview sides. Called on window resize."""
        for which in ("baseline", "tweaked"):
            if self._repair_preview_redraw_after.get(which) is not None:
                try:
                    self.master.after_cancel(self._repair_preview_redraw_after[which])
                except Exception:
                    pass
            self._repair_preview_redraw_after[which] = self.master.after(
                60, lambda w=which: self._repair_redraw_preview(w))

    def _repair_redraw_preview(self, which: str):
        """Rescale stored PIL image to fit current holder box, preserving aspect ratio."""
        pil = self.repair_pil_images.get(which)
        if pil is None:
            return
        if which == "baseline":
            label = self.repair_baseline_label
            holder = self.repair_base_holder
        else:
            label = self.repair_tweaked_label
            holder = self.repair_tweaked_holder
        # Ensure layout is current before we query sizes (first redraw can
        # fire before Tk has finished laying things out).
        try:
            holder.update_idletasks()
        except Exception:
            pass
        # Floor of 256 so pre-layout reads still produce a usable image.
        box_w = max(256, holder.winfo_width() - 8)
        box_h = max(256, holder.winfo_height() - 8)
        src_w, src_h = pil.size
        scale = min(box_w / src_w, box_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        from PIL import Image as _PILImage
        img = pil.resize((new_w, new_h), _PILImage.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.repair_thumbnails[which] = photo  # keep ref so Tk doesn't GC it
        label.configure(image=photo, text="")

    def _build_repair_top_controls(self, parent):
        r = 0
        # DiT toggle
        ttk.Label(parent, text="DiT:").grid(row=r, column=0, sticky=tk.W, padx=4, pady=2)
        self.repair_dit_choice_var = tk.StringVar(value="distilled")
        choice_frame = ttk.Frame(parent)
        choice_frame.grid(row=r, column=1, columnspan=3, sticky=tk.W, padx=4, pady=2)
        ttk.Radiobutton(choice_frame, text="Distilled (4-step, fast)",
                        variable=self.repair_dit_choice_var, value="distilled",
                        style="Surface.TRadiobutton").pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(choice_frame, text="Base (20-step, precise but slow)",
                        variable=self.repair_dit_choice_var, value="base",
                        style="Surface.TRadiobutton").pack(side=tk.LEFT)
        r += 1

        # Primary LoRA
        ttk.Label(parent, text="Primary LoRA:").grid(row=r, column=0, sticky=tk.W, padx=4, pady=2)
        self.repair_primary_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.repair_primary_var).grid(
            row=r, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(parent, text="Browse",
                   command=self._browse_and_load_primary).grid(
            row=r, column=2, columnspan=2, padx=4, pady=2, sticky=tk.W)
        r += 1

        # Donor LoRA
        ttk.Label(parent, text="Donor LoRA (optional):").grid(row=r, column=0, sticky=tk.W, padx=4, pady=2)
        self.repair_donor_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.repair_donor_var).grid(
            row=r, column=1, sticky=tk.EW, padx=4, pady=2)
        donor_btn_frame = ttk.Frame(parent)
        donor_btn_frame.grid(row=r, column=2, columnspan=2, padx=4, pady=2, sticky=tk.W)
        ttk.Button(donor_btn_frame, text="Browse",
                   command=self._browse_and_load_donor).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(donor_btn_frame, text="Unload Donor",
                   command=self._unload_repair_donor).pack(side=tk.LEFT)
        r += 1

        # Prompt row
        ttk.Label(parent, text="Prompt:").grid(row=r, column=0, sticky=tk.W, padx=4, pady=2)
        self.repair_prompt_var = tk.StringVar(value="")
        prompt_entry = ttk.Entry(parent, textvariable=self.repair_prompt_var)
        prompt_entry.grid(row=r, column=1, sticky=tk.EW, padx=4, pady=2)
        self.repair_prompt_var.trace_add("write", lambda *_: self._repair_mark_update_needed())
        params_frame = ttk.Frame(parent)
        params_frame.grid(row=r, column=2, columnspan=2, sticky=tk.EW, padx=4, pady=2)
        ttk.Label(params_frame, text="Seed:").pack(side=tk.LEFT, padx=(0, 4))
        self.repair_seed_var = tk.StringVar(value="42")
        seed_entry = ttk.Entry(params_frame, textvariable=self.repair_seed_var, width=10)
        seed_entry.pack(side=tk.LEFT, padx=(0, 2))
        self.repair_seed_var.trace_add("write", lambda *_: self._repair_mark_update_needed())
        tk.Button(params_frame, text="\u21bb", font=(FONT_FAMILY, 9),
                  bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
                  activebackground=COLORS["bg_surface"], activeforeground=COLORS["text_primary"],
                  relief="flat", bd=0, padx=4, pady=0, cursor="hand2",
                  command=self._repair_randomize_seed
                  ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(params_frame, text="Res:").pack(side=tk.LEFT, padx=(0, 4))
        self.repair_res_var = tk.StringVar(value="512")
        res_combo = ttk.Combobox(params_frame, textvariable=self.repair_res_var,
                                 values=["256", "384", "512", "768"], state="readonly", width=6)
        res_combo.pack(side=tk.LEFT)
        res_combo.bind("<<ComboboxSelected>>", lambda e: self._on_preview_param_changed())
        # Turbo Preview toggle
        self.repair_turbo_var = tk.BooleanVar(value=True)
        turbo_chk = ttk.Checkbutton(params_frame, text="Turbo Preview",
                                     variable=self.repair_turbo_var,
                                     command=self._on_turbo_toggled)
        turbo_chk.pack(side=tk.RIGHT)
        r += 1

        # Preset row
        ttk.Label(parent, text="Preset:").grid(row=r, column=0, sticky=tk.W, padx=4, pady=2)
        self.repair_preset_var = tk.StringVar()
        self.repair_preset_combo = ttk.Combobox(parent, textvariable=self.repair_preset_var,
                                                values=self._repair_preset_list(), state="readonly")
        self.repair_preset_combo.grid(row=r, column=1, sticky=tk.EW, padx=4, pady=2)
        self.repair_preset_combo.bind("<<ComboboxSelected>>",
                                      lambda e: self._load_repair_preset(self.repair_preset_var.get()))
        ttk.Button(parent, text="Save Preset…",
                   command=self._save_repair_preset).grid(row=r, column=2, columnspan=2, padx=4, pady=2, sticky=tk.W)
        r += 1

        # Status + Start button row
        status_row = tk.Frame(parent, bg=COLORS["bg_surface"])
        status_row.grid(row=r, column=0, columnspan=4, sticky=tk.EW, pady=(6, 0))
        self.repair_status_var = tk.StringVar(value="Set a LoRA path and prompt, then click Start.")
        tk.Label(status_row, textvariable=self.repair_status_var,
                 font=(FONT_FAMILY, 10, "italic"),
                 fg=COLORS["accent"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
        self._repair_start_btn = tk.Button(
            status_row, text="Start", font=(FONT_FAMILY, 11, "bold"),
            fg="#FFFFFF", bg="#2E8B57", activeforeground="#FFFFFF", activebackground="#256F46",
            relief="flat", bd=0, padx=24, pady=6, cursor="hand2",
            command=self._repair_start)
        self._repair_start_btn.pack(side=tk.RIGHT)
        r += 1

    def _build_repair_preview_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Baseline (LoRA at default 1.0)",
                  font=(FONT_FAMILY, 9, "bold")).grid(row=0, column=0, padx=4, pady=(2, 0))
        ttk.Label(parent, text="Tweaked (current sliders)",
                  font=(FONT_FAMILY, 9, "bold")).grid(row=0, column=1, padx=4, pady=(2, 0))

        # Pixel-size holders per side. pack_propagate(False) stops the holder
        # from shrinking to the label's text size — without this, the frames
        # collapse to tiny squares because the default "(no preview yet)" text
        # is small and the packed label would otherwise dictate the frame size.
        # sticky="nsew" + columnconfigure(weight=1) lets the holder stretch
        # wider when the window is enlarged; pack_propagate(False) keeps the
        # minimum height at 512.
        base_holder = tk.Frame(parent, width=512, height=512, bg="#1c1c1c",
                               highlightthickness=0)
        base_holder.grid(row=1, column=0, padx=4, pady=4, sticky="nsew")
        base_holder.pack_propagate(False)
        tweaked_holder = tk.Frame(parent, width=512, height=512, bg="#1c1c1c",
                                  highlightthickness=0)
        tweaked_holder.grid(row=1, column=1, padx=4, pady=4, sticky="nsew")
        tweaked_holder.pack_propagate(False)

        self.repair_baseline_label = ttk.Label(base_holder, text="(no baseline yet)",
                                               anchor=tk.CENTER, background="#1c1c1c")
        self.repair_baseline_label.pack(fill=tk.BOTH, expand=True)
        self.repair_tweaked_label = ttk.Label(tweaked_holder, text="(no preview yet)",
                                              anchor=tk.CENTER, background="#1c1c1c",
                                              cursor="hand2")
        self.repair_tweaked_label.pack(fill=tk.BOTH, expand=True)
        self.repair_tweaked_label.bind("<Button-1>", lambda e: self._repair_popout_preview())
        self.repair_base_holder = base_holder
        self.repair_tweaked_holder = tweaked_holder
        self._repair_popout_window = None
        self._repair_popout_label = None
        self._repair_popout_tk_img = None

        # Redraw on resize. Debounced so a drag doesn't spam Lanczos.
        def _mk_config_cb(which):
            def _cb(_e):
                if self._repair_preview_redraw_after.get(which) is not None:
                    try:
                        self.master.after_cancel(self._repair_preview_redraw_after[which])
                    except Exception:
                        pass
                self._repair_preview_redraw_after[which] = self.master.after(
                    60, lambda w=which: self._repair_redraw_preview(w))
            return _cb
        base_holder.bind("<Configure>", _mk_config_cb("baseline"))
        tweaked_holder.bind("<Configure>", _mk_config_cb("tweaked"))

    _REPAIR_MASTER_CATS = [
        ("style_composition", "Style+Comp"),
        ("style_ident_overlap", "Style/ID"),
        ("identity", "Identity"),
        ("ident_details_overlap", "ID/Detail"),
        ("details", "Details"),
    ]

    def _repair_quickset_buttons(self, parent, var, row, col_start, balance_cb=None):
        """Create [0] [1] [±] [⚖] quick-set buttons for a repair slider.

        Returns a list of button widgets (for greying in _refresh_block_slider_activity).
        balance_cb: optional callback for the balance button (sets complement on the other target).
        """
        btn_font = (FONT_FAMILY, 8)
        btn_kw = dict(bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
                      activebackground=COLORS["bg_surface"],
                      activeforeground=COLORS["text_primary"],
                      relief="flat", bd=0, padx=2, pady=0, width=2, font=btn_font)
        b0 = tk.Button(parent, text="0", command=lambda: var.set(0.0), **btn_kw)
        b0.grid(row=row, column=col_start, padx=1, pady=1)
        b1 = tk.Button(parent, text="1", command=lambda: var.set(1.0), **btn_kw)
        b1.grid(row=row, column=col_start + 1, padx=1, pady=1)
        bn = tk.Button(parent, text="\u00b1",
                       command=lambda: var.set(-var.get() or 0.0), **btn_kw)
        bn.grid(row=row, column=col_start + 2, padx=1, pady=1)
        btns = [b0, b1, bn]
        if balance_cb is not None:
            bb = tk.Button(parent, text="\u2696", command=balance_cb, **btn_kw)
            bb.grid(row=row, column=col_start + 3, padx=1, pady=1)
            btns.append(bb)
        return btns

    def _build_repair_master_controls(self, parent):
        """Target radio + 5 category master sliders + 5 donor category toggles."""
        # State vars
        self.repair_master_target_var = tk.StringVar(value="primary")
        self.repair_master_strength_vars = {
            cat: tk.DoubleVar(value=1.0) for cat, _ in self._REPAIR_MASTER_CATS
        }
        self.repair_master_strength_labels = {}
        self.repair_donor_category_vars = {
            cat: tk.BooleanVar(value=False) for cat, _ in self._REPAIR_MASTER_CATS
        }
        # Suppression flag: when a master slider moves, we set N per-block vars.
        # Each per-block trace would otherwise fire _schedule_preview individually
        # (cheap but noisy in logs). Keep the trace firing — it updates state —
        # but mute the preview-schedule during bulk updates, then fire ONE
        # preview at the end.
        self._repair_master_mutating = False

        r = 0
        # Target radio
        target_frame = ttk.Frame(parent)
        target_frame.grid(row=r, column=0, columnspan=6, sticky=tk.W, padx=6, pady=(4, 6))
        ttk.Label(target_frame, text="Master sliders affect:",
                  font=(FONT_FAMILY, 9, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(target_frame, text="Primary",
                        variable=self.repair_master_target_var, value="primary",
                        command=self._on_master_target_changed,
                        style="Surface.TRadiobutton").pack(side=tk.LEFT, padx=(0, 8))
        self._repair_master_donor_radio = ttk.Radiobutton(
            target_frame, text="Donor", variable=self.repair_master_target_var, value="donor",
            command=self._on_master_target_changed, style="Surface.TRadiobutton")
        self._repair_master_donor_radio.pack(side=tk.LEFT)
        self._repair_master_donor_radio.state(["disabled"])  # enabled when donor loads
        r += 1

        # 5 category sliders
        parent.columnconfigure(1, weight=1)
        for cat, short in self._REPAIR_MASTER_CATS:
            color = self._REPAIR_CAT_COLOR[cat]
            tk.Label(parent, text=short, fg=color, bg=COLORS["bg_surface"],
                     width=11, anchor=tk.W,
                     font=(FONT_FAMILY, 9, "bold")).grid(
                row=r, column=0, sticky=tk.W, padx=(10, 4), pady=1)
            var = self.repair_master_strength_vars[cat]
            scale = ttk.Scale(parent, from_=-3.0, to=3.0, variable=var, orient=tk.HORIZONTAL)
            scale.grid(row=r, column=1, sticky=tk.EW, padx=4, pady=1)
            val_lbl = ttk.Label(parent, text="1.00", width=5, anchor=tk.E)
            val_lbl.grid(row=r, column=2, padx=(4, 4), pady=1)
            self.repair_master_strength_labels[cat] = val_lbl
            self._repair_quickset_buttons(parent, var, r, 3,
                balance_cb=lambda c=cat: self._repair_balance_master(c))

            def _mk_trace(_var, _lbl, _cat):
                def _cb(*_a):
                    v = float(_var.get())
                    _lbl.configure(text=f"{v:+.2f}" if v != 1.0 else "1.00")
                    self._on_master_strength_changed(_cat, v)
                return _cb
            var.trace_add("write", _mk_trace(var, val_lbl, cat))
            r += 1

        # Donor category toggles now live in the Per-Block Sliders card (created in create_repair_studio_tab)

    def _repair_balance_master(self, category: str):
        """Balance master slider: set the other target's category to 1.0 - current value."""
        if self.repair_engine is None or self.repair_engine.donor_network is None:
            return
        target = self.repair_master_target_var.get()
        current = self.repair_master_strength_vars[category].get()
        if current < 0 or current > 1.0:
            return
        complement = 1.0 - current
        # Set the other target's blocks
        other_key = "donor_strength" if target == "primary" else "primary_strength"
        affected = [bid for bid in self.repair_block_vars
                    if self._repair_category_for_block(bid) == category]
        self._repair_master_mutating = True
        try:
            for bid in affected:
                self.repair_block_vars[bid][other_key].set(complement)
                if other_key == "donor_strength":
                    self.repair_block_vars[bid]["donor_enabled"].set(True)
        finally:
            self._repair_master_mutating = False
        if self.repair_engine is not None and self.repair_engine.primary_network is not None:
            self.repair_engine.mark_blocks_changed(affected)
            self._schedule_preview()

    def _repair_balance_block(self, block_id: str, source: str):
        """Balance a single block: set the other side to 1.0 - current value."""
        if self.repair_engine is None or self.repair_engine.donor_network is None:
            return
        v = self.repair_block_vars[block_id]
        if source == "primary":
            current = v["primary_strength"].get()
            if current < 0 or current > 1.0:
                return
            v["donor_strength"].set(1.0 - current)
            v["donor_enabled"].set(True)
        else:
            current = v["donor_strength"].get()
            if current < 0 or current > 1.0:
                return
            v["primary_strength"].set(1.0 - current)

    def _on_master_strength_changed(self, category: str, value: float):
        """Mirror master slider value to per-block strength vars for affected blocks."""
        # If the master var is being set programmatically (e.g. on target switch
        # to reflect current per-block values), skip the mirror — otherwise we'd
        # flatten the very diversity we're trying to display.
        if getattr(self, "_repair_master_mutating", False):
            return
        target = self.repair_master_target_var.get()
        affected = [bid for bid in self.repair_block_vars
                    if self._repair_category_for_block(bid) == category]
        if not affected:
            return
        # Bulk-set: each per-block strength trace will update state, but we
        # want ONE preview schedule for the whole batch. Mark mutating so
        # _on_block_changed can short-circuit the preview schedule.
        self._repair_master_mutating = True
        try:
            key = "primary_strength" if target == "primary" else "donor_strength"
            for bid in affected:
                self.repair_block_vars[bid][key].set(value)
        finally:
            self._repair_master_mutating = False
        # Fire one preview for the whole batch.
        if self.repair_engine is not None and self.repair_engine.primary_network is not None:
            self.repair_engine.mark_blocks_changed(affected)
            self._schedule_preview()

    def _on_master_target_changed(self):
        """When target radio flips (Primary↔Donor), refresh master sliders to
        show the current average per-block strength for the new target. Without
        this, the master sliders would display stale values from the previous
        target and mislead the user."""
        target = self.repair_master_target_var.get()
        key = "primary_strength" if target == "primary" else "donor_strength"
        self._repair_master_mutating = True
        try:
            for cat, _ in self._REPAIR_MASTER_CATS:
                affected = [bid for bid in self.repair_block_vars
                            if self._repair_category_for_block(bid) == cat]
                if not affected:
                    continue
                values = [float(self.repair_block_vars[bid][key].get()) for bid in affected]
                avg = sum(values) / len(values) if values else 1.0
                self.repair_master_strength_vars[cat].set(round(avg, 3))
        finally:
            self._repair_master_mutating = False

    def _on_donor_category_toggled(self, category: str):
        """Mirror donor category toggle to donor_enabled vars for affected blocks."""
        on = bool(self.repair_donor_category_vars[category].get())
        affected = [bid for bid in self.repair_block_vars
                    if self._repair_category_for_block(bid) == category]
        if not affected:
            return
        self._repair_master_mutating = True
        try:
            for bid in affected:
                self.repair_block_vars[bid]["donor_enabled"].set(on)
        finally:
            self._repair_master_mutating = False
        if self.repair_engine is not None and self.repair_engine.primary_network is not None:
            self.repair_engine.mark_blocks_changed(affected)
            self._schedule_preview()

    def _build_repair_slider_panel(self, parent):
        # Scrollable canvas (vertical) holding two columns: double on left, single on right.
        # Bounded height (500px) so the panel stays compact inside the outer scroll
        # and the user can independently scroll all 32 rows without losing the preview.
        canvas = tk.Canvas(parent, highlightthickness=0, bg=COLORS["bg_surface"], height=500)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_config(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_config)

        def _on_canvas_config(e):
            canvas.itemconfigure(inner_id, width=e.width)
        canvas.bind("<Configure>", _on_canvas_config)

        # Mousewheel scrolling when hovering the panel
        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)

        # Two balanced columns: doubles + singles 0-7 on left, singles 8-23 on right
        col_left = ttk.Frame(inner)
        col_left.grid(row=0, column=0, sticky=tk.NSEW, padx=4)
        col_right = ttk.Frame(inner)
        col_right.grid(row=0, column=1, sticky=tk.NSEW, padx=4)

        # Left column: double blocks, then single 0-7
        r = 0
        ttk.Label(col_left, text="Double Blocks", font=(FONT_FAMILY, 10, "bold")).grid(
            row=r, column=0, padx=0, pady=(2, 4), sticky=tk.W)
        r += 1
        for i in range(8):
            self._build_repair_block_row(col_left, f"double_{i}", r)
            r += 1
        ttk.Label(col_left, text="Single Blocks 0\u20137", font=(FONT_FAMILY, 10, "bold")).grid(
            row=r, column=0, padx=0, pady=(8, 4), sticky=tk.W)
        r += 1
        for i in range(8):
            self._build_repair_block_row(col_left, f"single_{i}", r)
            r += 1

        # Right column: single 8-23
        r = 0
        ttk.Label(col_right, text="Single Blocks 8\u201323", font=(FONT_FAMILY, 10, "bold")).grid(
            row=r, column=0, padx=0, pady=(2, 4), sticky=tk.W)
        r += 1
        for i in range(8, 24):
            self._build_repair_block_row(col_right, f"single_{i}", r)
            r += 1

    def _build_repair_block_row(self, parent, block_id: str, row: int):
        cat = self._repair_category_for_block(block_id)
        color = self._REPAIR_CAT_COLOR[cat]
        cat_short = self._REPAIR_CAT_SHORT[cat]
        kind, idx = block_id.split("_")

        rowf = ttk.Frame(parent)
        rowf.grid(row=row, column=0, sticky=tk.EW, pady=1)
        rowf.columnconfigure(3, weight=1)

        # Primary checkbox + label + slider + value
        primary_enabled = tk.BooleanVar(value=True)
        primary_strength = tk.DoubleVar(value=1.0)
        donor_enabled = tk.BooleanVar(value=True)
        donor_strength = tk.DoubleVar(value=0.0)

        chk_p = ttk.Checkbutton(rowf, variable=primary_enabled,
                                command=lambda b=block_id: self._on_block_changed(b))
        chk_p.grid(row=0, column=0, padx=(2, 4))
        # Block label with category color
        lbl_text = f"{kind} {idx}"
        lbl = tk.Label(rowf, text=lbl_text, fg=color, bg=COLORS["bg_surface"],
                       width=10, anchor=tk.W, font=(FONT_FAMILY, 9, "bold"))
        lbl.grid(row=0, column=1, padx=(0, 2))
        cat_lbl = tk.Label(rowf, text=f"[{cat_short}]", fg=color, bg=COLORS["bg_surface"],
                           width=11, anchor=tk.W, font=(FONT_FAMILY, 8))
        cat_lbl.grid(row=0, column=2, padx=(0, 4))

        scale_p = ttk.Scale(rowf, from_=-3.0, to=3.0, variable=primary_strength, orient=tk.HORIZONTAL)
        scale_p.grid(row=0, column=3, sticky=tk.EW, padx=2)

        val_lbl_p = ttk.Label(rowf, text="1.00", width=5, anchor=tk.E)
        val_lbl_p.grid(row=0, column=4, padx=(2, 2))
        btns_p = self._repair_quickset_buttons(rowf, primary_strength, 0, 5,
            balance_cb=lambda b=block_id: self._repair_balance_block(b, "primary"))

        # Donor row (hidden until donor is loaded)
        donor_rowf = ttk.Frame(rowf)
        donor_rowf.grid(row=1, column=0, columnspan=9, sticky=tk.EW, padx=(20, 0))
        donor_rowf.columnconfigure(2, weight=1)
        donor_rowf.grid_remove()
        chk_d = ttk.Checkbutton(donor_rowf, variable=donor_enabled,
                                command=lambda b=block_id: self._on_block_changed(b))
        chk_d.grid(row=0, column=0, padx=(2, 4))
        donor_tag_lbl = ttk.Label(donor_rowf, text="donor", foreground="#888",
                                  font=(FONT_FAMILY, 8, "italic"),
                                  width=11, anchor=tk.W)
        donor_tag_lbl.grid(row=0, column=1, padx=(0, 4))
        scale_d = ttk.Scale(donor_rowf, from_=-3.0, to=3.0, variable=donor_strength, orient=tk.HORIZONTAL)
        scale_d.grid(row=0, column=2, sticky=tk.EW, padx=2)
        val_lbl_d = ttk.Label(donor_rowf, text="1.00", width=5, anchor=tk.E)
        val_lbl_d.grid(row=0, column=3, padx=(2, 2))
        btns_d = self._repair_quickset_buttons(donor_rowf, donor_strength, 0, 4,
            balance_cb=lambda b=block_id: self._repair_balance_block(b, "donor"))

        # Bind variable traces to mirror into self.repair_state and live-update labels
        def _mk_strength_trace(var, lbl, bid, which):
            def _cb(*_a):
                v = float(var.get())
                lbl.configure(text=f"{v:+.2f}" if v != 1.0 else "1.00")
                bs = self.repair_state.blocks[bid]
                if which == "primary":
                    bs.primary_strength = v
                else:
                    bs.donor_strength = v
                self._on_block_changed(bid)
            return _cb

        def _mk_enabled_trace(var, bid, which):
            def _cb(*_a):
                bs = self.repair_state.blocks[bid]
                if which == "primary":
                    bs.primary_enabled = bool(var.get())
                else:
                    bs.donor_enabled = bool(var.get())
                self._on_block_changed(bid)
            return _cb

        primary_strength.trace_add("write", _mk_strength_trace(primary_strength, val_lbl_p, block_id, "primary"))
        donor_strength.trace_add("write", _mk_strength_trace(donor_strength, val_lbl_d, block_id, "donor"))
        primary_enabled.trace_add("write", _mk_enabled_trace(primary_enabled, block_id, "primary"))
        donor_enabled.trace_add("write", _mk_enabled_trace(donor_enabled, block_id, "donor"))

        self.repair_block_vars[block_id] = {
            # StringVars / BooleanVars
            "primary_enabled": primary_enabled,
            "primary_strength": primary_strength,
            "donor_enabled": donor_enabled,
            "donor_strength": donor_strength,
            # value readouts
            "primary_lbl": val_lbl_p,
            "donor_lbl": val_lbl_d,
            # row frames
            "donor_rowf": donor_rowf,
            # widget handles (for _refresh_block_slider_activity greying)
            "chk_p": chk_p,
            "scale_p": scale_p,
            "block_lbl": lbl,
            "cat_lbl": cat_lbl,
            "chk_d": chk_d,
            "scale_d": scale_d,
            "donor_tag_lbl": donor_tag_lbl,
            # quick-set buttons (for greying)
            "btns_p": btns_p,
            "btns_d": btns_d,
            # category color (for restore after greying)
            "cat_color": color,
        }

    # ------------------------------------------------------------
    # Repair Studio actions
    # ------------------------------------------------------------

    def _browse_repair_lora(self, var):
        filepath = filedialog.askopenfilename(
            title="Select LoRA file", filetypes=[("SafeTensors", "*.safetensors")]
        )
        if filepath:
            var.set(filepath)

    def _ensure_repair_engine(self):
        """Lazy-load the engine + pipeline. Returns True on success, False on failure."""
        if self.repair_engine is not None and self.repair_engine.pipeline is not None and self.repair_engine.pipeline.is_loaded:
            return True

        dit_choice = self.repair_dit_choice_var.get()
        dit_pref_key = "base_dit" if dit_choice == "base" else "distilled_dit"
        dit_path = self.prefs_vars[dit_pref_key].get() if dit_pref_key in self.prefs_vars else ""
        vae_path = self._get_path("VAE_MODEL")
        te_path = self._get_path("TEXT_ENCODER")

        if not dit_path or not os.path.exists(dit_path):
            messagebox.showerror("Error", f"{dit_choice.capitalize()} DiT path not set or not found.\nConfigure on Preferences tab.")
            return False
        if not vae_path or not os.path.exists(vae_path):
            messagebox.showerror("Error", "VAE path not set or not found.\nConfigure on Preferences tab.")
            return False
        if not te_path or not os.path.exists(te_path):
            messagebox.showerror("Error", "Text encoder path not set or not found.\nConfigure on Preferences tab.")
            return False

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        from fizgig.repair_studio.engine import RepairEngine

        if self.repair_engine is None:
            self.repair_engine = RepairEngine()
        self.repair_engine._turbo_enabled = self.repair_turbo_var.get()

        # Auto-detect fp8 + model_version from filename, mirroring profiler.
        dit_basename = os.path.basename(dit_path).lower()
        model_version = "klein-base-9b" if "base" in dit_basename else "klein-9b"
        is_fp8_model = "fp8" in dit_basename
        try:
            self.repair_status_var.set(f"Loading models ({model_version})…")
            self.master.update_idletasks()
            self.repair_engine.ensure_pipeline(
                dit_path=dit_path, vae_path=vae_path, text_encoder_path=te_path,
                model_version=model_version, device="cuda",
                fp8_scaled=False if is_fp8_model else True,
                blocks_to_swap=self._get_inference_blocks_to_swap(),
            )
            self.repair_status_var.set("Models loaded.")
            return True
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Failed to load models:\n{traceback.format_exc()}")
            self.repair_status_var.set("Error loading models.")
            return False

    def _repair_start(self):
        """Smart Start: load/swap primary, load/swap donor, or regenerate."""
        primary_path = self.repair_primary_var.get().strip()
        donor_path = self.repair_donor_var.get().strip()

        if not primary_path:
            messagebox.showerror("Error", "Set a primary LoRA path first.")
            return

        if not os.path.exists(primary_path):
            messagebox.showerror("Error", f"Primary LoRA not found:\n{primary_path}")
            return

        # Check if primary needs loading or swapping
        current_primary = self.repair_engine.primary_path if self.repair_engine else None
        if current_primary != primary_path:
            # New or changed primary — reset and reload
            if self.repair_engine is not None and self.repair_engine.primary_network is not None:
                self._reset_repair_session()
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self._load_repair_primary()
        elif self.repair_engine is None or self.repair_engine.primary_network is None:
            self._load_repair_primary()

        # Check if donor needs loading or swapping
        if donor_path and os.path.exists(donor_path):
            current_donor = self.repair_engine.donor_path if self.repair_engine else None
            if current_donor != donor_path:
                if self.repair_engine and self.repair_engine.donor_network is not None:
                    self._unload_repair_donor()
                self._load_repair_donor()

        # If everything is already loaded and nothing changed, just regenerate
        if (self.repair_engine is not None and self.repair_engine.primary_network is not None
                and current_primary == primary_path):
            self._force_regenerate_preview()

        # Reset button text back to Start
        self._repair_reset_start_button()

    def _load_repair_primary(self):
        path = self.repair_primary_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Pick a valid primary LoRA file first.")
            return
        if not self._ensure_repair_engine():
            return
        try:
            self.repair_status_var.set("Loading primary LoRA…")
            self.master.update_idletasks()
            # Detect format for user info
            from safetensors.torch import load_file as _lf
            from fizgig.networks.lora import ensure_kohya_lora_state_dict as _ek, detect_lora_format as _df
            _fmt = _df(_ek(_lf(path)))
            self.repair_engine.load_primary(path)
            self._refresh_block_slider_activity()
            n_active = len(self.repair_engine.primary_block_ids)
            if _fmt in ("lokr", "loha"):
                messagebox.showinfo("LyCORIS LoRA loaded",
                    f"This is a {_fmt.upper()} LoRA. Live preview works normally.\n\n"
                    f"If you save, blocks will be converted to standard LoRA via SVD "
                    f"(slight approximation).")
            # Look up a matching Profiler sidecar by content hash and render
            # the inline info panel if one exists.
            self._find_repair_profile_match()
            self.repair_status_var.set(
                f"Primary loaded: {os.path.basename(path)} ({n_active}/32 blocks). Generating preview…")
            self._schedule_preview(force=True)
        except Exception as ex:
            from fizgig.networks.lora import UnsupportedLoRAFormat
            if isinstance(ex, UnsupportedLoRAFormat):
                messagebox.showerror("Unsupported LoRA format", str(ex))
                self.repair_status_var.set(f"Unsupported format: {os.path.basename(path)}.")
            else:
                import traceback
                messagebox.showerror("Error", f"Failed to load primary:\n{traceback.format_exc()}")
                self.repair_status_var.set("Error loading primary.")

    def _load_repair_donor(self):
        path = self.repair_donor_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Pick a valid donor LoRA file first.")
            return
        if self.repair_engine is None or self.repair_engine.primary_network is None:
            messagebox.showerror("Error", "Load a primary LoRA before adding a donor.")
            return
        try:
            self.repair_status_var.set("Loading donor LoRA…")
            self.master.update_idletasks()
            from safetensors.torch import load_file as _lf
            from fizgig.networks.lora import ensure_kohya_lora_state_dict as _ek, detect_lora_format as _df
            _fmt_d = _df(_ek(_lf(path)))
            self.repair_engine.load_donor(path)
            if _fmt_d in ("lokr", "loha"):
                messagebox.showinfo("LyCORIS donor loaded",
                    f"This donor is a {_fmt_d.upper()} LoRA (LyCORIS format). "
                    f"Live preview works normally.\n\n"
                    f"If you save with blended blocks, they will be converted to standard "
                    f"LoRA via SVD. This may take a minute or two for large LoRAs "
                    f"(GPU-accelerated when available).")
            self._repair_donor_loaded = True
            # Show donor sub-rows + master section toggles + enable the "Donor" master target radio
            for vars_ in self.repair_block_vars.values():
                vars_["donor_rowf"].grid()
            self._repair_master_donor_radio.state(["!disabled"])
            for cat in self.repair_donor_category_vars:
                self.repair_donor_category_vars[cat].set(False)
            self._refresh_block_slider_activity()
            n_donor = len(self.repair_engine.donor_block_ids)
            self.repair_status_var.set(
                f"Donor loaded: {os.path.basename(path)} ({n_donor}/32 blocks). Enable per-block to mix in.")
        except Exception as ex:
            from fizgig.networks.lora import UnsupportedLoRAFormat
            if isinstance(ex, UnsupportedLoRAFormat):
                messagebox.showerror("Unsupported LoRA format", str(ex))
                self.repair_status_var.set(f"Unsupported format: {os.path.basename(path)}.")
            else:
                import traceback
                messagebox.showerror("Error", f"Failed to load donor:\n{traceback.format_exc()}")
                self.repair_status_var.set("Error loading donor.")

    def _repair_mark_update_needed(self):
        """Prompt or seed changed — show 'Update' on the Start button instead of auto-regenerating."""
        if self.repair_engine is not None and self.repair_engine.primary_network is not None:
            self._repair_start_btn.configure(text="Update")

    def _repair_randomize_seed(self):
        """Randomize seed and mark update needed."""
        import random
        self.repair_seed_var.set(str(random.randint(1, 99999)))
        self._repair_mark_update_needed()

    def _repair_reset_start_button(self):
        """Reset the Start button text back to 'Start'."""
        self._repair_start_btn.configure(text="Start")

    def _browse_and_load_primary(self):
        """Browse for a primary LoRA, and auto-swap if one is already loaded."""
        self._browse_repair_lora(self.repair_primary_var)
        path = self.repair_primary_var.get().strip()
        if not path or not os.path.exists(path):
            return
        if self.repair_engine is None or self.repair_engine.primary_network is None:
            return  # not loaded yet — user will click Start
        # Path changed — auto-swap
        if self.repair_engine.primary_path != path:
            # Remember donor path before reset clears it
            donor_path = self.repair_donor_var.get().strip()
            self._reset_repair_session()
            # Force GC + CUDA flush between unload and reload to prevent OOM
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._load_repair_primary()
            # Reload donor if one was set
            if donor_path and os.path.exists(donor_path):
                self._load_repair_donor()
            # Refresh master slider display to match reloaded state
            self._on_master_target_changed()
            self._repair_reset_start_button()

    def _browse_and_load_donor(self):
        """Browse for a donor LoRA, and auto-load if primary is loaded."""
        self._browse_repair_lora(self.repair_donor_var)
        path = self.repair_donor_var.get().strip()
        if not path or not os.path.exists(path):
            return
        if self.repair_engine is None or self.repair_engine.primary_network is None:
            return
        # Swap donor if one is already loaded
        if self.repair_engine.donor_network is not None:
            current_donor = self.repair_engine.donor_path
            if current_donor == path:
                return  # same file, nothing to do
            self._unload_repair_donor()
        self._load_repair_donor()
        self._repair_reset_start_button()

    def _unload_repair_donor(self):
        if self.repair_engine is None or self.repair_engine.donor_network is None:
            return
        self.repair_engine.unload_donor()
        self._repair_donor_loaded = False
        # Hide donor sub-rows + master section toggles + revert donor master radio
        for vars_ in self.repair_block_vars.values():
            vars_["donor_rowf"].grid_remove()
            vars_["donor_enabled"].set(True)
            vars_["donor_strength"].set(0.0)
        # donor toggles removed — donor blocks managed via master sliders
        self._repair_master_donor_radio.state(["disabled"])
        if self.repair_master_target_var.get() == "donor":
            self.repair_master_target_var.set("primary")
        self._refresh_block_slider_activity()
        self.repair_status_var.set("Donor unloaded.")
        self._schedule_preview(force=True)

    def _on_block_changed(self, block_id: str):
        if self.repair_engine is None or self.repair_engine.primary_network is None:
            return
        # During a master-slider bulk update, skip the per-block preview schedule;
        # the master handler fires ONE preview at the end of the batch.
        if getattr(self, "_repair_master_mutating", False):
            return
        # v2 hook
        self.repair_engine.mark_blocks_changed([block_id])
        self._schedule_preview()

    def _on_preview_param_changed(self):
        # seed/prompt/resolution change → invalidate baseline cache and regen.
        print(f"[repair] param change: res={self.repair_res_var.get()!r} "
              f"seed={self.repair_seed_var.get()!r} prompt={self.repair_prompt_var.get()!r}")
        if self.repair_engine is not None:
            self.repair_engine._invalidate_baseline_cache()
        self._repair_preview_dirty = True
        self._schedule_preview(force=True)

    def _on_turbo_toggled(self):
        """Sync Turbo Preview checkbox to the engine and invalidate cache on toggle."""
        if self.repair_engine is not None:
            self.repair_engine._turbo_enabled = self.repair_turbo_var.get()
            self.repair_engine._invalidate_activation_cache()

    def _force_regenerate_preview(self):
        if self.repair_engine is not None:
            self.repair_engine._invalidate_baseline_cache()
        self._repair_preview_dirty = True
        self._schedule_preview(force=True)

    def _schedule_preview(self, force: bool = False):
        if self.repair_engine is None or self.repair_engine.primary_network is None:
            return
        if self._repair_preview_after_id is not None:
            try:
                self.master.after_cancel(self._repair_preview_after_id)
            except Exception:
                pass
        turbo_on = getattr(self, "repair_turbo_var", None) and self.repair_turbo_var.get()
        if force:
            delay = 100
        elif turbo_on:
            delay = 150
        else:
            delay = 400
        print(f"[repair] schedule preview: force={force} in_flight={self._repair_preview_in_flight} "
              f"delay={delay}ms dirty={self._repair_preview_dirty}")
        self._repair_preview_after_id = self.master.after(delay, self._run_preview_async)

    def _run_preview_async(self):
        self._repair_preview_after_id = None
        if self._repair_preview_in_flight:
            # A preview is running. Mark dirty so its completion hook
            # (_set_repair_preview_images) fires a fresh preview once it lands.
            # (No more 500ms polling — was fragile and made races worse.)
            self._repair_preview_dirty = True
            print("[repair] run_async: in-flight, marked dirty; will refire after completion")
            return
        # Sync state from UI to repair_state (prompt/seed/res live in entries, not bound)
        prompt_text = self.repair_prompt_var.get().strip()
        if not prompt_text:
            # Distilled with empty conditioning → 4 steps of unguided denoising =
            # blocky noise (VAE decoding pure latent noise). Require a prompt.
            self.repair_status_var.set("Enter a prompt (include the LoRA trigger word) to generate previews.")
            return
        try:
            self.repair_state.seed = int(self.repair_seed_var.get() or "42")
        except ValueError:
            self.repair_state.seed = 42
        self.repair_state.prompt = prompt_text
        try:
            res = int(self.repair_res_var.get())
        except ValueError:
            res = 512
        self.repair_state.preview_width = res
        self.repair_state.preview_height = res

        # Snapshot for thread (dataclass copy via JSON round-trip)
        from fizgig.repair_studio.state import SliderState
        snapshot = self.repair_state.copy()

        # Clear the dirty flag NOW; any param change after this point will
        # set it again and trigger a re-fire when this preview completes.
        self._repair_preview_dirty = False
        self._repair_preview_in_flight = True
        print(f"[repair] run_async: starting worker w={snapshot.preview_width} "
              f"h={snapshot.preview_height} seed={snapshot.seed} prompt={snapshot.prompt!r}")
        self.repair_status_var.set("Generating preview…")
        import threading
        thread = threading.Thread(target=self._repair_preview_worker, args=(snapshot,), daemon=True)
        thread.start()

    def _repair_preview_worker(self, snapshot):
        try:
            if self.repair_engine is None:
                self._repair_preview_in_flight = False
                return
            print(f"[repair] worker: generating baseline at "
                  f"{snapshot.preview_width}x{snapshot.preview_height}")
            baseline = self.repair_engine.generate_baseline(snapshot)
            print(f"[repair] worker: baseline done, size={baseline.size}")
            print(f"[repair] worker: generating tweaked at "
                  f"{snapshot.preview_width}x{snapshot.preview_height}")
            tweaked = self.repair_engine.generate_preview(snapshot)
            print(f"[repair] worker: tweaked done, size={tweaked.size}")
            self.master.after(0, lambda: self._set_repair_preview_images(baseline, tweaked))
        except Exception:
            import traceback
            err = traceback.format_exc()
            def _show():
                self.repair_status_var.set("Preview error — see console.")
                print(err)
                self._repair_preview_in_flight = False
                # If params changed while we were erroring, still re-fire.
                if self._repair_preview_dirty:
                    self._repair_preview_dirty = False
                    self._schedule_preview(force=True)
            self.master.after(0, _show)

    def _set_repair_preview_images(self, baseline_img, tweaked_img):
        try:
            # Store the raw PIL so <Configure> resize can re-render at any size.
            self.repair_pil_images["baseline"] = baseline_img
            self.repair_pil_images["tweaked"] = tweaked_img
            self._repair_redraw_preview("baseline")
            self._repair_redraw_preview("tweaked")
            self._repair_update_popout()
            self.repair_status_var.set("Ready.")
            print(f"[repair] preview displayed: baseline={baseline_img.size} tweaked={tweaked_img.size}")
        finally:
            self._repair_preview_in_flight = False
            # Dirty flag was set during the in-flight preview → re-fire with
            # fresh state (pulls newest res/seed/prompt/slider values).
            if self._repair_preview_dirty:
                self._repair_preview_dirty = False
                print("[repair] dirty flag set during preview — refiring")
                self._schedule_preview(force=True)

    def _repair_popout_preview(self):
        """Open (or raise) a resizable pop-out window showing the tweaked preview."""
        if self._repair_popout_window is not None:
            try:
                if self._repair_popout_window.winfo_exists():
                    self._repair_popout_window.lift()
                    self._repair_update_popout()
                    return
            except Exception:
                pass
            self._repair_popout_window = None

        pil_img = self.repair_pil_images.get("tweaked")
        if pil_img is None:
            return

        win = tk.Toplevel(self.master)
        win.title("Repair Studio \u2014 Tweaked Preview")
        win.configure(bg="#000000")
        win.geometry(f"{pil_img.width}x{pil_img.height}")
        win.minsize(128, 128)

        lbl = tk.Label(win, bg="#000000")
        lbl.pack(fill=tk.BOTH, expand=True)

        self._repair_popout_window = win
        self._repair_popout_label = lbl

        def _on_close():
            self._repair_popout_window = None
            self._repair_popout_label = None
            self._repair_popout_tk_img = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

        def _on_resize(event):
            if event.widget == win:
                self._repair_update_popout()

        win.bind("<Configure>", _on_resize)
        self._repair_update_popout()

    def _repair_update_popout(self):
        """Push the current tweaked PIL image to the pop-out window, scaled to fit."""
        if self._repair_popout_window is None or self._repair_popout_label is None:
            return
        try:
            if not self._repair_popout_window.winfo_exists():
                self._repair_popout_window = None
                return
        except Exception:
            self._repair_popout_window = None
            return

        pil_img = self.repair_pil_images.get("tweaked")
        if pil_img is None:
            return

        from PIL import ImageTk
        w = self._repair_popout_window.winfo_width()
        h = self._repair_popout_window.winfo_height()
        if w < 10 or h < 10:
            return

        # Scale to fit window, preserving aspect ratio (upscale allowed)
        img_w, img_h = pil_img.size
        scale = min(w / img_w, h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        resized = pil_img.resize((new_w, new_h), resample=3)  # LANCZOS=3
        self._repair_popout_tk_img = ImageTk.PhotoImage(resized)
        self._repair_popout_label.configure(image=self._repair_popout_tk_img)

    def _save_repaired_lora_action(self):
        if self.repair_engine is None or self.repair_engine.primary_network is None:
            messagebox.showerror("Error", "Load a primary LoRA first.")
            return
        # Does the current slider state enable any donor blocks?
        donor_enabled_bids = [bid for bid, bs in self.repair_state.blocks.items() if bs.donor_enabled]
        donor_loaded = (self.repair_engine.donor_network is not None)
        donor_path = self.repair_engine.donor_path if donor_loaded else None

        primary_stem = os.path.splitext(os.path.basename(self.repair_engine.primary_path))[0]
        if donor_enabled_bids and donor_loaded:
            donor_stem = os.path.splitext(os.path.basename(donor_path))[0]
            default_name = f"{primary_stem}_with_{donor_stem}.safetensors"
        else:
            default_name = f"{primary_stem}_repaired.safetensors"

        out = filedialog.asksaveasfilename(
            title="Save Repaired LoRA",
            defaultextension=".safetensors",
            filetypes=[("SafeTensors", "*.safetensors")],
            initialfile=default_name,
        )
        if not out:
            return
        from fizgig.repair_studio.bake import save_repaired_lora
        from fizgig.networks.lora import UnsupportedLoRAFormat
        try:
            summary = save_repaired_lora(
                self.repair_engine.primary_path,
                self.repair_state,
                out,
                donor_path=donor_path if donor_enabled_bids else None,
            )
            msg = (
                f"Saved: {out}\n\n"
                f"Keys: {summary['keys_in']} → {summary['keys_out']}\n"
                f"Dropped blocks ({len(summary['dropped_blocks'])}): "
                f"{', '.join(summary['dropped_blocks']) or 'none'}\n"
                f"Rescaled blocks ({len(summary['rescaled_blocks'])}): "
                f"{', '.join(summary['rescaled_blocks']) or 'none'}\n"
                f"Donor-blended blocks ({len(summary['blended_blocks'])}): "
                f"{', '.join(summary['blended_blocks']) or 'none'}"
            )
            if summary['blended_blocks']:
                msg += "\n\nNote: blended blocks have rank = rank_primary + rank_donor. File size grows proportionally."
            messagebox.showinfo("Repaired LoRA saved", msg)
        except UnsupportedLoRAFormat as ex:
            messagebox.showerror("Bake not supported for this LoRA format", str(ex))
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Save failed:\n{traceback.format_exc()}")

    def _reset_repair_sliders(self):
        from fizgig.repair_studio.state import SliderState
        defaults = SliderState.default_klein9b()
        # Suppress per-block preview spam while bulk-resetting.
        self._repair_master_mutating = True
        try:
            for bid, bs in defaults.blocks.items():
                v = self.repair_block_vars.get(bid)
                if v is None:
                    continue
                v["primary_enabled"].set(bs.primary_enabled)
                v["primary_strength"].set(bs.primary_strength)
                v["donor_enabled"].set(bs.donor_enabled)
                v["donor_strength"].set(bs.donor_strength)
            # Reset master sliders + donor category toggles too.
            for cat in self.repair_master_strength_vars:
                self.repair_master_strength_vars[cat].set(1.0)
            for cat in self.repair_donor_category_vars:
                self.repair_donor_category_vars[cat].set(False)
        finally:
            self._repair_master_mutating = False
        self._schedule_preview(force=True)

    def _unload_repair_studio_models(self):
        """Unload Repair Studio pipeline + networks to free VRAM when leaving the tab.

        Unlike _reset_repair_session, this preserves all UI state (slider
        positions, loaded paths, etc.) so re-entering the tab and clicking
        Load again is seamless.  The engine is fully reset so the next load
        rebuilds the pipeline from scratch.
        """
        if self.repair_engine is not None and self.repair_engine.pipeline is not None:
            try:
                self.repair_engine.reset()
            except Exception:
                pass
            self.repair_engine = None
            self.repair_status_var.set("Models unloaded (tab switch). Load a LoRA to resume.")

    def _repair_explore_in_explorer(self):
        """Send current Repair Studio slider state to the Explorer for evolutionary discovery."""
        if self.repair_engine is None or self.repair_engine.primary_network is None:
            messagebox.showerror("Error", "Load a primary LoRA first.")
            return

        # Warn if LyCORIS — saving from Explorer will require SVD
        lora_path = self.repair_engine.primary_path
        try:
            from safetensors.torch import load_file as _lf
            from fizgig.networks.lora import ensure_kohya_lora_state_dict as _ek, detect_lora_format as _df
            _fmt = _df(_ek(_lf(lora_path)))
            if _fmt in ("lokr", "loha"):
                proceed = messagebox.askyesno(
                    "LyCORIS LoRA",
                    f"This is a {_fmt.upper()} LoRA. Explorer preview works normally, "
                    f"but saving will require SVD conversion (may take a minute).\n\n"
                    f"Consider using the Extract tab to convert to standard LoRA first "
                    f"for faster saves.\n\nContinue anyway?")
                if not proceed:
                    return
        except Exception:
            pass

        # Warn if donor is loaded — Explorer only supports primary
        if self.repair_engine.donor_network is not None:
            proceed = messagebox.askyesno(
                "Donor LoRA loaded",
                "The Explorer only works with a single primary LoRA — "
                "donor blending isn't supported there.\n\n"
                "Continue with just the primary LoRA's slider state?\n\n"
                "Tip: you can Save Repaired LoRA first to bake the primary+donor "
                "blend into a single file, then explore that.")
            if not proceed:
                return

        from fizgig.repair_studio.state import SliderState

        # Capture current state
        lora_path = self.repair_engine.primary_path
        current_state = self.repair_state.copy()
        prompt = self.repair_prompt_var.get()
        seed = self.repair_seed_var.get()
        res = self.repair_res_var.get()

        # Reset Repair Studio (frees VRAM)
        self._reset_repair_session()

        # Set up Explorer fields
        self.explorer_lora_var.set(lora_path)
        self.explorer_prompt_var.set(prompt)
        self.explorer_seed_var.set(seed)
        self.explorer_res_var.set(res)

        # Switch to Explorer tab
        self.notebook.select(self.explorer_tab)

        # Load LoRA in Explorer
        if not self._explorer_ensure_engine():
            return
        try:
            self.explorer_status_var.set("Loading from Repair Studio...")
            self.master.update_idletasks()
            if self._explorer_engine.primary_network is not None:
                self._explorer_engine.reset()
                self._explorer_engine = None
                if not self._explorer_ensure_engine():
                    return
            self._explorer_engine.load_primary(lora_path)

            # Set the Explorer baseline to the Repair Studio's slider state
            self._explorer_baseline_state = current_state
            self._explorer_baseline_state.prompt = prompt
            self._explorer_baseline_state.seed = int(seed or 42)
            r = int(res or 512)
            self._explorer_baseline_state.preview_width = r
            self._explorer_baseline_state.preview_height = r
            self._explorer_history.clear()
            self._explorer_locked_blocks.clear()
            self._explorer_last_pick_blocks.clear()
            self._explorer_baseline_image = None
            self._explorer_undo_btn.configure(state="disabled")
            self._explorer_save_btn.configure(state="disabled")
            self._explorer_refine_btn.configure(state="disabled")
            self._explorer_freeze_btn.configure(state="disabled")
            self._explorer_roll_btn.configure(state="normal")

            # Set low intensity + structure for refinement (subtle variants)
            self.explorer_intensity_var.set(0.25)   # ±0.9
            self.explorer_structure_var.set(0.15)    # 15%

            n_active = len(self._explorer_engine.primary_block_ids)
            self.explorer_status_var.set(
                f"Loaded from Repair Studio: {os.path.basename(lora_path)} ({n_active}/32 blocks). "
                f"Refining with low intensity. Generating variants...")
            self._explorer_generate_baseline_and_roll()

            self.master.after(500, lambda: messagebox.showinfo(
                "Refinement Mode",
                "Your Repair Studio slider state is now the Explorer baseline.\n\n"
                "Intensity and Structure have been set low so variants are subtle "
                "refinements of your current settings. Increase them if you want "
                "bolder exploration."))
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Failed to load in Explorer:\n{traceback.format_exc()}")

    def _reset_repair_session(self):
        # Close pop-out preview window if open
        if self._repair_popout_window is not None:
            try:
                self._repair_popout_window.destroy()
            except Exception:
                pass
            self._repair_popout_window = None
            self._repair_popout_label = None
            self._repair_popout_tk_img = None
        if self.repair_engine is not None:
            try:
                self.repair_engine.reset()
            except Exception:
                pass
        self.repair_engine = None
        self._repair_donor_loaded = False
        self._repair_preview_in_flight = False
        self._repair_preview_dirty = False
        for vars_ in self.repair_block_vars.values():
            vars_["donor_rowf"].grid_remove()
        # Hide master donor toggles + disable donor radio
        try:
            # donor toggles removed — donor blocks managed via master sliders
            self._repair_master_donor_radio.state(["disabled"])
            self.repair_master_target_var.set("primary")
        except Exception:
            pass
        # Reset master sliders to defaults (no preview — nothing loaded)
        self._repair_master_mutating = True
        try:
            for cat in self.repair_master_strength_vars:
                self.repair_master_strength_vars[cat].set(1.0)
            for cat in self.repair_donor_category_vars:
                self.repair_donor_category_vars[cat].set(False)
        finally:
            self._repair_master_mutating = False
        # Restore all sliders to the pre-load visual default.
        self._refresh_block_slider_activity()
        self.repair_thumbnails.clear()
        self.repair_pil_images["baseline"] = None
        self.repair_pil_images["tweaked"] = None
        self.repair_baseline_label.configure(image="", text="(no baseline yet)")
        self.repair_tweaked_label.configure(image="", text="(no preview yet)")
        # Clear any profile-match panel.
        self.repair_profile_match = None
        try:
            self.repair_profile_frame.pack_forget()
        except Exception:
            pass
        self.repair_status_var.set("Session reset. Load a primary LoRA to start.")

    def _find_repair_profile_match(self):
        """Look up a Profiler sidecar whose hash matches the loaded primary.
        If found, render the info panel; otherwise hide it."""
        self.repair_profile_match = None
        if self.repair_engine is None or not self.repair_engine.primary_hash:
            self._render_repair_profile_panel()
            return
        from fizgig.repair_studio.engine import find_profile_for_hash
        profiles_dir = self.prefs_vars["profiles_dir"].get() if "profiles_dir" in self.prefs_vars else ""
        try:
            match = find_profile_for_hash(profiles_dir, self.repair_engine.primary_hash)
        except Exception:
            match = None
        self.repair_profile_match = match
        self._render_repair_profile_panel()

    def _render_repair_profile_panel(self):
        """Populate (or hide) the profile-match info panel."""
        frame = self.repair_profile_frame
        # Clear previous children.
        for child in list(frame.winfo_children()):
            try:
                child.destroy()
            except Exception:
                pass

        match = self.repair_profile_match
        if not match:
            frame.pack_forget()
            return

        lora_name = match.get("lora_name") or "(unknown)"
        created = match.get("created") or ""
        top = match.get("top_active_blocks") or []
        if not top:
            frame.pack_forget()
            return

        # Ensure visible in the correct slot (between Status line and Preview card).
        frame.pack(fill=tk.X, padx=36, pady=(0, 16), before=self._repair_profile_anchor)

        tk.Label(frame, text="Profile found for this LoRA",
                 font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(
            anchor=tk.W, padx=20, pady=(16, 2)
        )
        tk.Label(
            frame,
            text=f"{lora_name}  ·  profiled {created[:10] if created else ''}  ·  most active blocks:",
            font=(FONT_FAMILY, 9, "italic"),
            fg=COLORS["text_muted"], bg=COLORS["bg_surface"],
        ).pack(anchor=tk.W, padx=20, pady=(0, 8))

        body = tk.Frame(frame, bg=COLORS["bg_surface"])
        body.pack(fill=tk.X, padx=20, pady=(0, 16))

        pills_frame = tk.Frame(body, bg=COLORS["bg_surface"])
        pills_frame.pack(side=tk.LEFT, anchor=tk.W)
        for idx, b in enumerate(top[:8]):
            name = b.get("name", "")
            category = b.get("category", "identity")
            pct = b.get("pct", 0)
            color = self._REPAIR_CAT_COLOR.get(category, "#888")
            pill = tk.Label(
                pills_frame, text=f"{name}  {pct:.1f}%",
                fg="#FFFFFF", bg=color,
                font=(FONT_FAMILY, 9, "bold"),
                padx=8, pady=3,
            )
            pill.pack(side=tk.LEFT, padx=(0, 4))

        # Open Report button — launches the HTML in the system browser.
        sidecar_path = match.get("_sidecar_path")
        html_name = match.get("html_report") or ""
        html_path = os.path.join(os.path.dirname(sidecar_path), html_name) if (sidecar_path and html_name) else None
        if html_path and os.path.isfile(html_path):
            ttk.Button(body, text="📊 Open full report",
                       command=lambda p=html_path: self._open_repair_profile_report(p)).pack(
                side=tk.RIGHT, padx=(8, 0))

    def _open_repair_profile_report(self, html_path: str):
        import webbrowser
        try:
            webbrowser.open(html_path)
        except Exception:
            messagebox.showerror("Error", f"Could not open report:\n{html_path}")

    def _refresh_block_slider_activity(self):
        """Grey out primary/donor rows for blocks the loaded LoRA doesn't touch.

        Called after primary/donor load, donor unload, and session reset.
        When no engine is loaded, everything is restored to normal.
        """
        primary_ids = (
            self.repair_engine.primary_block_ids
            if self.repair_engine is not None and self.repair_engine.primary_network is not None
            else None
        )
        donor_ids = (
            self.repair_engine.donor_block_ids
            if self.repair_engine is not None and self.repair_engine.donor_network is not None
            else None
        )
        grey_fg = "#555"

        for block_id, v in self.repair_block_vars.items():
            # Primary activity
            p_active = primary_ids is None or block_id in primary_ids
            p_state = ["!disabled"] if p_active else ["disabled"]
            try:
                v["chk_p"].state(p_state)
                v["scale_p"].state(p_state)
            except Exception:
                pass
            for btn in v.get("btns_p", []):
                btn.configure(state="normal" if p_active else "disabled")
            p_color = v["cat_color"] if p_active else grey_fg
            v["block_lbl"].configure(fg=p_color)
            v["cat_lbl"].configure(fg=p_color)
            v["primary_lbl"].configure(foreground=p_color if not p_active else "")
            if not p_active:
                # Reset var to default so an absent block never carries stale edits.
                v["primary_enabled"].set(True)
                v["primary_strength"].set(1.0)

            # Donor activity (only meaningful when donor row is visible)
            d_active = donor_ids is None or block_id in donor_ids
            d_state = ["!disabled"] if d_active else ["disabled"]
            try:
                v["chk_d"].state(d_state)
                v["scale_d"].state(d_state)
            except Exception:
                pass
            for btn in v.get("btns_d", []):
                btn.configure(state="normal" if d_active else "disabled")
            d_color = "#888" if d_active else grey_fg
            v["donor_tag_lbl"].configure(foreground=d_color)
            v["donor_lbl"].configure(foreground=d_color if d_active else grey_fg)
            if donor_ids is not None and not d_active:
                v["donor_enabled"].set(True)
                v["donor_strength"].set(0.0)

    # ---------------- Presets (built-in + user JSON) -----------------

    _REPAIR_BUILTIN_PRESETS = {
        "✨Reset All": "reset",
        "✨Identity Only": "identity",
        "✨Style+Composition Only": "style",
        "✨Details Only": "details",
    }

    def _repair_preset_dir(self) -> str:
        d = os.path.join(os.path.dirname(__file__), "presets", "repair_studio")
        os.makedirs(d, exist_ok=True)
        return d

    def _repair_preset_list(self) -> list:
        names = list(self._REPAIR_BUILTIN_PRESETS.keys())
        try:
            for fn in sorted(os.listdir(self._repair_preset_dir())):
                if fn.lower().endswith(".json"):
                    names.append(os.path.splitext(fn)[0])
        except Exception:
            pass
        return names

    def _refresh_repair_preset_combo(self):
        if hasattr(self, "repair_preset_combo"):
            self.repair_preset_combo.configure(values=self._repair_preset_list())

    def _apply_repair_state_to_widgets(self, state):
        for bid, bs in state.blocks.items():
            v = self.repair_block_vars.get(bid)
            if v is None:
                continue
            v["primary_enabled"].set(bs.primary_enabled)
            v["primary_strength"].set(bs.primary_strength)
            v["donor_enabled"].set(bs.donor_enabled)
            v["donor_strength"].set(bs.donor_strength)
        self.repair_seed_var.set(str(state.seed))
        self.repair_prompt_var.set(state.prompt)
        self.repair_res_var.set(str(state.preview_width))

    def _repair_builtin_state(self, kind: str):
        from fizgig.repair_studio.state import SliderState
        s = SliderState.default_klein9b()
        s.seed = self.repair_state.seed
        s.prompt = self.repair_state.prompt
        s.preview_width = self.repair_state.preview_width
        s.preview_height = self.repair_state.preview_height
        if kind == "reset":
            return s
        if kind == "identity":
            for bid, bs in s.blocks.items():
                cat = self._repair_category_for_block(bid)
                bs.primary_enabled = cat in ("identity", "style_ident_overlap", "ident_details_overlap")
            return s
        if kind == "style":
            for bid, bs in s.blocks.items():
                cat = self._repair_category_for_block(bid)
                bs.primary_enabled = cat in ("style_composition", "style_ident_overlap")
            return s
        if kind == "details":
            for bid, bs in s.blocks.items():
                cat = self._repair_category_for_block(bid)
                bs.primary_enabled = cat in ("details", "ident_details_overlap")
            return s
        return s

    def _save_repair_preset(self):
        name = simpledialog.askstring("Save Repair Studio Preset", "Preset name:")
        if not name:
            return
        # Sanitize
        name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
        if not name or name.startswith("✨"):
            messagebox.showerror("Error", "Invalid preset name.")
            return
        path = os.path.join(self._repair_preset_dir(), f"{name}.json")
        if os.path.exists(path):
            if not messagebox.askokcancel("Overwrite?", f"Overwrite existing preset '{name}'?"):
                return
        try:
            # Sync seed/prompt/res from widgets first
            try:
                self.repair_state.seed = int(self.repair_seed_var.get() or "42")
            except ValueError:
                self.repair_state.seed = 42
            self.repair_state.prompt = self.repair_prompt_var.get()
            try:
                self.repair_state.preview_width = int(self.repair_res_var.get())
                self.repair_state.preview_height = self.repair_state.preview_width
            except ValueError:
                pass
            with open(path, "w", encoding="utf-8") as f:
                import json as _json
                _json.dump(self.repair_state.to_json(), f, indent=2)
            self._refresh_repair_preset_combo()
            self.repair_preset_var.set(name)
            messagebox.showinfo("Saved", f"Saved preset: {name}")
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Save failed:\n{traceback.format_exc()}")

    def _load_repair_preset(self, name: str):
        if not name:
            return
        from fizgig.repair_studio.state import SliderState
        if name in self._REPAIR_BUILTIN_PRESETS:
            state = self._repair_builtin_state(self._REPAIR_BUILTIN_PRESETS[name])
            self._apply_repair_state_to_widgets(state)
            self._schedule_preview(force=True)
            return
        path = os.path.join(self._repair_preset_dir(), f"{name}.json")
        if not os.path.exists(path):
            return
        try:
            import json as _json
            with open(path, "r", encoding="utf-8") as f:
                d = _json.load(f)
            state = SliderState.from_json(d)
            self._apply_repair_state_to_widgets(state)
            self._schedule_preview(force=True)
        except Exception:
            import traceback
            messagebox.showerror("Error", f"Load failed:\n{traceback.format_exc()}")

    # endregion

    def auto_save_dataset_config_silent(self):
        """Silently auto-save dataset config on startup if all required fields are valid"""
        try:
            dataset_name = self.dataset_name_var.get().strip()
            dataset_type = self.dataset_type_var.get()

            # Skip if no dataset name
            if not dataset_name:
                return

            # Check for invalid chars
            invalid_chars = '<>:"/\\|?*'
            if any(c in dataset_name for c in invalid_chars):
                return

            is_video = "Video" in dataset_type
            is_jsonl = "JSONL" in dataset_type

            # Check required fields exist
            if is_jsonl:
                jsonl_file = self.dataset_jsonl_file_var.get().strip()
                if not jsonl_file or not os.path.exists(jsonl_file):
                    return
            else:
                if is_video:
                    data_dir = self.dataset_video_dir_var.get().strip()
                else:
                    data_dir = self.image_folder_var.get().strip()
                if not data_dir or not os.path.exists(data_dir):
                    return

            # Validate numeric fields
            try:
                megapixels = float(self.dataset_megapixels_var.get())
                if megapixels <= 0:
                    return
                side = int(math.sqrt(megapixels * 1_000_000))
                side = (side // 16) * 16
                res_width = side
                res_height = side
                batch_size = int(self.dataset_batch_size_var.get())
                num_repeats = 1  # hardcoded — UI removed (Klein workflow always uses 1)
            except ValueError:
                return

            # Build TOML string
            toml_lines = ["[general]"]
            toml_lines.append(f"resolution = [{res_width}, {res_height}]")

            if not is_jsonl:
                caption_ext = self.dataset_caption_ext_var.get().strip()
                toml_lines.append(f'caption_extension = "{caption_ext}"')

            toml_lines.append(f"batch_size = {batch_size}")
            toml_lines.append(f"num_repeats = {num_repeats}")
            toml_lines.append(f"enable_bucket = {'true' if self.dataset_enable_bucket_var.get() else 'false'}")
            toml_lines.append(f"bucket_no_upscale = {'true' if self.dataset_no_upscale_var.get() else 'false'}")
            toml_lines.append("")
            toml_lines.append("[[datasets]]")

            # Cache directory is now sourced from Preferences (no longer a Dataset-tab field)
            cache_dir = self.prefs_vars["cache_dir"].get().strip() if "cache_dir" in self.prefs_vars else ""

            if is_jsonl:
                jsonl_file = self.dataset_jsonl_file_var.get().strip().replace("\\", "/")
                if is_video:
                    toml_lines.append(f'video_jsonl_file = "{jsonl_file}"')
                else:
                    toml_lines.append(f'image_jsonl_file = "{jsonl_file}"')
            else:
                if is_video:
                    video_dir = self.dataset_video_dir_var.get().strip().replace("\\", "/")
                    toml_lines.append(f'video_directory = "{video_dir}"')
                else:
                    image_dir = self.image_folder_var.get().strip().replace("\\", "/")
                    toml_lines.append(f'image_directory = "{image_dir}"')

            if cache_dir:
                toml_lines.append(f'cache_directory = "{cache_dir.replace(chr(92), "/")}"')

            if is_video:
                try:
                    target_frames = [int(x.strip()) for x in self.dataset_target_frames_var.get().split(",")]
                    toml_lines.append(f"target_frames = [{', '.join(str(f) for f in target_frames)}]")
                    toml_lines.append(f'frame_extraction = "{self.dataset_frame_extraction_var.get()}"')
                    source_fps = float(self.dataset_source_fps_var.get())
                    toml_lines.append(f"source_fps = {source_fps}")
                except ValueError:
                    pass

            toml_content = "\n".join(toml_lines) + "\n"

            # Save to file (silently overwrite if exists)
            output_path = os.path.join(DATASET_DIR, f"{dataset_name}.toml")

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(toml_content)

            # Set as active dataset
            self._dataset_config_var.set(output_path)
            self.settings["DATASET_CONFIG"] = output_path

        except Exception:
            pass  # Silently fail - user can manually save if needed

    def show_context_menu(self, event):
        """Show context menu on right-click"""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def copy_selected_text(self):
        """Copy selected text to clipboard"""
        if self.console_output.selection_get():
            self.master.clipboard_clear()
            self.master.clipboard_append(self.console_output.selection_get())

    def browse_directory(self, setting_name):
        path = filedialog.askdirectory()
        if path:
            self.entries[setting_name].delete(0, tk.END)
            self.entries[setting_name].insert(0, path)

    def on_mousewheel(self, event):
        """Handle scroll event"""
        if self.console_output.yview()[1] < 1.0:
            self.user_scrolled = True
        else:
            self.user_scrolled = False

    def update_console(self, line):
        """Update training console — only auto-scroll if user was already at the bottom.
        Uses the widget's own yview() position as the authoritative signal; the older
        self.user_scrolled flag sometimes got out of sync with actual widget state."""
        try:
            at_bottom = self.console_output.yview()[1] >= 0.999
        except Exception:
            at_bottom = True
        self.console_output.configure(state="normal")
        self.console_output.insert(tk.END, line)
        if at_bottom:
            self.console_output.see(tk.END)
        self.console_output.configure(state="disabled")

        # Detect CUDA OOM and suggest increasing block swap
        if "CUDA out of memory" in line or "OutOfMemoryError" in line:
            if not getattr(self, "_oom_warning_shown", False):
                self._oom_warning_shown = True
                current_swap = self._parse_blocks_swap()
                messagebox.showwarning("Out of Memory",
                    f"CUDA ran out of memory during training.\n\n"
                    f"Current Block Swap: {current_swap}\n\n"
                    f"Try increasing Block Swap on the Training tab "
                    f"(Memory & FP8 section) to move more blocks to CPU. "
                    f"If set to Auto, switch to a manual value like "
                    f"{min(current_swap + 4, 16)}.")

    def _browse_context_lora(self):
        """File picker for the Context LoRA, filtered to .safetensors."""
        path = filedialog.askopenfilename(
            title="Select Context LoRA",
            filetypes=[("SafeTensors", "*.safetensors"), ("All files", "*.*")],
        )
        if path:
            self.entries["CONTEXT_LORA_PATH"].delete(0, tk.END)
            self.entries["CONTEXT_LORA_PATH"].insert(0, path)

    def browse_file(self, setting_name, input_type):
        if input_type == "directory":
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename()
        if path:
            self.settings[setting_name] = path
            self.entries[setting_name].delete(0, tk.END)
            self.entries[setting_name].insert(0, self.settings[setting_name])

    def validate_inputs(self):
        """Validate all inputs before starting training"""
        errors = []

        # Get current architecture config
        arch = self.architecture_var.get()
        config = ARCHITECTURES.get(arch, ARCHITECTURES["Flux 2 Klein Base 9B"])

        # Check required paths exist (sources: prefs_vars for model paths, hidden var for dataset)
        dataset_config = self._get_path("DATASET_CONFIG")
        if not dataset_config:
            errors.append("Dataset config file path is empty — set the training image folder on the Start tab")
        elif not os.path.exists(dataset_config):
            errors.append(f"Dataset config file does not exist: {dataset_config}")

        vae_model = self._get_path("VAE_MODEL")
        if not vae_model:
            errors.append("VAE model file path is empty (set on the Preferences tab)")
        elif not os.path.exists(vae_model):
            errors.append(f"VAE model file does not exist: {vae_model}")

        dit_model = self._get_path("DIT_MODEL")
        if not dit_model:
            errors.append("DiT model file path is empty (set on the Preferences tab)")
        elif not os.path.exists(dit_model):
            errors.append(f"DiT model file does not exist: {dit_model}")

        # Architecture-specific validation (T5/CLIP are dead for Klein but kept for future flexibility)
        if config["uses_t5"]:
            t5_model = self._get_path("T5_MODEL")
            if not t5_model:
                errors.append("T5 model file path is empty")
            elif not os.path.exists(t5_model):
                errors.append(f"T5 model file does not exist: {t5_model}")

        if config["uses_text_encoder"]:
            text_encoder = self._get_path("TEXT_ENCODER")
            if not text_encoder:
                errors.append("Text encoder file path is empty (set on the Preferences tab)")
            elif not os.path.exists(text_encoder):
                errors.append(f"Text encoder file does not exist: {text_encoder}")

        if config["uses_clip"]:
            clip_model = self._get_path("CLIP_MODEL")
            if not clip_model:
                errors.append("CLIP model file path is empty")
            elif not os.path.exists(clip_model):
                errors.append(f"CLIP model file does not exist: {clip_model}")

        # Validate numeric fields
        try:
            lr = float(self.entries["LEARNING_RATE"].get())
            if lr <= 0:
                errors.append("Learning rate must be positive")
            # When adaptive LR is enabled, starting LR must not exceed max LR
            if hasattr(self, 'adaptive_lr_var') and self.adaptive_lr_var.get():
                try:
                    max_lr_str = self.entries["ADAPTIVE_LR_MAX"].get()
                    min_lr_str = self.entries["ADAPTIVE_LR_MIN"].get()
                    max_lr_val = float(max_lr_str)
                    min_lr_val = float(min_lr_str)
                    if lr > max_lr_val:
                        errors.append(f"Starting Learning Rate ({lr}) exceeds Adaptive Max LR ({max_lr_str}). Lower Learning Rate or raise Max LR.")
                    if min_lr_val >= max_lr_val:
                        errors.append(f"Adaptive Min LR ({min_lr_str}) must be lower than Max LR ({max_lr_str}).")
                except (ValueError, KeyError):
                    errors.append("Adaptive Min/Max LR must be valid numbers.")
        except ValueError:
            errors.append("Learning rate must be a valid number")

        # Context LoRA validation
        ctx_path = self.entries.get("CONTEXT_LORA_PATH").get().strip() if "CONTEXT_LORA_PATH" in self.entries else ""
        if ctx_path:
            if not os.path.exists(ctx_path):
                errors.append(f"Context LoRA file does not exist: {ctx_path}")
            elif not ctx_path.lower().endswith(".safetensors"):
                errors.append(f"Context LoRA must be a .safetensors file: {ctx_path}")
            try:
                ctx_strength = float(self.entries["CONTEXT_LORA_STRENGTH"].get())
                if not (0.0 <= ctx_strength <= 2.0):
                    errors.append(f"Context LoRA Strength ({ctx_strength}) must be between 0.0 and 2.0")
            except (ValueError, KeyError):
                errors.append("Context LoRA Strength must be a valid number")

        try:
            network_dim = int(self.entries["NETWORK_DIM"].get())
            if network_dim <= 0:
                errors.append("Network dim must be a positive integer")
        except ValueError:
            errors.append("Network dim must be a valid integer")

        try:
            network_alpha = float(self.entries["NETWORK_ALPHA"].get())
            if network_alpha < 0:
                errors.append("Network alpha must be non-negative")
        except ValueError:
            errors.append("Network alpha must be a valid number")

        try:
            epochs = int(self.entries["MAX_TRAIN_EPOCHS"].get())
            if epochs <= 0:
                errors.append("Max train epochs must be a positive integer")
        except ValueError:
            errors.append("Max train epochs must be a valid integer")

        try:
            save_epochs = int(self.entries["SAVE_EVERY_N_EPOCHS"].get())
            if save_epochs <= 0:
                errors.append("Save every N epochs must be a positive integer")
        except ValueError:
            errors.append("Save every N epochs must be a valid integer")

        try:
            blocks_swap = self._parse_blocks_swap()
            if blocks_swap < 0:
                errors.append("Blocks swap must be non-negative")
            elif blocks_swap > config["blocks_swap_max"]:
                errors.append(f"Blocks swap ({blocks_swap}) exceeds maximum for {arch} ({config['blocks_swap_max']})")
        except ValueError:
            errors.append("Blocks swap must be a valid integer")

        # Check LoRA name is not empty
        lora_name = self.entries["LORA_NAME"].get()
        if not lora_name or not lora_name.strip():
            errors.append("LoRA name cannot be empty")

        # Check output directory
        output_dir = self.entries["LORA_OUTPUT_DIR"].get()
        if not output_dir:
            errors.append("LoRA output directory is empty")

        # Check resume path if specified
        resume_path = self.entries["RESUME_TRAINING"].get()
        if resume_path and resume_path.strip() and not os.path.exists(resume_path):
            errors.append(f"Resume training path does not exist: {resume_path}")

        if errors:
            error_message = "Please fix the following issues:\n\n" + "\n".join(f"• {e}" for e in errors)
            messagebox.showerror("Validation Error", error_message)
            return False

        return True

    def run_subprocess(self, cmd, name, callback=None):
        """Run a subprocess and handle its output with UTF-8 encoding"""
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"  # flush stdout/stderr line-by-line so log output streams live

        if os.name == 'nt':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            preexec_fn = None
        else:
            creationflags = 0
            preexec_fn = os.setsid

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            env=env,
            creationflags=creationflags,
            preexec_fn=preexec_fn
        )
        self.current_process = process
        if os.name == 'nt':
            self.process_group_id = process.pid

        def read_output(pipe, output_type):
            """Read subprocess output line by line"""
            while True:
                line = pipe.readline()
                if not line:
                    break
                self.master.after(0, self.update_console, f"{name} {output_type}: {line}")
            pipe.close()

        threading.Thread(target=read_output, args=(process.stdout, "STDOUT"), daemon=True).start()
        threading.Thread(target=read_output, args=(process.stderr, "STDERR"), daemon=True).start()

        def check_process():
            """Check subprocess completion"""
            process.wait()
            self.master.after(0, self.update_console, f"{name} process completed.\n")
            self.current_process = None
            # Route training-subprocess exit through the pause/resume state machine
            if name and "training" in name.lower():
                self.master.after(0, self._on_training_subprocess_exited, process.returncode)
            if process.returncode != 0:
                self.master.after(0, self.update_console,
                    f"ERROR: {name} failed with exit code {process.returncode}. Pipeline stopped.\n")
                self.master.after(0, self.stop_samples_watcher)
                return
            if callback:
                callback()

        threading.Thread(target=check_process, daemon=True).start()

    def start_training(self):
        """Start training with sequential cache process execution"""
        # Validate inputs before starting
        if not self.validate_inputs():
            return

        # Reset OOM warning flag for this run
        self._oom_warning_shown = False

        # Auto-uncheck FP8 Base if the Base DiT file is already fp8-quantised
        base_dit_path = self.prefs_vars.get("base_dit", tk.StringVar()).get()
        if "fp8" in os.path.basename(base_dit_path).lower() and self.fp8_var.get():
            self.fp8_var.set(False)
            self.scaled_var.set(False)
            self.toggle_scaled()

        # Snapshot current settings for the "Load Last Train" button
        self._save_last_train_settings()

        # Unload Florence model to free VRAM before training
        if self.florence_model is not None:
            self.unload_florence_model(silent=True)

        # Start samples watcher for live gallery updates
        if self.sample_enabled_var.get():
            self.start_samples_watcher()

        # Clear cache directory before training
        cache_dir = self.dataset_cache_dir_var.get().strip()
        if cache_dir and os.path.isdir(cache_dir):
            try:
                import shutil
                for item in os.listdir(cache_dir):
                    item_path = os.path.join(cache_dir, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
            except Exception as e:
                self.update_console(f"Warning: Could not clear cache: {e}\n")

        # Check for unsupported optimizer
        optimizer_type = self.entries["OPTIMIZER_TYPE"].get()
        if optimizer_type == "came":
            messagebox.showwarning(
                "Warning",
                "The 'came' optimizer is not supported in the current version. Please select another optimizer like 'adamw' or 'adamw8bit'."
            )
            return

        # Get current architecture
        arch = self.architecture_var.get()
        config = ARCHITECTURES.get(arch, ARCHITECTURES["Flux 2 Klein Base 9B"])

        # Validate blocks swap
        try:
            is_auto = self.entries["BLOCKS_SWAP"].get().strip().lower().startswith("auto")
            blocks_swap = self._parse_blocks_swap()
            if is_auto:
                self.update_console(f"Block Swap: Auto detected → {blocks_swap} (based on GPU VRAM)\n")
            if blocks_swap > config["blocks_swap_max"]:
                messagebox.showwarning(
                    "Warning",
                    f"Blocks Swap value ({blocks_swap}) exceeds maximum for {arch} ({config['blocks_swap_max']}). Using maximum value."
                )
                blocks_swap = config["blocks_swap_max"]
                self.entries["BLOCKS_SWAP"].delete(0, tk.END)
                self.entries["BLOCKS_SWAP"].insert(0, str(blocks_swap))
        except ValueError:
            blocks_swap = config["blocks_swap_max"]

        # Update settings from entries
        # Path keys read via _get_path() (sourced from prefs_vars or hidden _dataset_config_var)
        # since the Model Paths section is no longer visible on the Training tab.
        self.settings.update({
            "ARCHITECTURE": arch,
            "MODEL_TYPE": self.entries["MODEL_TYPE"].get() if config["uses_model_type"] else "",
            "LEARNING_RATE": float(self.entries["LEARNING_RATE"].get()),
            "LORA_LR_RATIO": int(self.entries["LORA_LR_RATIO"].get()),
            "NETWORK_DIM": int(self.entries["NETWORK_DIM"].get()),
            "NETWORK_ALPHA": float(self.entries["NETWORK_ALPHA"].get()),
            "MAX_TRAIN_EPOCHS": int(self.entries["MAX_TRAIN_EPOCHS"].get()),
            "SAVE_EVERY_N_EPOCHS": int(self.entries["SAVE_EVERY_N_EPOCHS"].get()),
            "SEED": int(self.entries["SEED"].get()),
            "BLOCKS_SWAP": blocks_swap,
            "DATASET_CONFIG": self._get_path("DATASET_CONFIG"),
            "VAE_MODEL": self._get_path("VAE_MODEL"),
            "CLIP_MODEL": self._get_path("CLIP_MODEL"),
            "T5_MODEL": self._get_path("T5_MODEL"),
            "TEXT_ENCODER": self._get_path("TEXT_ENCODER"),
            "DIT_MODEL": self._get_path("DIT_MODEL"),
            "LORA_OUTPUT_DIR": self.entries["LORA_OUTPUT_DIR"].get(),
            "LORA_NAME": self.entries["LORA_NAME"].get(),
            "RESUME_TRAINING": self.entries["RESUME_TRAINING"].get(),
            "OPTIMIZER_TYPE": optimizer_type,
            "OPTIMIZER_ARGS": self.entries["OPTIMIZER_ARGS"].get(),
            "ATTENTION_MECHANISM": self.entries["ATTENTION_MECHANISM"].get(),
            "LOGGING_DIR": self.entries["LOGGING_DIR"].get(),
            "LOG_WITH": self.entries["LOG_WITH"].get(),
            "LOG_PREFIX": self.entries["LOG_PREFIX"].get(),
            "IMG_IN_TXT_IN_OFFLOADING": self.entries["IMG_IN_TXT_IN_OFFLOADING"].get(),
            "LR_SCHEDULER": self.entries["LR_SCHEDULER"].get(),
            "LR_WARMUP_STEPS": self.entries["LR_WARMUP_STEPS"].get(),
            "LR_DECAY_STEPS": self.entries["LR_DECAY_STEPS"].get(),
            "GRADIENT_ACCUMULATION": self.entries["GRADIENT_ACCUMULATION"].get(),
            "MAX_GRAD_NORM": self.entries["MAX_GRAD_NORM"].get(),
            "NETWORK_DROPOUT": self.entries["NETWORK_DROPOUT"].get(),
            "TIMESTEP_SAMPLING": self.ts_sampling_var.get(),
            "DISCRETE_FLOW_SHIFT": self.entries["DISCRETE_FLOW_SHIFT"].get(),
            "SIGMOID_SCALE": self.entries["SIGMOID_SCALE"].get(),
            "MIN_TIMESTEP": self.entries["MIN_TIMESTEP"].get(),
            "MAX_TIMESTEP": self.entries["MAX_TIMESTEP"].get(),
            "PRESERVE_DISTRIBUTION": self.preserve_dist_var.get(),
            "ADAPTIVE_LR": self.adaptive_lr_var.get(),
            "ADAPTIVE_LR_MIN": self.entries["ADAPTIVE_LR_MIN"].get(),
            "ADAPTIVE_LR_MAX": self.entries["ADAPTIVE_LR_MAX"].get(),
            "WEIGHTING_SCHEME": self.weighting_scheme_var.get(),
            "LOGIT_MEAN": self.entries["LOGIT_MEAN"].get(),
            "LOGIT_STD": self.entries["LOGIT_STD"].get(),
            "MODE_SCALE": self.entries["MODE_SCALE"].get(),
            "METADATA_TITLE": self.entries["METADATA_TITLE"].get(),
            "METADATA_AUTHOR": self.entries["METADATA_AUTHOR"].get(),
            "METADATA_DESCRIPTION": self.entries["METADATA_DESCRIPTION"].get(),
            "METADATA_LICENSE": self.entries["METADATA_LICENSE"].get(),
            "METADATA_TAGS": self.entries["METADATA_TAGS"].get(),
            "FP8": self.fp8_var.get(),
            "SCALED": self.scaled_var.get(),
            "FP8_TEXT_ENCODER": self.fp8_text_encoder_var.get(),
            "ENABLE_BUCKET": self.dataset_enable_bucket_var.get(),
            "BUCKET_NO_UPSCALE": self.dataset_no_upscale_var.get(),
        })

        # Build training command based on architecture
        command = self.build_training_command(config)
        cache_latents_cmd = self.build_cache_latents_command(config)
        cache_text_cmd = self.build_cache_text_command(config)

        self.console_output.configure(state="normal")
        self.console_output.delete(1.0, tk.END)
        self.console_output.configure(state="disabled")

        def on_training_complete():
            """Called when training finishes - cleanup watchers"""
            self.stop_samples_watcher()

        # On resume, skip cache preparation entirely — the cache is already built from the original launch.
        is_resuming = bool(self.settings.get("RESUME_TRAINING", "").strip())
        if self.enable_cache_var.get() and not is_resuming:
            self.update_console(f"Starting cache preparation for {arch}...\n")

            def on_text_encoder_caching_complete():
                self.update_console("Text encoder caching completed.\nStarting training...\n")
                self.run_subprocess(command, "Training", on_training_complete)

            def on_cache_preparation_complete():
                self.update_console("Cache preparation completed.\nStarting text encoder caching...\n")
                self.run_subprocess(cache_text_cmd, "Text Encoder Caching", on_text_encoder_caching_complete)

            self.run_subprocess(cache_latents_cmd, "Cache Preparation", on_cache_preparation_complete)
        else:
            if is_resuming:
                self.update_console("Resuming from saved state — skipping cache preparation (cache already built).\n")
            else:
                self.update_console(f"Starting {arch} training without caching...\n")
            self.run_subprocess(command, "Training", on_training_complete)
        # Mark as running for the pause/resume state machine
        self.training_state = "running"
        self._refresh_training_buttons()

    def build_training_command(self, config):
        """Build the training command based on architecture configuration"""
        arch = self.settings["ARCHITECTURE"]
        if os.name == 'nt':
            accelerate_path = os.path.join(FIZGIG_DIR, "venv", "Scripts", "accelerate.exe")
        else:
            accelerate_path = os.path.join(FIZGIG_DIR, "venv", "bin", "accelerate")
        train_script_path = self._resolve_script(config, "train_script")

        # Auto-detect mixed precision from DiT model filename
        # fp16 model files require fp16 mixed precision, bf16 requires bf16
        dit_path = self.settings["DIT_MODEL"]
        dit_filename = os.path.basename(dit_path).lower()
        if "fp16" in dit_filename:
            mixed_precision = "fp16"
        else:
            mixed_precision = "bf16"

        command = [
            accelerate_path, "launch",
            "--num_cpu_threads_per_process", "2",
            "--mixed_precision", mixed_precision,
            train_script_path,
        ]

        # Architecture-specific parameters
        if arch.startswith("Wan"):
            command.extend(["--task", self.settings["MODEL_TYPE"]])
        elif config["uses_model_version"]:
            command.extend(["--model_version", config["model_version"]])

        command.extend([
            "--dit", self.settings["DIT_MODEL"],
            "--dataset_config", self.settings["DATASET_CONFIG"],
            "--mixed_precision", mixed_precision,
        ])

        # VAE parameter (same flag for all architectures)
        command.extend(["--vae", self.settings["VAE_MODEL"]])

        # Text encoder parameters based on architecture
        if config["uses_text_encoder"]:
            command.extend(["--text_encoder", self.settings["TEXT_ENCODER"]])

        # FP8 base optimization
        if self.settings["FP8"]:
            command.append("--fp8_base")
            if self.settings["SCALED"]:
                command.append("--fp8_scaled")

        # FP8 text encoder
        if self.settings["FP8_TEXT_ENCODER"] and config["fp8_text_encoder_flag"]:
            command.append(config["fp8_text_encoder_flag"])

        command.extend([
            "--blocks_to_swap", str(self.settings["BLOCKS_SWAP"]),
            "--optimizer_type", self.settings["OPTIMIZER_TYPE"],
            "--learning_rate", str(self.settings["LEARNING_RATE"]),
            "--gradient_checkpointing",
            "--max_data_loader_n_workers", "2",
            "--persistent_data_loader_workers",
            "--network_module", config["network_module"],
            "--network_dim", str(self.settings["NETWORK_DIM"]),
            "--network_alpha", str(self.settings["NETWORK_ALPHA"]),
            "--network_args", f"loraplus_lr_ratio={self.settings['LORA_LR_RATIO']}",
            "--timestep_sampling", self.settings["TIMESTEP_SAMPLING"],
        ])

        # Target layers (selective layer training)
        # Block assignments based on empirical testing on Klein 9B:
        #   single_blocks 0-1: composition (layout, structure)
        #   single_blocks 2-11: identity/face (the core face signal)
        #   single_blocks 12-23: style (aesthetic, color, lighting)
        #   double_blocks 0-7: cross-attention (included in All Layers only)
        preset = self.training_preset_var.get() if hasattr(self, 'training_preset_var') else "Full Model"
        STYLE_COMP_PATTERNS = [r".*double_blocks\..*", r".*single_blocks\.[01]\..*"]
        IDENTITY_PATTERNS = [r".*single_blocks\.(1[0-6]|[1-9])\..*"]
        DETAILS_PATTERNS = [r".*single_blocks\.(1[2-9]|2[0-3])\..*"]

        patterns = None
        if preset == "Identity":
            patterns = IDENTITY_PATTERNS
        elif preset in ("Style", "Style+Composition"):
            patterns = STYLE_COMP_PATTERNS
        elif preset == "Details":
            patterns = DETAILS_PATTERNS
        elif preset == "Custom":
            patterns = self._build_custom_training_patterns()
            if patterns is None:
                print("[Warning] Training Preset is Custom but no blocks are selected — training full model.")
        # "Full Model" → patterns stays None (train everything)

        if patterns:
            # Escape backslashes for the shell-parsed network_args value
            quoted = ",".join(f'"{p.replace(chr(92), chr(92) * 2)}"' for p in patterns)
            command.extend(["--network_args", f"include_patterns=[{quoted}]"])

        # Discrete flow shift (not for Flux 2 which uses flux2_shift automatic)
        if config.get("supports_discrete_flow_shift", True):
            command.extend(["--discrete_flow_shift", str(self.settings["DISCRETE_FLOW_SHIFT"])])

        # Sigmoid scale (only meaningful for sigmoid/shift sampling)
        ts_sampling = self.settings["TIMESTEP_SAMPLING"]
        sigmoid_scale = self.settings.get("SIGMOID_SCALE", "1.0")
        if ts_sampling in ("sigmoid", "shift") and sigmoid_scale and sigmoid_scale != "1.0":
            command.extend(["--sigmoid_scale", str(sigmoid_scale)])

        # Timestep range (from user settings, not hardcoded config)
        min_ts = self.settings.get("MIN_TIMESTEP", "")
        max_ts = self.settings.get("MAX_TIMESTEP", "")
        if min_ts:
            command.extend(["--min_timestep", str(min_ts)])
        if max_ts:
            command.extend(["--max_timestep", str(max_ts)])
        if self.settings.get("PRESERVE_DISTRIBUTION", False):
            command.append("--preserve_distribution_shape")

        command.extend([
            "--max_train_epochs", str(self.settings["MAX_TRAIN_EPOCHS"]),
            "--save_every_n_epochs", str(self.settings["SAVE_EVERY_N_EPOCHS"]),
            "--save_state",
            "--seed", str(self.settings["SEED"]),
            "--output_dir", self.settings["LORA_OUTPUT_DIR"],
            "--output_name", self.settings["LORA_NAME"],
            "--pause_flag_path", os.path.join(self.settings["LORA_OUTPUT_DIR"], ".pause_requested"),
        ])

        # Optional parameters
        if self.settings["OPTIMIZER_ARGS"]:
            command.extend(["--optimizer_args", self.settings["OPTIMIZER_ARGS"]])

        # Gradient accumulation (effective batch = batch × this)
        gradient_accum = self.settings.get("GRADIENT_ACCUMULATION", 1)
        if isinstance(gradient_accum, str):
            gradient_accum = int(gradient_accum) if gradient_accum else 1
        if gradient_accum > 1:
            command.extend(["--gradient_accumulation_steps", str(gradient_accum)])

        # Max gradient norm (0 to disable clipping)
        max_grad_norm = self.settings.get("MAX_GRAD_NORM", 1.0)
        if isinstance(max_grad_norm, str):
            max_grad_norm = float(max_grad_norm) if max_grad_norm else 1.0
        if max_grad_norm > 0:
            command.extend(["--max_grad_norm", str(max_grad_norm)])

        # Network dropout (LoRA regularization)
        network_dropout = self.settings.get("NETWORK_DROPOUT", 0)
        if isinstance(network_dropout, str):
            network_dropout = float(network_dropout) if network_dropout else 0
        if network_dropout > 0:
            command.extend(["--network_dropout", str(network_dropout)])

        # Attention mechanism (user's choice, default is "sdpa")
        attention = self.settings["ATTENTION_MECHANISM"]
        if attention != "none":
            command.append(f"--{attention}")

        logging_dir = self.settings["LOGGING_DIR"]
        if logging_dir:
            command.extend(["--logging_dir", logging_dir])

        log_with = self.settings["LOG_WITH"]
        if log_with != "none":
            command.extend(["--log_with", log_with])

        log_prefix = self.settings["LOG_PREFIX"]
        if log_prefix:
            command.extend(["--log_prefix", log_prefix])

        if self.settings["IMG_IN_TXT_IN_OFFLOADING"]:
            command.append("--img_in_txt_in_offloading")

        # Adaptive LR overrides the step-based scheduler — force constant pre-phase.
        adaptive_on = bool(self.settings.get("ADAPTIVE_LR", False))
        lr_scheduler = "constant" if adaptive_on else self.settings["LR_SCHEDULER"]
        if lr_scheduler:
            command.extend(["--lr_scheduler", lr_scheduler])

        if not adaptive_on:
            lr_warmup_steps = self.settings["LR_WARMUP_STEPS"]
            if lr_warmup_steps:
                command.extend(["--lr_warmup_steps", lr_warmup_steps])

            lr_decay_steps = self.settings["LR_DECAY_STEPS"]
            if lr_decay_steps:
                command.extend(["--lr_decay_steps", lr_decay_steps])

        if adaptive_on:
            command.append("--adaptive_lr")
            min_lr = self.settings.get("ADAPTIVE_LR_MIN", "1e-5") or "1e-5"
            max_lr = self.settings.get("ADAPTIVE_LR_MAX", "4e-4") or "4e-4"
            command.extend(["--adaptive_lr_min", str(min_lr)])
            command.extend(["--adaptive_lr_max", str(max_lr)])

        # Context LoRA — train new LoRA with an existing one frozen + active
        ctx_path = self.settings.get("CONTEXT_LORA_PATH", "").strip()
        if ctx_path:
            command.extend(["--context_lora_path", ctx_path])
            ctx_strength = self.settings.get("CONTEXT_LORA_STRENGTH", "1.0") or "1.0"
            command.extend(["--context_lora_strength", str(ctx_strength)])

        weighting_scheme = self.settings["WEIGHTING_SCHEME"]
        if weighting_scheme != "none":
            command.extend(["--weighting_scheme", weighting_scheme])
            if weighting_scheme == "logit_normal":
                logit_mean = self.settings.get("LOGIT_MEAN", "0.0")
                logit_std = self.settings.get("LOGIT_STD", "1.0")
                if logit_mean and logit_mean != "0.0":
                    command.extend(["--logit_mean", str(logit_mean)])
                if logit_std and logit_std != "1.0":
                    command.extend(["--logit_std", str(logit_std)])
            elif weighting_scheme == "mode":
                mode_scale = self.settings.get("MODE_SCALE", "1.29")
                if mode_scale and mode_scale != "1.29":
                    command.extend(["--mode_scale", str(mode_scale)])

        # Metadata
        metadata_title = self.settings["METADATA_TITLE"]
        if metadata_title:
            command.extend(["--metadata_title", metadata_title])

        metadata_author = self.settings["METADATA_AUTHOR"]
        if metadata_author:
            command.extend(["--metadata_author", metadata_author])

        metadata_description = self.settings["METADATA_DESCRIPTION"]
        if metadata_description:
            command.extend(["--metadata_description", metadata_description])

        metadata_license = self.settings["METADATA_LICENSE"]
        if metadata_license:
            command.extend(["--metadata_license", metadata_license])

        metadata_tags = self.settings["METADATA_TAGS"]
        if metadata_tags:
            command.extend(["--metadata_tags", metadata_tags])

        if self.settings["RESUME_TRAINING"].strip():
            command.append(f"--resume={self.settings['RESUME_TRAINING']}")

        # Sample generation (only if enabled and architecture supports it)
        if self.sample_enabled_var.get() and config.get("supports_samples", False):
            # Generate prompt file
            prompt_file = self.generate_sample_prompt_file()
            command.extend(["--sample_prompts", prompt_file])

            # Frequency settings
            every_n_epochs = self.sample_every_n_epochs_var.get()
            if every_n_epochs and int(every_n_epochs) > 0:
                command.extend(["--sample_every_n_epochs", every_n_epochs])

            every_n_steps = self.sample_every_n_steps_var.get()
            if every_n_steps and int(every_n_steps) > 0:
                command.extend(["--sample_every_n_steps", every_n_steps])

            if self.sample_at_first_var.get():
                command.append("--sample_at_first")

        return command

    def build_cache_latents_command(self, config):
        """Build the cache latents command based on architecture"""
        arch = self.settings["ARCHITECTURE"]
        if os.name == 'nt':
            python_path = os.path.join(FIZGIG_DIR, "venv", "Scripts", "python.exe")
        else:
            python_path = os.path.join(FIZGIG_DIR, "venv", "bin", "python")
        cache_script_path = self._resolve_script(config, "cache_latents_script")

        command = [
            python_path,
            cache_script_path,
            "--dataset_config", self.settings["DATASET_CONFIG"],
            "--vae", self.settings["VAE_MODEL"],
        ]

        # Wan needs CLIP for latent caching
        if config["uses_clip"]:
            command.extend(["--clip", self.settings["CLIP_MODEL"]])

        # Flux 2 needs model version
        if config["uses_model_version"]:
            command.extend(["--model_version", config["model_version"]])

        return command

    def build_cache_text_command(self, config):
        """Build the cache text encoder command based on architecture"""
        arch = self.settings["ARCHITECTURE"]
        if os.name == 'nt':
            python_path = os.path.join(FIZGIG_DIR, "venv", "Scripts", "python.exe")
        else:
            python_path = os.path.join(FIZGIG_DIR, "venv", "bin", "python")
        cache_script_path = self._resolve_script(config, "cache_text_script")

        command = [
            python_path,
            cache_script_path,
            "--dataset_config", self.settings["DATASET_CONFIG"],
        ]

        # Different text encoder parameters based on architecture
        if config["uses_t5"]:
            command.extend(["--t5", self.settings["T5_MODEL"]])
        elif config["uses_text_encoder"]:
            command.extend(["--text_encoder", self.settings["TEXT_ENCODER"]])

        command.extend(["--batch_size", "16"])

        # FP8 text encoder flag
        if self.settings["FP8_TEXT_ENCODER"] and config["fp8_text_encoder_flag"]:
            command.append(config["fp8_text_encoder_flag"])

        # Flux 2 needs model version
        if config["uses_model_version"]:
            command.extend(["--model_version", config["model_version"]])

        return command

    # === Pause / Resume support ===

    def _pause_flag_path(self) -> str:
        """Path to the pause sentinel file in the current output directory."""
        out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
        return os.path.join(out_dir, ".pause_requested")

    def _paused_sidecar_path(self) -> str:
        """Path to the JSON sidecar that records paused-state metadata."""
        out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
        return os.path.join(out_dir, ".fizgig_paused.json")

    def _refresh_training_buttons(self):
        """Show/hide Pause and Resume buttons based on self.training_state."""
        if not hasattr(self, "training_state"):
            self.training_state = "idle"
        # Pause: visible while running
        if self.training_state == "running":
            try: self._pause_training_btn.pack(side=tk.LEFT, padx=(0, 12), after=self._start_training_btn)
            except Exception: pass
        else:
            try: self._pause_training_btn.pack_forget()
            except Exception: pass
        # Resume: visible while paused
        if self.training_state == "paused":
            try: self._resume_training_btn.pack(side=tk.LEFT, padx=(0, 12), after=self._start_training_btn)
            except Exception: pass
        else:
            try: self._resume_training_btn.pack_forget()
            except Exception: pass

    def _pause_training(self):
        """Request a graceful pause — trainer will save state at end of current epoch and exit."""
        if not getattr(self, "current_process", None) or self.current_process.poll() is not None:
            messagebox.showinfo("Not Running", "No active training to pause.")
            return
        if getattr(self, "training_state", "idle") != "running":
            return
        try:
            os.makedirs(os.path.dirname(self._pause_flag_path()) or ".", exist_ok=True)
            open(self._pause_flag_path(), "w").close()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write pause flag:\n{e}")
            return
        self.update_console(
            "\n=== PAUSE REQUESTED — trainer will save full state and exit cleanly at end of current epoch. "
            "GPU memory will be freed. Click Resume Training afterwards to continue. ===\n\n"
        )
        messagebox.showinfo(
            "Pause Requested",
            "Pause queued. The trainer will finish the CURRENT epoch, save full state, "
            "and exit cleanly to free GPU memory.\n\n"
            "Click Resume Training afterwards to continue.",
        )
        self.training_state = "pausing"
        self._refresh_training_buttons()

    def _detect_latest_state_dir(self):
        """Find the highest-numbered <output_name>-NNNNNN-state/ directory in the output dir."""
        import re as _re
        out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
        out_name = self.settings.get("LORA_NAME", "") or ""
        if not out_name or not os.path.isdir(out_dir):
            return None
        pattern = _re.compile(rf"^{_re.escape(out_name)}-(\d{{6}})-state$")
        candidates = []
        try:
            for entry in os.listdir(out_dir):
                m = pattern.match(entry)
                if m and os.path.isdir(os.path.join(out_dir, entry)):
                    candidates.append((int(m.group(1)), entry))
        except Exception:
            return None
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return os.path.join(out_dir, candidates[0][1])

    def _on_training_subprocess_exited(self, return_code: int):
        """Called from check_process when the training subprocess ends. Routes to paused or idle."""
        # Clean up pause flag if still present
        try:
            flag = self._pause_flag_path()
            if os.path.exists(flag):
                os.remove(flag)
        except Exception:
            pass
        if getattr(self, "training_state", "idle") == "pausing" and return_code == 0:
            # Successful graceful exit — record paused state
            state_dir = self._detect_latest_state_dir()
            if state_dir is None:
                self.update_console("[pause] WARN: no state directory found after pause exit. Treating as idle.\n")
                self.training_state = "idle"
            else:
                self.paused_state_path = state_dir
                # Persist sidecar so paused state survives GUI restart
                try:
                    import json as _json
                    out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
                    sidecar = {
                        "state_path": state_dir,
                        "output_name": self.settings.get("LORA_NAME", ""),
                        "dataset_config": self.settings.get("DATASET_CONFIG", ""),
                        "network_dim": str(self.settings.get("NETWORK_DIM", "")),
                        "network_alpha": str(self.settings.get("NETWORK_ALPHA", "")),
                        "max_train_epochs": str(self.settings.get("MAX_TRAIN_EPOCHS", "")),
                    }
                    with open(self._paused_sidecar_path(), "w") as f:
                        _json.dump(sidecar, f, indent=2)
                except Exception as e:
                    self.update_console(f"[pause] WARN: failed to write sidecar: {e}\n")
                self.training_state = "paused"
                self.update_console(
                    f"\n=== PAUSED — state saved at {state_dir}. Click Resume Training to continue. ===\n\n"
                )
        else:
            self.training_state = "idle"
        self._refresh_training_buttons()

    def _resume_training(self):
        """Re-launch training from the latest paused state directory."""
        if getattr(self, "training_state", "idle") != "paused":
            messagebox.showinfo("Not Paused", "No paused training to resume.")
            return
        state_path = getattr(self, "paused_state_path", None)
        if not state_path or not os.path.isdir(state_path):
            messagebox.showerror("Error", f"Paused state directory not found:\n{state_path}")
            return
        # Inject resume path into settings + entry field, then reuse the standard start_training flow
        self.settings["RESUME_TRAINING"] = state_path
        try:
            entry = self.entries.get("RESUME_TRAINING")
            if entry is not None:
                entry.delete(0, tk.END)
                entry.insert(0, state_path)
        except Exception:
            pass
        # Clean up paused sidecar — we're consuming it
        try:
            sidecar = self._paused_sidecar_path()
            if os.path.exists(sidecar):
                os.remove(sidecar)
        except Exception:
            pass
        self.update_console(f"\n=== RESUMING from {state_path} ===\n\n")
        self.training_state = "running"
        self._refresh_training_buttons()
        self.start_training()

    def _check_for_paused_state_on_startup(self):
        """On GUI launch, detect a leftover paused state and restore the Resume button."""
        try:
            sidecar = self._paused_sidecar_path()
            if not os.path.exists(sidecar):
                return
            import json as _json
            with open(sidecar, "r") as f:
                meta = _json.load(f)
            state_path = meta.get("state_path", "")
            if state_path and os.path.isdir(state_path):
                self.paused_state_path = state_path
                self.training_state = "paused"
                self._refresh_training_buttons()
                self.update_console(
                    f"=== Paused training detected: {meta.get('output_name','?')} "
                    f"at state {os.path.basename(state_path)}. Click Resume Training to continue. ===\n"
                )
        except Exception:
            pass

    def stop_training(self):
        """Stop the current running process"""
        # Stop samples watcher
        self.stop_samples_watcher()

        if self.current_process and self.current_process.poll() is None:
            try:
                if os.name == 'nt':
                    self.current_process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
            except Exception as e:
                self.update_console("Error stopping process: " + str(e) + "\n")
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self.current_process.kill()
                    self.current_process.wait()
                except Exception as e:
                    self.update_console("Error killing process: " + str(e) + "\n")
            self.current_process = None
            if self.training_thread:
                self.training_thread.join(timeout=1)
                self.training_thread = None
            self.update_console("Training stopped\n")
        else:
            self.update_console("No active process to stop\n")

    def save_settings(self):
        """Save all settings, including conversion settings, to a JSON file"""
        current_settings = {}
        for key, entry in self.entries.items():
            if isinstance(entry, ttk.Combobox):
                current_settings[key] = entry.get()
            elif isinstance(entry, tk.BooleanVar):
                current_settings[key] = entry.get()
            else:
                current_settings[key] = entry.get()
        current_settings["ARCHITECTURE"] = self.architecture_var.get()
        current_settings["PRESERVE_DISTRIBUTION"] = self.preserve_dist_var.get()
        current_settings["ADAPTIVE_LR"] = self.adaptive_lr_var.get()
        current_settings["BILINGUAL_SKIP_EXISTING"] = self.skip_bilingual_var.get()
        current_settings["FP8"] = self.fp8_var.get()
        current_settings["SCALED"] = self.scaled_var.get()
        current_settings["FP8_TEXT_ENCODER"] = self.fp8_text_encoder_var.get()
        current_settings["ENABLE_BUCKET"] = self.dataset_enable_bucket_var.get()
        current_settings["BUCKET_NO_UPSCALE"] = self.dataset_no_upscale_var.get()
        current_settings["ENABLE_CACHE"] = self.enable_cache_var.get()

        # Training preset + custom block selections
        if hasattr(self, 'training_preset_var'):
            current_settings["TARGET_LAYERS"] = self.training_preset_var.get()
        if hasattr(self, 'training_block_vars'):
            current_settings["TRAINING_CUSTOM_BLOCKS"] = {
                k: v.get() for k, v in self.training_block_vars.items()
            }

        # Save sample settings
        current_settings["SAMPLE_PROMPT"] = self.sample_prompt_text.get("1.0", tk.END).strip()

        # Save caption settings
        current_settings["CAPTION_TRIGGER_WORD"] = self.caption_trigger_var.get()
        current_settings["CAPTION_MODEL"] = self.caption_model_var.get()
        current_settings["CAPTION_TASK"] = self.caption_task_var.get()
        current_settings["CAPTION_MAX_TOKENS"] = self.caption_max_tokens_var.get()

        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as f:
                json.dump(current_settings, f, indent=4)

    def load_settings(self):
        """Load settings from a JSON file, including conversion settings"""
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as f:
                loaded_settings = json.load(f)

            # Load architecture first to update UI
            if "ARCHITECTURE" in loaded_settings:
                self.architecture_var.set(loaded_settings["ARCHITECTURE"])
                self.update_ui_for_architecture()

            for key, value in loaded_settings.items():
                if key in self.entries:
                    if isinstance(self.entries[key], ttk.Combobox):
                        self.entries[key].set(value)
                    elif isinstance(self.entries[key], tk.BooleanVar):
                        self.entries[key].set(value)
                    else:
                        self.entries[key].delete(0, tk.END)
                        self.entries[key].insert(0, value)
            if "FP8" in loaded_settings:
                self.fp8_var.set(loaded_settings["FP8"])
            if "SCALED" in loaded_settings:
                self.scaled_var.set(loaded_settings["SCALED"])
            if "FP8_TEXT_ENCODER" in loaded_settings:
                self.fp8_text_encoder_var.set(loaded_settings["FP8_TEXT_ENCODER"])
            if "ENABLE_BUCKET" in loaded_settings:
                self.dataset_enable_bucket_var.set(loaded_settings["ENABLE_BUCKET"])
            if "BUCKET_NO_UPSCALE" in loaded_settings:
                self.dataset_no_upscale_var.set(loaded_settings["BUCKET_NO_UPSCALE"])
            if "ENABLE_CACHE" in loaded_settings:
                self.enable_cache_var.set(loaded_settings["ENABLE_CACHE"])

            # Load sample settings
            if "SAMPLE_PROMPT" in loaded_settings:
                self.sample_prompt_text.delete("1.0", tk.END)
                self.sample_prompt_text.insert("1.0", loaded_settings["SAMPLE_PROMPT"])

            # Load caption settings
            if "CAPTION_TRIGGER_WORD" in loaded_settings:
                self.caption_trigger_var.set(loaded_settings["CAPTION_TRIGGER_WORD"])
            if "CAPTION_MODEL" in loaded_settings:
                self.caption_model_var.set(loaded_settings["CAPTION_MODEL"])
            if "CAPTION_TASK" in loaded_settings:
                self.caption_task_var.set(loaded_settings["CAPTION_TASK"])
            if "CAPTION_MAX_TOKENS" in loaded_settings:
                self.caption_max_tokens_var.set(loaded_settings["CAPTION_MAX_TOKENS"])

            # Load timestep boolean settings
            if "PRESERVE_DISTRIBUTION" in loaded_settings:
                self.preserve_dist_var.set(loaded_settings["PRESERVE_DISTRIBUTION"])
            if "ADAPTIVE_LR" in loaded_settings:
                self.adaptive_lr_var.set(bool(loaded_settings["ADAPTIVE_LR"]))
                self._on_adaptive_lr_toggle()
            if "BILINGUAL_SKIP_EXISTING" in loaded_settings:
                self.skip_bilingual_var.set(bool(loaded_settings["BILINGUAL_SKIP_EXISTING"]))

            # Back-compat: old settings JSONs stored model paths under Training-tab keys.
            # Route them into the prefs_vars (the new source of truth) so users don't lose paths on upgrade.
            for old_key, pref_key in (("VAE_MODEL", "vae"), ("DIT_MODEL", "base_dit"),
                                       ("TEXT_ENCODER", "text_encoder"), ("LORA_OUTPUT_DIR", "lora_output_dir")):
                if old_key in loaded_settings and pref_key in self.prefs_vars:
                    self.prefs_vars[pref_key].set(loaded_settings[old_key])
            if "DATASET_CONFIG" in loaded_settings and hasattr(self, "_dataset_config_var"):
                self._dataset_config_var.set(loaded_settings["DATASET_CONFIG"])

            # Training preset + custom blocks (with back-compat for old names)
            if "TARGET_LAYERS" in loaded_settings and hasattr(self, 'training_preset_var'):
                raw = loaded_settings["TARGET_LAYERS"]
                legacy_map = {
                    "All Layers": "Full Model",
                    "Identity Blocks": "Identity",
                    "Style+Composition Blocks": "Style+Composition",
                    "Details Blocks": "Details",
                }
                mapped = legacy_map.get(raw, raw)
                valid = ("Full Model", "Identity", "Style", "Style+Composition", "Details", "Custom")
                self.training_preset_var.set(mapped if mapped in valid else "Full Model")
                self._on_training_preset_changed()
            if "TRAINING_CUSTOM_BLOCKS" in loaded_settings and hasattr(self, 'training_block_vars'):
                saved_blocks = loaded_settings["TRAINING_CUSTOM_BLOCKS"]
                for k, v in saved_blocks.items():
                    if k in self.training_block_vars:
                        self.training_block_vars[k].set(bool(v))

            self.toggle_scaled()  # Update Scaled checkbox state based on FP8
            if hasattr(self, 'ts_sampling_var'):
                self._on_timestep_sampling_changed()
                self._on_weighting_scheme_changed()
                self._update_noise_range_label()
            if hasattr(self, 'sample_settings_frame'):
                self.update_samples_ui_for_architecture()

root = tk.Tk()
gui = LoRATrainerGUI(root)
# Detect leftover paused training state from a prior session
try:
    gui._check_for_paused_state_on_startup()
except Exception:
    pass
root.mainloop()
