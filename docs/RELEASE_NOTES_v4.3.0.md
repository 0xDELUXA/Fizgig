# Fizgig v4.3.0 — AMD support arrives

Fizgig now trains on AMD Radeon cards. Plus: 16 GB cards get reference-mode training, and the
MiniMax presets got a rethink.

## AMD ROCm support — thanks @scryptio (#53)

Fizgig runs on AMD Radeon with ROCm — RDNA1 through RDNA4, Strix Point / Halo, and Instinct.

- **Windows** is the supported path: install Python 3.12, double-click `install_fizgig_rocm.bat`,
  launch with `run_fizgig_rocm.bat`. The installer detects your GPU and pulls the right wheels.
- **Linux** is available but highly experimental — driver resets and crashes are common on newer
  cards. Use Windows ROCm or NVIDIA Linux for production training.
- The status-bar VRAM readout works on AMD too, and RDNA4 cards get a known ROCm GEMM slowdown
  worked around automatically.
- Full install details are in the README. NVIDIA installs are completely untouched — the AMD
  path is separate files, separate venv steps.

This was a lot of work by @scryptio, tested along the way by @tsubasasora on Linux and sharpened
by @FNGarvin and @taisunyoung in the PR thread. Thank you all.

## 16 GB cards: identity distillation now fits — thanks @rintic-13 (#79)

Reference-mode caching (the teacher for identity distillation) used to peak at ~26 GB — out of
reach below a 32 GB card. It now streams the text encoder layer by layer and peaks at ~12.7 GB,
with output verified bit-for-bit identical to the old path. Nothing to configure: Fizgig
measures your free VRAM and streams only when the resident path wouldn't fit. This closes #74 —
@rintic-13's second major contribution to the 16 GB story, after the int8 streaming in v4.0.

## MiniMax H3 presets, reshuffled

- **✨ MiniMax H3 Fast is now the default** — it's what loads when you pick the family. It also
  trains 50 epochs now (was 40).
- The rank-16 preset is renamed **✨ MiniMax H3 (Lower LR - slower)**, with a note beside the
  dropdown: more suitable for larger datasets with longer trains.
- The Style preset follows Fast's recipe on the measured style blocks, as before.

## Also

- **Korean localization** — a community add-on by @ssain3d-lgtm translates the whole UI to
  Korean without touching a single Fizgig file. Linked in the README under Getting started.

## Upgrading

Nothing to do. Settings, models, caches and presets are untouched. MiniMax users will see Fast
selected by default on the Training tab — your own saved presets are exactly where you left them.
