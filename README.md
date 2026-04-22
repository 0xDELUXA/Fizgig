<h1 align="center">Fizgig — Klein 9B LoRA Studio</h1>

<p align="center">
  A focused, local trainer and workbench for <strong>Flux 2 Klein 9B</strong> LoRAs.<br>
  Train, profile, repair, and extract — all in one Tkinter app.
</p>

<p align="center">
  <a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
</p>

<p align="center">
  <a href="https://youtu.be/sH-kGR8yzBU"><img src="https://img.youtube.com/vi/sH-kGR8yzBU/maxresdefault.jpg" alt="Watch the walkthrough" width="600"></a><br>
  <em>Watch the full walkthrough on YouTube</em>
</p>

---

Fizgig is self-contained: no external CLI, no wrapper. Training, inference, profiling, extraction, and the live Repair Studio all run through the bundled `src/fizgig/` pipelines.

## Features

- **Train Klein 9B LoRAs** with proven presets (rank 4–16, single subject through multi-character).
- **Adaptive LR** — a bi-directional plateau tracker that probes up on steady loss and pulls down on plateau / grad-clip / weight-norm runaway, with optional weights rollback on stability events.
- **Context LoRA training** — bake an existing LoRA in frozen as an active layer, so the new LoRA learns to coexist at inference (no other trainer does this).
- **Pause / Resume** — graceful epoch-boundary pause that frees VRAM and resumes with no quality regression.
- **AI captions** — Florence-2 bulk captioning + optional English → Chinese bilingual translation via Helsinki-NLP.
- **Image Prep** — batch resize, PNG conversion, face-crop derivatives via InsightFace.
- **Profiler** — per-block activation profile with a 5-bucket HTML report, and a JSON sidecar the Repair Studio reads inline.
- **Repair Studio** — 32 live sliders per LoRA block with side-by-side Distilled preview, optional donor-LoRA blending (rank-concatenation bake), quick-set `[0]` `[1]` `[±]` buttons on every slider. **Turbo Preview** caches activations and prompt encodings for near-instant updates when tweaking individual blocks — up to 97% faster on late-block changes. Click the tweaked preview to pop it out into a resizable window for a closer look. Auto-unloads models when you switch tabs to free VRAM.
- **Extract** — distill any Klein LoRA down to lower rank with block + timestep targeting. Fast presets run pure weight SVD with no pipeline loaded; activation-weighted presets use GPU forward passes for better accuracy. Supports PEFT and LyCORIS (LoKR / LoHa) sources.
- **Output** — saves kohya-style weight keys. Drop straight into ComfyUI, no conversion step.
- **YouTube help** — every tab has a help button linking to a video walkthrough.

---

## Requirements

- **GPU** — NVIDIA RTX 30-series, 40-series, or 50-series (Blackwell). **16 GB+ VRAM** recommended (24 GB+ comfortable).
- **NVIDIA driver** — 555 or newer on Windows, 550 or newer on Linux. (Required for CUDA 12.8 PyTorch wheels.)
- **OS** — Windows 10 / 11, or Linux. (Install script supports macOS for captioning/prep but Klein training requires CUDA.)
- **Python** — 3.10, 3.11, 3.12, or 3.13.
- **Disk** — ~10 GB for the venv, plus ~40 GB for the model files (see below).

---

## Install

### Windows (one-click)

Double-click `install_fizgig.bat`. The installer creates a venv, installs CUDA 12.8 PyTorch + all dependencies, pre-downloads InsightFace face-detection models, and verifies that CUDA is visible to PyTorch.

Launch with `run_fizgig.bat` when install completes.

### Linux / macOS

```bash
python install_fizgig.py
chmod +x run_fizgig.sh
./run_fizgig.sh
```

---

## Model downloads (you provide)

Fizgig doesn't bundle model weights — they're ~40 GB combined and licensing varies. Each row in the **Preferences** tab has a **Download** link that opens the correct HuggingFace page in your browser.

| Model | File | Size | Source |
|---|---|---|---|
| Base DiT | `flux-2-klein-base-9b.safetensors` | ~17 GB bf16 | [black-forest-labs/FLUX.2-klein-dev](https://huggingface.co/black-forest-labs/FLUX.2-klein-dev) |
| Distilled DiT | `flux-2-klein-9b-fp8.safetensors` | ~9 GB fp8 | [Comfy-Org/flux2_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/flux2_ComfyUI_repackaged) |
| VAE / AE | `ae.safetensors` | ~320 MB | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) (from root, **not** the `vae/` subfolder) |
| Text Encoder | Qwen3-8B single-file safetensors | ~15 GB | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |

Training runs on the Base DiT. The Distilled DiT is used for 4-step sample previews during training, the Profiler, the Extractor, and the Repair Studio — so you want both if you'll use the workbench features.

Three small models auto-download on first use: InsightFace (`buffalo_l`, ~300 MB, during installer), Florence-2 (~500 MB–1.5 GB, first time you caption with AI), Helsinki-NLP/opus-mt-en-zh (~300 MB, first time you translate captions).

---

## Getting started

After install, launch Fizgig and work left-to-right through the numbered tabs:

1. **Start** — friendly intro + a single *"Training image folder"* picker. This is the single source of truth shared with Image Prep, Captions, and the training config.
2. **Image Prep** (optional) — resize, PNG-convert, or face-crop your training images.
3. **Captions** — write trigger-word captions or generate with Florence-2 AI. Optional bilingual English+Chinese translation.
4. **Samples** — set the preview prompts that render during training (Distilled 4-step).
5. **Training** — pick a preset, tune, click **Start Training**.

The unnumbered tabs are post-training tools:

- **Profiler** — analyze any Klein LoRA (your own or a download).
- **Repair Studio** — live-tweak a LoRA's per-block contributions with side-by-side preview; bake to a new `.safetensors`.
- **Extract** — distill to a lower rank with optional block / timestep targeting.
- **Preferences** — model paths, output directories, inference block-swap for 16 GB cards.

---

## VRAM guidance

Inference (Profiler / Repair Studio / Extract) on Distilled 4-step:

| Block Swap | Min VRAM | Notes |
|---|---|---|
| 0 | 24 GB+ | No swap — fastest |
| 4 | 20 GB | Light swap |
| 8 | 16 GB | Moderate |
| 12 | 14 GB | Aggressive |
| 16 | 12 GB | Max swap — slow but runs |

**DiT Block Swap (inference)** in Preferences applies to all three tool tabs. Training has its own separate *Blocks Swap* setting on the Training tab. On first launch, Fizgig auto-detects your GPU VRAM and picks a sensible default — once you explicitly choose a value, your choice is saved and the auto-detect is bypassed.

---

## Support the project

If Fizgig saves you time or helps you make better LoRAs, consider supporting development:

<a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>

---