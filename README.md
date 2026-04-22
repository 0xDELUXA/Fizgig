<h1 align="center">Fizgig — Klein 9B LoRA Studio</h1>

<p align="center">
  Fix broken LoRAs without retraining. Remix any LoRA into new variations in seconds.<br>
  Train, profile, repair, and explore — all in one app for <strong>Flux 2 Klein 9B</strong>.
</p>

<p align="center">
  <a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
</p>

<p align="center">
  <a href="https://youtu.be/sH-kGR8yzBU"><img src="https://img.youtube.com/vi/sH-kGR8yzBU/maxresdefault.jpg" alt="Watch the walkthrough" width="600"></a><br>
  <em>Watch the full walkthrough on YouTube</em>
</p>

---

### What makes Fizgig different?

- **Fix broken LoRAs without retraining** — overbaked identity? crushed style? Adjust per-block sliders with live side-by-side preview and save a repaired `.safetensors` in seconds.
- **Explore variations like a game** — the computer proposes random mutations, you pick favourites, and the LoRA evolves through selection. Hold Mode locks your picks and narrows the search until every block is sculpted.
- **See exactly what a LoRA does** — the Profiler shows which transformer blocks carry style, identity, and detail signal, so you know what to fix before you touch a slider.
- **Train with intelligence** — adaptive learning rate adjusts itself based on loss, gradient clipping, and weight-norm growth. Context LoRA lets you train on top of an existing frozen LoRA so they coexist at inference.

---

## Features

### Repair Studio
32 live sliders — one per transformer block — with side-by-side Distilled preview. See the effect of every change instantly. Optional donor-LoRA blending via rank concatenation lets you mix blocks from two LoRAs. Quick-set `[0]` `[1]` `[±]` buttons on every slider. **Turbo Preview** caches activations and prompt encodings — up to 97% faster on late-block changes. Click the tweaked preview to pop it out into a resizable window. Saves a baked `.safetensors` that works in ComfyUI at strength 1.0.

### LoRA the Explorer
Evolutionary LoRA discovery. The computer randomly mutates blocks and shows you 4 variants — pick your favourite and it becomes the new baseline. **Hold Mode** locks picked blocks so the search space narrows with each selection, sculpting the LoRA until every block is dialled in. Seed cycling verifies variants across different seeds. Adjustable intensity and mutation count for fine or bold exploration.

### Profiler
Per-block activation profile with a colour-coded 5-bucket HTML report. Identifies which blocks carry style, identity, and detail signal — and where they overlap. Writes a JSON sidecar that the Repair Studio reads automatically, showing you the profiler's findings inline when you load the same LoRA.

### Training
- **Proven presets** for rank 4–16, single subject through multi-character — or build your own.
- **Adaptive LR** — bi-directional plateau tracker that probes up on steady loss descent and pulls down (with optional weight rollback) on plateau, heavy gradient clipping, or weight-norm runaway.
- **Context LoRA** — load an existing LoRA as a frozen active layer during training, so the new LoRA learns to coexist at inference. No other trainer does this.
- **Pause / Resume** — graceful epoch-boundary pause that frees VRAM and resumes with full optimizer state and no quality regression.
- **Model Area targeting** — train only Identity blocks, Style blocks, Details blocks, or the full model.

### Dataset Prep
- **Florence-2 AI captioning** — bulk-generate detailed captions with one click.
- **Bilingual translation** — optionally append Chinese translations via Helsinki-NLP. Klein's Qwen3 text encoder has deep Chinese training; bilingual captions act as text-level data augmentation, improving visual quality without changing loss.
- **Image Prep** — batch resize, PNG conversion, and face-crop derivatives via InsightFace.

### Extract
Distil any Klein LoRA down to a lower rank with block and timestep targeting. Fast presets run pure weight SVD with no GPU models loaded; activation-weighted presets use forward passes for better accuracy. Supports PEFT and LyCORIS (LoKR / LoHa) sources.

### Compatibility
- **Formats** — loads kohya, PEFT, and LyCORIS (LoKR / LoHa) LoRAs. PEFT keys auto-converted on load. LyCORIS files work for preview, profiling, and extraction; bake converts them to standard LoRA via SVD.
- **Output** — kohya-style `.safetensors` that drop straight into ComfyUI Klein nodes.
- **YouTube help** — every tab links to the relevant section of the walkthrough video.

---

## Requirements

- **GPU** — NVIDIA RTX 30-series, 40-series, or 50-series (Blackwell). **16 GB+ VRAM** recommended (24 GB+ comfortable).
- **NVIDIA driver** — 555+ on Windows, 550+ on Linux (required for CUDA 12.8 PyTorch wheels).
- **OS** — Windows 10 / 11, or Linux. macOS works for captioning and image prep but training requires CUDA.
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

Fizgig doesn't bundle model weights — they're ~40 GB combined and licensing varies. Each row in the **Preferences** tab has a **Download** link that opens the correct HuggingFace page.

| Model | File | Size | Source |
|---|---|---|---|
| Base DiT | `flux-2-klein-base-9b.safetensors` | ~17 GB bf16 | [black-forest-labs/FLUX.2-klein-dev](https://huggingface.co/black-forest-labs/FLUX.2-klein-dev) |
| Distilled DiT | `flux-2-klein-9b-fp8.safetensors` | ~9 GB fp8 | [Comfy-Org/flux2_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/flux2_ComfyUI_repackaged) |
| VAE / AE | `ae.safetensors` | ~320 MB | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) (from root, **not** the `vae/` subfolder) |
| Text Encoder | Qwen3-8B single-file safetensors | ~15 GB | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |

Training runs on the **Base DiT**. The **Distilled DiT** is used for fast 4-step previews during training, the Profiler, the Repair Studio, and the Explorer — so you want both if you'll use the workbench features.

Three smaller models auto-download on first use: InsightFace (`buffalo_l`, ~300 MB, during install), Florence-2 (~500 MB–1.5 GB, first AI caption), Helsinki-NLP/opus-mt-en-zh (~300 MB, first bilingual translation).

---

## Getting started

After install, launch Fizgig and work left-to-right through the numbered tabs:

1. **Start** — set your training image folder. If model paths aren't configured, a prompt guides you to Preferences.
2. **Image Prep** (optional) — resize, PNG-convert, or face-crop your training images.
3. **Captions** — write trigger-word captions or generate with Florence-2 AI. Optionally translate to bilingual English+Chinese.
4. **Samples** — configure the preview prompts that render during training.
5. **Training** — pick a preset, tune settings, click **Start Training**.

The unnumbered tabs are post-training tools (also work on any Klein LoRA you've downloaded):

- **Profiler** — analyse which blocks are active and how strong their contributions are.
- **Repair Studio** — live per-block editing with Turbo Preview and optional donor blending.
- **LoRA the Explorer** — evolutionary discovery via human-guided selection.
- **Extract** — distil to a lower rank with block and timestep targeting.
- **Preferences** — model paths, output directories, inference block-swap preset.

---

## VRAM guidance

Inference tools (Profiler / Repair Studio / Explorer / Extract) on Distilled 4-step:

| Block Swap | Min VRAM | Notes |
|---|---|---|
| 0 | 24 GB+ | No swap — fastest |
| 4 | 20 GB | Light swap |
| 8 | 16 GB | Moderate swap |
| 12 | 14 GB | Aggressive swap |
| 16 | 12 GB | Maximum swap — slower but fits |

**DiT Block Swap (inference)** in Preferences applies to all tool tabs. Training has its own separate block swap setting. On first launch, Fizgig auto-detects your GPU VRAM and picks a sensible default — once you explicitly choose a value, your choice is saved.

---

## Support the project

If Fizgig saves you time or helps you make better LoRAs, consider supporting development:

<a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>

---
