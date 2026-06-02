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

## Why Fizgig for Klein 9B?

Fizgig is a dedicated **training, LoRA-surgery, and gamified-exploration** studio built end-to-end for one model — **Flux 2 Klein 9B** — so everything is tuned for it instead of bolted on. Training is fast and light: on the fp8 Base DiT it runs the frozen-base matmuls in fp8 on the tensor cores for about **1.5× faster steps** (RTX 40/50-series), and the fp8 model stays resident at ~9.6 GB so a full 9B LoRA trains comfortably on a **16 GB card** — or, with the opt-in **4-bit (NF4) base** mode, in ~7.5 GB, putting Klein 9B within reach of **10–12 GB cards** that can't fit fp8. It also includes things most trainers skip — **Context LoRA** training (learn a new LoRA on top of an existing frozen one so they coexist: a face that sits on a style, an outfit that drapes over a character), **bilingual captions** for richer convergence, **distilled 4-step previews** that match ComfyUI, a self-tuning **adaptive learning rate**, and **pause/resume** that frees your GPU mid-run and picks up exactly where it left off, full optimizer state and no quality regression — so you can fire up Rocket League without sacrificing your training run.

On the data side, the **Image Prep** tab can auto-cut tight face crops from your wider shots and add them as extra datapoints, with optional **gender targeting** (largest male/female face) so it locks onto your subject even in group shots — pairing a close-up with a full shot adds a lot to a character dataset (works best from high-res originals). Training defaults to ~512² (0.25 MP) and resizes in-cache, so a dataset of any resolution or aspect ratio just works — nothing has to be square or pre-sized.

But the real reason is what happens *after* training. Fizgig is a workbench, not just a trainer: **fix** a broken LoRA block-by-block in the Repair Studio with no retraining, **explore** new variations like a game in LoRA the Explorer, **profile** exactly which blocks carry style vs identity vs detail, and **extract** a LoRA down to a smaller rank or a specific block range — all in one app, each tool reading the others' output.

Fizgig is **free and open source** — and a good first run is the **✨ Old Reliable** preset on the Training tab; after that, try the lighter **✨ Old Reliable - Flavour 8** (rank 8). A lot of the old rank-16 instinct dates from models with far fewer parameters than Klein 9B — rank 8 is often plenty.

### What makes Fizgig different?

- **Fix broken LoRAs without retraining** — overbaked identity? crushed style? Adjust per-block sliders with live side-by-side preview and save a repaired `.safetensors` in seconds.
- **Explore variations like a game** — the computer proposes random mutations, you pick favourites, and the LoRA evolves through selection. **Freeze** locks tweaked blocks when you're ready, so future mutations only touch what's left.
- **Discover → Refine → Discover** — the Explorer and Repair Studio are bidirectionally connected. Find something interesting in the Explorer? One click sends it to the Repair Studio with all 32 sliders pre-set. Fine-tuning in the Repair Studio? One click sends your state back to the Explorer for more evolutionary discovery.
- **See exactly what a LoRA does** — the Profiler shows which transformer blocks carry style, identity, and detail signal, so you know what to fix before you touch a slider.
- **Train with intelligence** — adaptive learning rate adjusts itself based on loss, gradient clipping, and weight-norm growth.
- **Just works on your GPU** — block swap auto-detects from your VRAM at both training and inference time. 16 GB, 24 GB, 32 GB — Fizgig picks the right setting. If you do run out of memory, it tells you exactly what to change.
- **Make LoRAs work together** — Context LoRA training loads an existing LoRA as a frozen layer, so the new one learns to coexist. Train a face on top of a style and they stop fighting at inference. Train an outfit on top of a character and the clothes drape correctly. Fix compatibility between two LoRAs that conflict.

---

## Features

### Repair Studio
32 live sliders — one per transformer block — with side-by-side Distilled preview. See the effect of every change instantly. Optional donor-LoRA blending via rank concatenation lets you mix blocks from two LoRAs. Quick-set `[0]` `[1]` `[±]` `[⚖]` buttons on every slider — **Balance** keeps the total primary+donor contribution at 1.0 per block, perfect for cross-fading between two LoRAs. **Turbo Preview** caches activations and prompt encodings — up to 97% faster on late-block changes. Click the tweaked preview to pop it out into a resizable window. Browse a new LoRA and it auto-swaps — no manual reset needed. Saves a baked `.safetensors` that works in ComfyUI at strength 1.0. Jump to the Explorer for evolutionary discovery, or receive a baseline from the Explorer for precision editing — the two tools are seamlessly connected.

### LoRA the Explorer
Evolutionary LoRA discovery. The computer randomly mutates blocks and shows you 4 variants — pick your favourite and it becomes the new baseline. **Freeze Tweaked Blocks** locks your changes when you're ready — future mutations only touch what's still unlocked. A **Structure** slider controls how much the composition/style anchor changes each round. Seed cycling verifies variants across different seeds. When you find a direction you love, **"Refine this baseline in Repair Studio"** sends your current slider state directly to the Repair Studio with all 32 sliders pre-set — seamless handoff from discovery to precision editing.

### Profiler
Per-block activation profile with a colour-coded 5-bucket HTML report. Identifies which blocks carry style, identity, and detail signal — and where they overlap. Writes a JSON sidecar that the Repair Studio reads automatically, showing you the profiler's findings inline when you load the same LoRA.

### Training
- **Proven presets** for rank 4–16, single subject through multi-character — or build your own.
- **Distilled training samples** — 4-step Distilled previews that match ComfyUI output exactly. Uses a separate Distilled DiT loaded alongside the training Base model, with the ComfyUI Euler Simple schedule. **On by default** — toggle via the checkbox on the Samples tab. Falls back to Base multi-step samples (~40 steps) when off or when VRAM is tight.
- **Adaptive LR** — bi-directional plateau tracker that probes up on steady loss descent and pulls down (with optional weight rollback) on plateau, heavy gradient clipping, or weight-norm runaway.
- **Context LoRA** — load an existing LoRA as a frozen *active* layer during training, so the new LoRA learns to coexist at inference. No other trainer does this.

> **⚠️ Context LoRA note:** Training sample previews in context mode often don't reflect the final quality of the trained LoRA. The samples can look distorted even when the LoRA itself is excellent. Always evaluate the output LoRA in ComfyUI for accurate results. This is a known issue being worked on.

> **⚠️ Training samples note:** Training samples in general often look inferior to the actual LoRA when deployed in ComfyUI — less detail, weaker likeness, or slightly off colours. The LoRA is usually better than the samples suggest. Always evaluate checkpoints in ComfyUI rather than judging quality from training previews alone. This is an active area of investigation — if you have insights into the discrepancy, contributions and ideas are welcome.

- **Pause / Resume** — graceful epoch-boundary pause that frees VRAM and resumes with full optimizer state and no quality regression.
- **Model Area targeting** — train only Identity blocks, Style blocks, Details blocks, or the full model.
- **Faster fp8 training (free speedup)** — when you train on the fp8 Base DiT, Fizgig runs the frozen-base matmuls in fp8 on the tensor cores (`torch._scaled_mm`, forward *and* backward) — **~1.5× faster training steps** at typical LoRA resolutions, with no quality cost in our testing. It's automatic — no flag, no config — and works alongside block swap and context LoRA. Needs an RTX 40/50-series (or newer) GPU; older cards fall back to the standard path automatically. Yet another reason to train on fp8 Base.
- **Auto VRAM management** — block swap auto-detects from GPU VRAM, OOM detection suggests fixes. Supports both bf16 and fp8 Base DiT. Training with fp8 Base and block swap works correctly.
- **Diffusers LoRA support** — OneTrainer LoRAs with split Q/K/V keys auto-fused on load.

### Dataset Prep
- **Florence-2 AI captioning** — bulk-generate detailed captions with one click.
- **Bilingual translation** — optionally append Chinese translations via Helsinki-NLP. Klein's Qwen3 text encoder has deep Chinese training; bilingual captions act as text-level data augmentation, improving visual quality without changing loss.
- **Image Prep** — batch resize, PNG conversion, and face-crop derivatives via InsightFace.

### Extract
Distil any Klein LoRA down to a lower rank with block and timestep targeting. Fast presets run pure weight SVD with no GPU models loaded; activation-weighted presets use forward passes for better accuracy. Supports PEFT and LyCORIS (LoKR / LoHa) sources.

### Compatibility
- **Formats** — loads kohya, PEFT, OneTrainer (OMI + legacy), AI-Toolkit, and LyCORIS (LoKR / LoHa) LoRAs. All formats auto-converted on load. LyCORIS files work for preview, profiling, and extraction; bake converts them to standard LoRA via SVD.
- **Output** — kohya-style `.safetensors` that drop straight into ComfyUI Klein nodes.
- **LyCORIS bake** — LoKR and LoHa LoRAs can be saved/baked via GPU-accelerated SVD materialization.
- **YouTube help** — every tab links to the relevant section of the walkthrough video.

---

## Requirements

- **GPU** — NVIDIA RTX 30-series, 40-series, or 50-series (Blackwell). **16 GB+ VRAM** recommended (24 GB+ comfortable). The fp8 training **speedup** needs a 40-series or newer (fp8 tensor cores); 30-series still gets the fp8 VRAM savings, just not the extra speed.
- **NVIDIA driver** — 555+ on Windows, 550+ on Linux (required for CUDA 12.8 PyTorch wheels).
- **OS** — Windows 10 / 11, or Linux. macOS works for captioning and image prep but training requires CUDA.
- **Python** — 3.10, 3.11, 3.12, or 3.13.
- **Disk** — ~10 GB for the venv, plus ~40 GB for the model files (see below).
- **Visual Studio Build Tools** (Windows only) — required for compiling InsightFace. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select **"Desktop development with C++"** workload. If you see errors about `cl.exe` or missing C++ compiler during install, this is what's needed.

---

## Install

First, get the code — clone the repo (or download the ZIP via the green **Code** button and extract it):

```bash
git clone https://github.com/shootthesound/Fizgig.git
cd Fizgig
```

### Windows (one-click)

Double-click `install_fizgig.bat`. The installer creates a venv, installs CUDA 12.8 PyTorch + all dependencies, pre-downloads InsightFace face-detection models, and verifies that CUDA is visible to PyTorch.

Launch with `run_fizgig.bat` when install completes. To update later, double-click `update_fizgig.bat`.

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
| **Base DiT (fp8) — recommended** | `flux-2-klein-base-9b-fp8.safetensors` | ~9.5 GB fp8 | [black-forest-labs/FLUX.2-klein-base-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8) |
| Base DiT (bf16) | `flux-2-klein-base-9b.safetensors` | ~17 GB bf16 | [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) |
| Distilled DiT | `flux-2-klein-9b-fp8.safetensors` | ~9 GB fp8 | [black-forest-labs/FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) |
| VAE / AE | `ae.safetensors` | ~320 MB | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) (from root, **not** the `vae/` subfolder) |
| Text Encoder | `qwen_3_8b.safetensors` | ~15 GB | [Comfy-Org/vae-text-encorder-for-flux-klein-9b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors) |

Training runs on the **Base DiT** — the **fp8 version is recommended on every GPU**: same training quality at ~half the VRAM (it stays resident at ~9.6 GB, so a 9B LoRA trains in ~14 GB and fits 16 GB cards). The speed depends on your GPU:

- **RTX 40 / 50-series** — you *also* get **~1.5× faster training steps**: the frozen-base matmuls run in fp8 directly on the tensor cores (`torch._scaled_mm`, automatic, no config).
- **RTX 30-series and older** — the speedup is skipped automatically (these GPUs lack fp8 tensor cores), but you still get the **full VRAM savings and the same quality**, so fp8 Base is worth it here too.

Either way it's automatic — Fizgig auto-detects pre-quantised files and the right path for your GPU, so you don't need to touch the "FP8 Base" checkbox; the bf16 version works too if you prefer it. The **Distilled DiT** is used for fast 4-step previews (on by default during training, and always in the Profiler, Repair Studio, and Explorer) — so you want both if you'll use the workbench features.

**Smaller cards — 4-bit (NF4) base.** fp8 training needs ~14 GB, so it wants a 16 GB card. For **10–12 GB cards** there's an opt-in **4-bit (NF4) base** mode (the *4-bit Base* toggle in Memory & FP8 / FP4): it quantizes the frozen base to 4-bit, halving DiT VRAM to **~5.6 GB** so a full 9B LoRA trains in **~7.5 GB** — the LoRA still trains in bf16 on top, QLoRA-style. It loads layer-by-layer so the card never has to hold the whole model. It's a lower-precision base than fp8, so it's a **slight quality trade** — always check the output LoRA in ComfyUI — and **16 GB+ cards should stick with fp8** (same quality, plus the speedup).

Three smaller models auto-download on first use: InsightFace (`buffalo_l`, ~300 MB, during install), Florence-2 (~500 MB–1.5 GB, first AI caption), Helsinki-NLP/opus-mt-en-zh (~300 MB, first bilingual translation).

---

## Getting started

After install, launch Fizgig and work left-to-right through the numbered tabs:

1. **Start** — set your training image folder. If model paths aren't configured, a prompt guides you to Preferences.
2. **Image Prep** (optional) — resize, PNG-convert, or face-crop your training images.
3. **Captions** — write trigger-word captions or generate with Florence-2 AI. Optionally translate to bilingual English+Chinese.
4. **Samples** — configure the preview prompts that render during training. Distilled 4-step previews (faster, ComfyUI-accurate) are on by default; toggle on the Samples tab.
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

**Training:** the fp8 Base DiT stays resident at ~9.6 GB (not dequantised to bf16), so training a 9B LoRA fits comfortably in **16 GB** — around **14 GB** observed at block-swap 0 with a context LoRA active (a little less without). VRAM scales with resolution and batch size; raise block swap to fit smaller cards.

**DiT Block Swap (inference)** in Preferences applies to all tool tabs. Training has its own separate block swap setting. On first launch, Fizgig auto-detects your GPU VRAM and picks a sensible default — once you explicitly choose a value, your choice is saved.

---

## Support the project

If Fizgig saves you time or helps you make better LoRAs, consider supporting development:

<a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>

---
