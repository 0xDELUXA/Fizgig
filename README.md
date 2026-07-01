<h1 align="center">Fizgig — Klein 9B & Krea 2 LoRA Studio</h1>

<p align="center">
  <strong>Fix broken LoRAs without retraining. Remix any LoRA into new variations in seconds.</strong><br>
  A train · repair · explore workbench built end-to-end for <strong>Flux 2 Klein 9B</strong> and <strong>Krea 2</strong>.
</p>

<p align="center">
  <a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
</p>

<p align="center">
  <a href="https://youtu.be/sH-kGR8yzBU"><img src="https://img.youtube.com/vi/sH-kGR8yzBU/maxresdefault.jpg" alt="Watch the walkthrough" width="600"></a><br>
  <em>Watch the full walkthrough on YouTube</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/models-Klein%209B%20%2B%20Krea%202-blue?style=for-the-badge" alt="Klein 9B + Krea 2">
</p>

> **🎉 New — Krea 2.** Fizgig now supports a **second, fully native model family**: **Krea 2 (12.9B)**. The whole workbench works with it — Repair Studio, Explorer, Royale, Profiler, Extract — plus Context LoRA, Adaptive LR, Pause/Resume, 4-bit (NF4) low-VRAM training, and the live sample override. [Details below ↓](#krea-2--second-model-family)

---

## What Fizgig is

Every trainer makes LoRAs. Fizgig is built around what you do with them **afterwards** — and that's the part nobody else has.

- **Fix** a baked LoRA block-by-block, no retraining — overbaked identity, crushed style, drag a slider, save a new `.safetensors`.
- **Explore** new variations like a game — the app proposes mutations, you pick favourites, the LoRA evolves through selection.
- **Find** the best LoRA by eye — **LoRA Royale** renders every epoch of a run (or any folder of LoRAs) on one seed; crossfade to the one that *feels* right.
- **Share** what you made — LoRA Royale exports the epoch morph, or travels a single LoRA through seeds, prompts, or strength, as a looping MP4/GIF made to share.
- **Profile** exactly which blocks carry identity, style, and detail — so you know what to touch before you touch it.

Under that workbench sits a fast, light trainer tuned for its models. Because everything is built natively for Klein 9B and Krea 2 instead of bolted on to a dozen models, the whole thing can do things the generalists can't: a full 9B LoRA trains comfortably on a **16 GB card**, and the post-training tools all read each other's output.

**Free and open source.** A good first run is the **✨ Old Reliable** preset on the Training tab — then try **✨ Old Reliable · Flavour 8** (rank 8). Much of the old rank-16 instinct predates models this size; on Klein 9B, rank 8 is often plenty.

---

## The workbench

The reason to use Fizgig. Each tool works on a trained run's output **or any Klein LoRA you've downloaded** — and they hand off to each other.

### Repair Studio
Thirty-two live sliders — one per transformer block — with a side-by-side Distilled preview that updates instantly. **Turbo Preview** — activation caching for a live LoRA-tweaking UI, which no other LoRA tool does — caches per-block outputs and prompt encodings across the denoising steps, so late-block edits redraw up to **97% faster**; the baked save is always exact, Turbo or not. Quick-set buttons on every slider (`[0]` `[1]` `[±]` `[⚖]`); **Balance** holds the combined primary + donor weight at 1.0 per block, ideal for cross-fading two LoRAs. Optional donor-LoRA blending mixes blocks from a second LoRA via rank concatenation. Previews can be conditioned on a **reference image** (Klein is an edit model), so you see how your LoRA edits a real photo. Click a preview to pop it into a resizable window. Browse a new LoRA and it auto-swaps — no manual reset. Saves a baked `.safetensors` that works in ComfyUI at strength 1.0.

### LoRA the Explorer
Evolutionary discovery. The app mutates blocks and shows four variants — pick a favourite and it becomes the new baseline. **Freeze Tweaked Blocks** locks what you like so future mutations only touch the rest. A **Structure** slider sets how far the composition anchor drifts each round; seed cycling checks variants across seeds. Found a direction you love? **Refine this baseline in Repair Studio** sends all 32 slider values straight over — and Repair Studio sends state back the same way. Discover → refine → discover, in a loop.

### LoRA Royale
Find the best LoRA the human way — then turn the winner into share-ready clips. Point it at a training output folder and it renders **every epoch on one fixed seed** (Distilled 4-step), with a **crossfade slider** that blends smoothly between consecutive epochs — drag until it looks best and stop. A thumbnail grid sits below; click any epoch to jump there. Drop in a **reference image** (Klein is an edit model) and every epoch edits the same photo. An optional **likeness score** (InsightFace ArcFace, CPU — no extra VRAM) rates each epoch against a training shot — **close-up headshots included** (it pads-and-retries so a face that fills the frame still detects) — and flags the closest in gold with one-click **Jump to best**. **Export likeness clip** turns it into a share-ready, side-by-side *subject vs each epoch* comparison with the score burned in, morphing epoch by epoch. **Promote** copies the winner to a clean `.safetensors`. Not a training run? Point it at **any folder of LoRAs** and it compares them by name — or flip to **Single-LoRA mode** to run everything below on one downloaded LoRA, no folder required.

Because the morph *is* the magic, the payoff is four **travel** tools that each render a sequence you **scrub to review and only save if you like it** — as a looping MP4 or GIF, re-saveable in either format without re-rendering, with an optional **deflicker** pass (the timelapse trick DaVinci uses) for flicker-free clips. **Export the morph** saves the whole epoch sweep, a face resolving epoch by epoch — or **Save all stills** dumps every rendered epoch to a folder as full-res PNGs (the renders otherwise live only in memory). **Seed travel** slerps through a journey of seeds to show the LoRA's range. **Prompt travel** interpolates the text embedding through waypoints — Time of day, Season, Age, Era, or your own words — so one subject flows through the change; pick a **Preset + Subject** and it writes the prompt for you. And **LoRA strength travel** ramps the LoRA from 0 (base model) to full and beyond, so you literally *watch the effect fade in*. Every travel can be anchored to a reference to hold the subject steady, with interpolation and seed-drift knobs for a smooth, brightness-even result. (The epoch morph shows the LoRA *learning*; the travels show what it can *do*.)

### Profiler
A per-block activation profile with a colour-coded, five-bucket HTML report — which blocks carry style, identity, and detail signal, and where they overlap. Writes a JSON sidecar that Repair Studio reads automatically, showing the findings inline when you load the same LoRA.

### Extract
Distil any Klein LoRA to a lower rank with block and timestep targeting. Fast presets run pure weight SVD with no GPU models loaded; activation-weighted presets use forward passes for better accuracy. Supports PEFT and LyCORIS (LoKR / LoHa) sources.

---

## Krea 2 — second model family

Krea 2 is a from-scratch **native** port — no external tooling at runtime: a 12.9B single-stream MMDiT, the Qwen-Image VAE, and a Qwen3-VL-4B text encoder. **Train on the RAW model** (fp8, ~14 GB resident) and **preview on the fp8 Turbo** (8-step, CFG-free) with your live LoRA applied. Pick it from the **Base Model selector** at the top of the Training tab.

Everything works on Krea 2: **all five workbench tools** (Profiler, Extract, Repair Studio, Explorer, Royale), plus **Pause/Resume** (full state), **Context LoRA**, **Adaptive LR**, **reference images** (through the text encoder's vision path — "prompt from a picture"), and the live sample override. A few Training-tab controls are hidden for Krea 2 for now (not removed): per-block Model-Area targeting (no Krea 2 block map yet), the Optimizer and Timestep sections, and the FP8-Scaled / FP8-TE / Gradient-Checkpointing toggles.

> **📣 Help map Krea 2's blocks — [open an issue](https://github.com/shootthesound/Fizgig/issues).** The colour-coded sliders, Model-Area targeting, and block-aware presets are Klein-only right now because Krea 2's per-block roles (which blocks carry **identity**, **style**, and **detail**) aren't mapped yet. The **Profiler**'s weight-only report is the instrument for discovering them. If you find patterns — a block that clearly drives identity, a range that governs style — please share your findings in the **[GitHub Issues](https://github.com/shootthesound/Fizgig/issues)**. Community block-discovery is what will drive the colour-coding and finer layer targeting coming to the Krea 2 presets and Repair Studio.

**Runs on smaller cards, and adapts to yours.** Krea 2 is a bigger model than Klein, but the low-VRAM paths are wired:

- **4-bit (NF4) base** — the *4-bit Base* toggle (Memory & FP8) trains the frozen base at ~5.6 GB (base + LoRA ~8.3 GB), so a full Krea 2 LoRA fits a **10–12 GB card with no block swap** — QLoRA-style, the LoRA still trains in bf16 on top.
- **VRAM-adaptive previews & workbench** — the fp8 Turbo (used for in-training previews *and* Repair Studio / Explorer / Royale) auto-sizes its own block swap to your GPU, so the tools fit smaller cards without you tuning anything.
- **Previews never crash a run** — if the Turbo preview can't fit even swapped, previews auto-disable and **training keeps going and saving**; evaluate the LoRA in ComfyUI. (With 4-bit, the base even parks off the GPU during each preview, so the two coexist.)

Krea 2 trains real, ComfyUI-compatible LoRAs, and its training recipe is verified against the reference implementation — same noised/target flow-matching, `krea2_shift` timestep sampling, and gradient clipping.

---

## Training

The foundation: fast, light, and tuned for one model.

- **Proven presets** for rank 4–16, single subject through multi-character — or roll your own.
- **Context LoRA** — load an existing LoRA as a frozen *active* layer so the new one learns to coexist. Train a face on top of a style and they stop fighting at inference; train an outfit on top of a character and the clothes drape correctly. No other trainer does this.
- **Distilled training samples** — 4-step previews that match ComfyUI output closely (a separate Distilled DiT, ComfyUI Euler Simple schedule). On by default; toggle on the Samples tab. On tight cards the sample model auto-swaps its own blocks by VRAM so 4-step previews keep working on 16 GB. On 24 GB+ it stays resident and is cached in system RAM between epochs (RAM-checked, saves ~3–4 s/epoch).
- **Reference-conditioned samples** — Klein is an edit model, so previews can *edit* a reference photo instead of generating from scratch. Auto-resized to ~0.20 MP so it can't OOM; works on Base and Distilled samples.
- **Adaptive LR** — a bi-directional plateau tracker that probes up on steady loss descent and pulls down (with optional weight rollback) on plateau, heavy gradient clipping, or weight-norm runaway.
- **fp8 Base training** — the fp8 Base DiT stays resident at ~9.6 GB instead of dequantising to ~18 GB, so a full 9B LoRA trains in ~14 GB and fits a 16 GB card — lossless, no quality cost. Automatic (Fizgig detects the pre-quantised file), no flag.
- **Gradient checkpointing toggle** — on by default (it's what fits a 9B LoRA on 16 GB). Turn it **off** on a 24 GB+ card for meaningfully faster steps. A VRAM-aware warning fires if you switch it off on a card that can't spare the activation memory.
- **Pause / Resume** — graceful epoch-boundary pause that frees your GPU mid-run and resumes with full optimizer state and no quality regression. Fire up Rocket League, come back, carry on.
- **Model Area targeting** — train only Identity, Style, or Detail blocks, or the full model.
- **Auto VRAM management** — block swap auto-detects from GPU VRAM; OOM detection tells you exactly what to change. Supports bf16 and fp8 Base DiT, with block swap.
- **Diffusers LoRA support** — OneTrainer LoRAs with split Q/K/V keys are auto-fused on load.

> **A note on Base previews:** the default Distilled 4-step previews track ComfyUI closely, including with a Context LoRA active. Only **Base multi-step** previews (Distilled toggled off) can look softer than the deployed LoRA — they come from a mid-training fp8 checkpoint, so colours and detail can be slightly off even when the LoRA is excellent. Judging from Base previews? Confirm final quality in ComfyUI.

### Live status bar
A bottom bar with stacked **VRAM and system-RAM gauges** (smooth gradient fills, plus a per-run peak marker so you can see how high a run pushed memory). VRAM is read at the device level, so it catches other apps holding the GPU too. A top-right **IDLE / BUSY** light shows at a glance whether the app is working. Hide or show the whole bar with one click; it remembers.

Beside it sits a **live sample override** — tick it to set a prompt, seed, width/height, and optional reference image for the *next* samples, mid-run, no restart. The text encoder only re-runs when the prompt text changes, so seed / resolution / reference tweaks are instant.

### Dataset prep
- **Florence-2 AI captioning** — bulk-generate detailed captions in one click.
- **Bilingual captions** — optionally append Chinese via Helsinki-NLP. Klein's Qwen3 text encoder has deep Chinese training, so bilingual captions act as text-level data augmentation, improving visual quality without changing loss. In a controlled A/B (same data, seed, and hyperparameters — captions the only change) the loss curves stayed within ±0.001/epoch, yet the bilingual run produced visibly more skin detail and faster visual convergence.
- **Image Prep** — batch resize, PNG conversion, and InsightFace face-crop derivatives, with optional **gender targeting** (largest male/female face) so it locks onto your subject in group shots. Pairing a tight crop with a full shot adds a lot to a character dataset. Training defaults to ~512² (0.25 MP) and resizes in-cache, so any resolution or aspect ratio just works — nothing has to be square or pre-sized.

### Compatibility
Loads kohya, PEFT, OneTrainer (OMI + legacy), AI-Toolkit, and LyCORIS (LoKR / LoHa) — all auto-converted on load. LoKR and LoHa run **natively at inference** — no pre-conversion — anywhere in the app: as a primary or donor in Repair Studio, in the Profiler, in Extract, even as a Context LoRA. **Bake** materialises them to a standard LoRA via GPU-accelerated SVD. Output is kohya-style `.safetensors` that drop straight into ComfyUI Klein nodes. Every tab links to the relevant section of the walkthrough video.

---

## Requirements

- **GPU** — NVIDIA RTX 30 / 40 / 50-series. **16 GB+ VRAM** recommended (24 GB+ comfortable). The fp8 Base's VRAM savings apply on every supported card.
- **NVIDIA driver** — 555+ on Windows, 550+ on Linux (for the CUDA 12.8 PyTorch wheels).
- **OS** — Windows 10 / 11 or Linux. macOS handles captioning and image prep, but training needs CUDA.
- **Python** — 3.10, 3.11, 3.12, or 3.13.
- **Disk** — ~10 GB for the venv, plus ~40 GB for model files.
- **Visual Studio Build Tools** (Windows only) — needed to compile InsightFace. Install [Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the **"Desktop development with C++"** workload. Errors about `cl.exe` or a missing C++ compiler mean this is what's needed.

---

## Install

Clone the repo (or download the ZIP via the green **Code** button and extract):

```bash
git clone https://github.com/shootthesound/Fizgig.git
cd Fizgig
```

**Windows (one-click)** — double-click `install_fizgig.bat`. It creates a venv, installs CUDA 12.8 PyTorch and all dependencies, pre-downloads the InsightFace models, and verifies CUDA is visible to PyTorch. Launch with `run_fizgig.bat`; update later with `update_fizgig.bat`.

**Linux / macOS:**

```bash
python install_fizgig.py
chmod +x run_fizgig.sh
./run_fizgig.sh
```

Three small models auto-download on first use: InsightFace `buffalo_l` (~300 MB, during install), Florence-2 (~500 MB–1.5 GB, first AI caption), and Helsinki-NLP `opus-mt-en-zh` (~300 MB, first bilingual translation).

---

## Model downloads (you provide)

Fizgig doesn't bundle weights — they're large and licensing varies. Each row in the **Preferences** tab has a **Download** link to the right HuggingFace page. You only need the family you're using.

### Klein 9B

| Model | File | Size | Source |
|---|---|---|---|
| **Base DiT (fp8) — recommended** | `flux-2-klein-base-9b-fp8.safetensors` | ~9.5 GB fp8 | [black-forest-labs/FLUX.2-klein-base-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8) |
| Base DiT (bf16) | `flux-2-klein-base-9b.safetensors` | ~17 GB bf16 | [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) |
| Distilled DiT | `flux-2-klein-9b-fp8.safetensors` | ~9 GB fp8 | [black-forest-labs/FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) |
| VAE / AE | `ae.safetensors` | ~320 MB | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) (from root, **not** the `vae/` subfolder) |
| Text Encoder | `qwen_3_8b.safetensors` | ~15 GB | [Comfy-Org/vae-text-encorder-for-flux-klein-9b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors) |

Training runs on the **Base DiT**, and the **fp8 version is recommended on every GPU**: same training quality at roughly half the VRAM (resident at ~9.6 GB, so a 9B LoRA trains in ~14 GB and fits a 16 GB card).

The VRAM savings and quality are the same across all supported cards (RTX 30 / 40 / 50-series) — fp8 Base is worth it on every GPU.

It's all automatic — Fizgig detects pre-quantised files and the right path for your GPU, so you never need to touch the "FP8 Base" checkbox (the bf16 version works too if you prefer). The **Distilled DiT** powers the fast 4-step previews — on by default during training, and always used in the Profiler, Repair Studio, and Explorer — so grab both if you'll use the workbench.

### Krea 2

All four files live in the one [**Comfy-Org/Krea-2**](https://huggingface.co/Comfy-Org/Krea-2) repo.

| Model | File | Size | Source |
|---|---|---|---|
| **RAW DiT (bf16) — training** | `krea2_raw_bf16.safetensors` | ~26 GB bf16 | [Comfy-Org/Krea-2 → diffusion_models](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_bf16.safetensors) |
| **Turbo DiT (fp8) — previews / workbench** | `krea2_turbo_fp8_scaled.safetensors` | ~13 GB fp8 | [Comfy-Org/Krea-2 → diffusion_models](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors) |
| Qwen-Image VAE | `qwen_image_vae.safetensors` | ~250 MB | [Comfy-Org/Krea-2 → vae](https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors) |
| Text Encoder (bf16) | `qwen3vl_4b_bf16.safetensors` | ~8 GB bf16 | [Comfy-Org/Krea-2 → text_encoders](https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_bf16.safetensors) |

Training runs on the **RAW DiT**; previews and the whole workbench run on the **fp8 Turbo** — grab both if you'll train *and* use the tools. The text encoder must be the **bf16** Qwen3-VL-4B (the fp8 ComfyUI variant can't run the vision path used for reference images, or training). On smaller cards, the **4-bit (NF4)** toggle shrinks the RAW base to ~5.6 GB so it fits 10–12 GB GPUs.

---

## VRAM guidance

### Klein 9B

**Inference tools** (Profiler / Repair Studio / Explorer / Extract) on Distilled 4-step:

| Block Swap | Min VRAM | Notes |
|---|---|---|
| 0 | 24 GB+ | No swap — fastest |
| 4 | 20 GB | Light swap |
| 8 | 16 GB | Moderate swap |
| 12 | 14 GB | Aggressive swap |
| 16 | 12 GB | Maximum swap — slower, but fits |

**Training** — the fp8 Base DiT stays resident at ~9.6 GB (not dequantised to bf16), so a 9B LoRA fits comfortably in **16 GB** — around 14 GB observed at block-swap 0 with a Context LoRA active, a little less without. VRAM scales with resolution and batch size; raise block swap to fit smaller cards.

**Smaller cards — 4-bit (NF4) base.** fp8 training needs ~14 GB: it fits a 16 GB card with no swap, but a **10–12 GB card has to block-swap**, paying a PCIe-transfer penalty every step. The opt-in **4-bit (NF4) base** mode (the *4-bit Base* toggle in Memory & FP8 / FP4) quantizes the frozen base to 4-bit — halving DiT VRAM to ~5.6 GB so a full 9B LoRA trains in **~7.5 GB**, which fits 10–12 GB cards with **no swap at all** (and so beats fp8-with-swap on those cards). The LoRA still trains in bf16 on top, QLoRA-style, and the base loads layer-by-layer so the card never holds the whole model. It's a lower-precision base, so it's a slight quality trade — always check the output in ComfyUI — and **16 GB+ cards should stick with fp8** (same quality, no swap).

**DiT Block Swap (inference)** in Preferences applies only to the workbench tools. Training has its own separate block-swap setting, and its Distilled samples auto-swap by VRAM — so this preference never touches a training run. On first launch Fizgig auto-detects your VRAM and picks a sensible default; once you choose a value, your choice sticks.

### Krea 2

Krea 2 is a bigger model, so the numbers differ — but Fizgig **auto-sizes block swap to your card** for both training and the workbench, so there's nothing to tune:

- **Training** runs on the RAW fp8 base (~14 GB resident); block swap auto-detects from VRAM (32 GB → none, scaling up to maximum on sub-16 GB cards). The **4-bit (NF4)** toggle drops the base to ~5.6 GB (base + LoRA ~8.3 GB), fitting a **10–12 GB card with no swap**.
- **Previews & workbench** (Repair Studio / Explorer / Royale / in-training previews) run on the fp8 Turbo, which peaks ~22.6 GB unswapped — heavier than Klein's Distilled, so Fizgig auto-swaps it to fit your GPU (≈17 GB at swap 12; 16 GB cards swap enough to fit). If a preview still can't fit, it **auto-disables and training keeps running and saving** — evaluate the LoRA in ComfyUI. With 4-bit, the base even parks off the GPU during each preview so the two coexist.

---

## Getting started

Launch Fizgig and work left-to-right through the numbered tabs:

1. **Start** — set your training image folder. If model paths aren't configured, a prompt points you to Preferences.
2. **Image Prep** (optional) — resize, PNG-convert, or face-crop your images.
3. **Captions** — write trigger-word captions or generate with Florence-2; optionally translate to bilingual English + Chinese.
4. **Samples** — configure the preview prompts that render during training (Distilled 4-step on by default).
5. **Training** — pick a preset, tune, click **Start Training**.

The unnumbered tabs are the post-training workbench — and work on any Klein LoRA you've downloaded: **Profiler**, **Repair Studio**, **LoRA the Explorer**, **LoRA Royale**, **Extract**, and **Preferences** (model paths, output directories, inference block-swap preset, default Browse folders).

---

## Support the project

If Fizgig saves you time or helps you make better LoRAs, consider supporting development:

<a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>

---

## License

Fizgig is open source under the **[Apache License 2.0](LICENSE)** — free to use, modify, and redistribute, including commercially, with attribution and no warranty. It includes third-party components under compatible permissive licenses (musubi-tuner — Apache-2.0; ai-toolkit — MIT; Diffusers / FLUX — Apache-2.0); see **[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)**.

Copyright © 2026 Peter Neill.

Model weights are **not** covered by this license — each model carries its own terms from its publisher (see the Download links in Preferences).
