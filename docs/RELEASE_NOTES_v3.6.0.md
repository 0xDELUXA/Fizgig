# Fizgig v3.6.0 — Two subjects in one MiniMax LoRA, and previews you can trust

> **DRAFT — not tagged.** Multi Concept and identity-first are still being validated on real runs.

Two big things for MiniMax H3. Samples now render the way ComfyUI does, so what you see during a
run is what you'll get from the checkpoint. And a single LoRA can hold **two subjects**, each
with its own folder and its own trigger word.

## Previews now match ComfyUI

Five things in the preview path were doing something slightly different from the reference
implementation. Each was small on its own; together they were the difference between a sample
that shows your LoRA and one that only hints at it.

| | |
|---|---|
| **Video decode** | Runs at the precision the decoder's own weights ship in. Fixes soft detail and banding in smooth gradients. |
| **Audio track** | H3 generates sound alongside picture, and the picture is conditioned on it every step. The audio is now stepped exactly as ComfyUI steps it. |
| **Base weights** | The quantised base is unpacked without the small per-channel gain error it used to pick up. |
| **Modulation** | The per-block gain that scales attention and MLP now runs at full precision, as the reference does. |
| **Prompt length** | Prompts are no longer cut at 512 tokens. A long prompt used to lose its tail in silence — and because the prompt's length sets the video's temporal origin, it also rendered on a different grid than the same prompt in ComfyUI. |

**Two of those improve training, not just previews.** Your LoRA is now fitted against a more
accurate base, so runs from this version won't line up exactly with older ones. If you're
mid-comparison, finish it on one version or the other.

## Multi Concept — two subjects, one LoRA

Tick **Multi Concept** on the Training tab and a second folder picker appears. Each subject gets
its own folder, its own trigger word, and its own dataset entry.

That last part is what makes it work. In identity-learn mode every image is marked against
*other photos of the same person*, and that pairing runs per folder — so subject A is only ever
compared against A. Put two people in one folder and the pairing crosses them, which blends them
together rather than keeping them apart.

Ticking the mode also sets the settings that suit it — identity-learn on, 4 references, a short
identity-first phase, caption dropout at `0.10`, Adapter-relative LR off — and tells you in the
console what it changed. **Nothing is locked**; they're starting points.

Two things it expects of you: **caption both folders yourself**, each with its own unique trigger
word in *every* caption (that word is the only thing telling the two apart), and note the second
folder is training-only — Image Prep, Captions and the Look filter still follow the Start folder.

## Identity-first

An option on identity-learn mode: train the first stretch against the **teacher only**, then drop
the teacher entirely and train on the **photographs only**.

The idea is that photo training then starts from an adapter that already knows who the trigger
word means, instead of discovering the identity and the detail at the same time. Phase 1 runs at
a third of the Learning Rate box, since it's placing the identity rather than reproducing detail.

**Auto** sizes the first phase from your dataset — enough steps for the teacher side to converge,
which is roughly the same number of steps whatever your image count, and therefore rather more
epochs on a small set. Or pick a fixed number of epochs. Off keeps the blended loss.

Phase 2 skips the teacher pass entirely, so it also runs at about half the cost per step.

## Adapter-relative LR

**This is what MiniMax needs to train effectively, and it's on by default.**

A LoRA starts at zero, so a rate that's safe at epoch 1 is far too slow by epoch 50 — and one
that's right later wrecks a fresh adapter. There's no single number that works for both.

**Adapter-relative LR** turns the Learning Rate box into a **ceiling** rather than a setting. The
run starts below it and climbs, keeping every step a fixed fraction of the adapter's current
size:

| Setting | |
|---|---|
| `0.003` | slow build — **the shipped default** |
| `0.005` | climbs faster |
| `0.01` | fast build |

Set the Learning Rate to where you want to *end up*. Every epoch the console reports the
adapter's size, its growth rate, and how much of the ceiling is in use. Off gives a flat run at
the box value.

## Two presets, both standard LoRA

| | |
|---|---|
| **✨ MiniMax H3 Defaults** | dim/alpha 16, 60 epochs, Adapter-relative LR at `0.003` |
| **✨ MiniMax H3 Fast** | dim/alpha **8**, **40 epochs**, flat 2e-4 with no Adapter-relative LR |

Fast reaches likeness in a few hundred steps, and the lower rank tends to come out **more
flexible** — it hasn't room to memorise your backgrounds and framing, so it encodes the subject
instead.

Both ship **standard LoRA**, not LoKR. LoKR moves considerably further per unit of learning rate,
which meant the same Learning Rate box behaved very differently depending on which Network Type
sat above it. LoKR is still a dropdown away.

## A simpler Training tab

- **Per-step movement clip** — retired
- **LR warmup** — retired
- **Weight averaging (EMA)** — still there, now **off by default**
- **Caption dropout** — now has a control (Off / `0.05` / `0.10`). It was fixed at `0.05` with no
  way to change it, and it turns out to matter: it's doing real work on this family, so it stays
  on by default.

Existing presets and saved configs still load; the retired settings load as off.

## Also in this release

- **Re-launching the same dataset no longer re-encodes every image.** The VAE pass is skipped when
  the cache already matches — and it still re-encodes everything if you change Target Megapixels,
  because the check compares the cached latent to the current bucket rather than trusting the
  filename.
- **Identity-learn reports where the learning is coming from.** Each epoch prints the teacher and
  photo errors and what share of the loss real pixels actually carry — which is usually rather
  more than the weight alone suggests, since matching a real photograph is harder than matching
  the model's own output. The teacher weight now also offers `0.4` and `0.5`.
- **Gradient accumulation now works on MiniMax.** The field was on the tab but never reached the
  trainer, so it silently did nothing on this family.
- **Previews can be clips.** A **Sample length** dropdown renders each sample as a short video you
  can scrub in the gallery, for when motion is what you need to check. Stills at 1024×1024 remain
  the default — they render in seconds where a clip takes minutes. **640** added to the sample
  resolution list.
- **A preview that runs out of memory retries shorter** — 141 → 56 → 22 frames — instead of
  dropping the whole run to single frames. And a preview that's crawling now says so, and points
  you at Pause → check an epoch in ComfyUI → Resume.
- **Re-caching with fewer references no longer leaves the old ones behind**, where they could
  still be picked up and train against a pairing from a previous configuration.
- **The Samples tab no longer describes other models** when MiniMax is selected — the controls
  that don't apply to it are gone rather than greyed.

## Known issue

**Resume uses whatever the Training tab currently shows, not the settings the run was launched
with.** If you restart Fizgig and hit Resume without loading the run's settings first, it will
resume with different ones — and if the network type differs, the run fails partway through with
an unhelpful error. Worse, the failed attempt overwrites the last-train snapshot, so *Load
Settings From Last Train* then returns the broken config. Load the run's settings before
resuming. A proper fix is coming.

## Upgrading

Nothing to do. Your model paths, datasets and caches are untouched. If you've saved MiniMax
presets, they'll load with the retired controls off.
