# Fizgig v3.6.0 — MiniMax H3 previews you can trust

> **DRAFT — not tagged.** Pending validation from the three Adapter-relative LR runs.

MiniMax H3 samples used to come out softer than the same checkpoint rendered in ComfyUI, which
made them useful for tracking a run and not much else. They now render the same way ComfyUI
does. That single change makes everything else on the Training tab easier to judge — and it let
us take three controls off the tab entirely.

## Previews now match ComfyUI

Four things in the preview path were doing something slightly different from the reference
implementation. Each was small on its own; together they were the difference between a sample
that shows your LoRA and one that only hints at it.

| | |
|---|---|
| **Video decode** | Runs at the precision the decoder's own weights ship in. Fixes soft detail and banding in smooth gradients. |
| **Audio track** | H3 generates sound alongside picture, and the picture is conditioned on it every step. The audio is now stepped exactly as ComfyUI steps it. |
| **Base weights** | The quantised base is unpacked without the small per-channel gain error it used to pick up. |
| **Modulation** | The per-block gain that scales attention and MLP now runs at full precision, as the reference does. |

**Two of those improve training, not just previews.** Your LoRA is now fitted against a more
accurate base, so runs from this version won't line up exactly with older ones. If you're
mid-comparison, finish it on one version or the other.

## Adapter-relative LR

A LoRA starts at zero, so the first steps are enormous relative to its own size — which is where
epoch-1 distortion comes from. Later, once the adapter has grown, the same learning rate is
barely moving it. The rate that's safe at the start is too slow by the end, and there's no single
number that's right for both.

**Adapter-relative LR** turns the Learning Rate box into a **ceiling** rather than a setting. The
run starts well below it and works its way up, holding every step at a fixed fraction of the
adapter's current size:

| Setting | |
|---|---|
| `0.003` | slow build — **the shipped default** |
| `0.005` | climbs faster |
| `0.01` | fast build |

You get a gentle start without picking a warmup length, and full speed later without picking the
moment. Every epoch the console prints the adapter's size, its growth rate, and how much of your
ceiling is in use. Set it Off for a flat run at whatever the box says.

## The MiniMax preset now defaults to LoRA

The shipped preset is **standard LoRA at dim/alpha 16**, not LoKR. LoKR moves considerably
further per unit of learning rate, which meant the same Learning Rate box behaved very
differently depending on which Network Type sat above it. LoRA is the one the best results on
this family came from, and LoKR is still a dropdown away if you prefer it.

## A simpler Training tab

With the above doing the job, three controls came off:

- **Per-step movement clip** — retired
- **LR warmup** — retired
- **Weight averaging (EMA)** — still there, now **off by default**. Worth switching on when
  you're pushing the learning rate hard.

Existing presets and saved configs still load; the retired settings load as off.

## Also in this release

- **Gradient accumulation now works on MiniMax.** The field was on the tab but never reached the
  trainer, so it silently did nothing on this family.
- **Previews are clips.** Samples render as 56-frame clips at 640×640 that you can scrub in the
  gallery — the regime the model was built for. Both are dropdowns on the Samples tab; raise them
  when a preview needs to be judged rather than glanced at, and set *Generate every N epochs* to
  match, because a clip costs minutes rather than seconds.
- **640** added to the sample resolution list.
- **Train at 1 MP by default.** H3's canvas is 768 on the short edge, so training much below that
  starves the detail. 0.25 MP is still right for a tightly face-cropped set.
- **A clip preview that runs out of memory now retries shorter** — 141 → 56 → 22 frames — instead
  of dropping the whole run to single frames.

## Known issue

**Resume uses whatever the Training tab currently shows, not the settings the run was launched
with.** If you restart Fizgig and hit Resume without loading the run's settings first, it will
resume with different ones — and if the network type differs, the run fails partway through with
an unhelpful error. Load the run's settings before resuming. A proper fix is coming.

## Upgrading

Nothing to do. Your model paths, datasets and caches are untouched. If you've saved MiniMax
presets, they'll load with the retired controls off.
