# Design note: shared + private rank partitions for mixed voice/visual LoRAs

*Status: design candidate, post-release material. Not scheduled.*
*Origin: Peter's question (15 Aug 2026) — "is there a mathematical way for voice and video to
learn together BUT keep the files separate until train end?" — refined across the discussion
into the three-way split below, which was his rank-routing idea plus a shared core.*

## The problem

A mixed dataset (photos/clips + voice recordings) trains one LoRA whose every weight receives
gradients from both categories. That sharing is valuable — one "this person" concept whose
face and voice pull on common circuitry, bound to one trigger — but it makes the result
inseparable: there is no principled way, after training, to take "the voice as it was at
epoch 12" and "the visuals as they were at epoch 40" out of one adapter. Gradient updates
carry no category labels, and Adam's momentum/normalisation mixes histories further.

The shipped **per-category retirement** feature (anchor at a real 10% LR / stop) covers the
practical case — hold a finished category while the other cooks — but the file is still one
fused object, and the held category still drifts slightly (anchor) or goes blind (stop).

## Why the obvious fixes fail

- **Twin adapters with per-step gradient routing** (adapter V gets visual grads, adapter A
  audio grads, both active in every forward): clean separation, near-zero extra cost (the 33B
  base dominates; merging is exact rank concatenation, already shipped in the Repair Studio
  bake). But **no parameter is ever shared** — the identity concept is learned twice, split
  across files. Peter's verdict: losing the shared adapter is the one cost he won't pay.
- **Rank-partitioned single adapter** (rank 16, visual steps write ranks 1–8, audio steps
  ranks 9–16): identical to twins in one file. A rank-16 LoRA's delta decomposes as
  `B[:,:8]·A[:8,:] + B[:,8:]·A[8:,:]` — the slices contribute additively and share nothing.
  Interleaving rank indices changes nothing (rank order is meaningless). Sharing and
  category-separability over the SAME parameters are mathematically opposed: sharing *is*
  entanglement.
- **Post-hoc decomposition of a normally-trained adapter**: doesn't exist in a principled
  form. (A per-step applied-delta ledger — ΔW_total = ΔW_visual + ΔW_audio by bookkeeping —
  is exact as accounting, but recombining truncated ledgers is task-arithmetic extrapolation,
  and it preserves nothing structurally.)

## The design: three-way rank split

One adapter, rank R (e.g. 16), partitioned by *role* rather than fully by category:

| Ranks | Role | Trained by |
|---|---|---|
| 1–8 | **shared core** | every step, both categories — the entangled identity, on purpose |
| 9–12 | **visual-private** | visual steps only |
| 13–16 | **audio-private** | audio/voice steps only |

- The shared core keeps everything the single adapter buys today: cross-modal features
  learned once, trigger binding intact. Its inseparability stops being a limitation and
  becomes the design statement — the part you'd never want to split *can't* be split.
- The private slices hold each modality's residual — what that category needs beyond the
  shared concept — and are separable **by construction**: export surgery is slicing rank
  columns out of one file. "Shared@end + visual@epoch-X + audio@epoch-Y" is a supported
  operation, not a hack.
- Classic multi-task learning shape (shared trunk + task-specific capacity), expressed inside
  the LoRA rank dimension.

## Mechanics (all straightforward)

- **Routing**: per step, zero/skip gradients for the other category's private rank rows/cols.
  Adam state is per-parameter, so each slice's optimizer history stays clean. The voice-step
  detection (`batch["audio_only"]`) already exists; clips count as visual (their soundtrack
  rides with the clip, as in retirement).
- **Checkpointing**: each epoch save already contains all ranks; surgery needs nothing extra
  at save time. An export tool slices `lora_up[:, ranks]` / `lora_down[ranks, :]` from chosen
  checkpoints and concatenates — the Repair Studio bake's math, applied column-wise.
- **Retirement interaction**: subsumes it. Freezing a private slice = retiring that
  modality's residual while the shared core keeps learning from both. Could replace the
  current anchor/stop for mixed runs, or compose with it.
- **Metadata**: record the partition (`ss_rank_partition = "shared:8,visual:4,audio:4"`) so
  the export tool and ComfyUI-side users know what the file carries.

## Open questions / caveats

1. **Capacity split is a new hyperparameter** with zero data. Start 50/25/25 of total rank;
   measure. Too small a shared core pushes identity into the private slices (defeating the
   point); too small a private slice starves the modality residuals.
2. **Cross-epoch pairing is still an extrapolation.** A private slice at epoch X co-adapted
   with the shared core's trajectory up to X; pairing it with the core at epoch Y ≠ X is
   task-arithmetic surgery — expected to degrade gracefully (LoRA arithmetic usually does),
   guaranteed only near the diagonal.
3. **Does routing weaken the core?** The shared core sees both categories but its gradient
   *mix* differs from today's (private slices absorb some of each category's residual).
   Whether that changes likeness/voice quality is empirical.
4. **The decisive first experiment**: same mixed dataset, (a) plain rank-16 vs (b) 8/4/4
   split, both exported at matched epochs (diagonal pairing). If (b) ≈ (a) on likeness AND
   voice, the off-diagonal freedom comes essentially for free and the feature is justified.
   If (b) is worse, the shared-representation cost is real and the idea dies cleanly.

## Why not now

Touches gradient routing, the network class, export, metadata and the GUI at once — release
work first. Retirement covers the practical need meanwhile. Revisit when a concrete use case
outgrows retirement (e.g. the demo-video workflow wanting voice@12 + visuals@45 pairings as
routine practice).
