# MiniMax H3 — what we know about the blocks

Working notes for the block-targeting experiments on the Training tab (**Blocks to Train**,
**Train AdaLN**). Written 5 Aug 2026.

Short version: **there is no published map of what H3's blocks do, and our own weight data does
not supply one.** But the architecture rules out one large possibility outright, and that is
worth more than a guessed map.

---

## Confirmed from MiniMax's own material

| | |
|---|---|
| Blocks | **50**, dense single-stream |
| Hidden size | 5376 |
| Attention heads | 56 (head dim 128) |
| Position | 3-axis MM-RoPE over (t, h, w) |
| Total params | ~33 B, of which **~13 B are AdaLN branches** |

Two statements from the official write-ups matter for block targeting:

> "Neither the attention layers nor the FFN layers contain modality-specific structures.
> Modality-specific parameters are confined to the input/output layers and the AdaLN branches."

> "Because the AdaLN modulation outputs can be precomputed and cached, these parameters do not
> need to be loaded for inference-only deployment."

The first says the 50 blocks are genuinely uniform — there is no architectural seam to cut along,
unlike Klein's double/single split. The second is why the pruned checkpoint can replace
`adaln_proj` (13.04 B params) with an 8-dimensional timestep table and remain mathematically
equivalent.

Our `MiniMaxH3Config` matches this spec exactly — checked, not assumed.

---

## The one hard structural fact: AdaLN cannot carry identity

`DiTBlock.forward` (`src/fizgig/minimax/model.py`) calls:

```python
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
```

`t_emb` is the timestep embedding. **The AdaLN branch never sees the latent or the text
conditioning.** Its output is a function of noise level (and the modality tag of each row) and
nothing else.

The consequence is not a heuristic, it is a type constraint: **a LoRA on `adaln_proj` cannot
encode who a person is**, because it cannot distinguish one subject from another. It can only
reshape *how strongly each block contributes at each noise level* — a global, per-timestep gain
schedule shared by every image and every prompt.

That matters because on the pruned checkpoint we *do* train `blocks.N.adaln_proj`, and it carries
roughly **45% of all weight movement** in a matched reference epoch. Close to half the LoRA's
learning goes somewhere structurally incapable of storing a face.

This is not automatically waste — reshaping per-block gain across the noise axis is plausibly
useful, and is the same lever **Detail Focus** pulls from the other end. But it is the sharpest
available experiment, because it has a mechanism behind it rather than an analogy.

**Note the asymmetry between checkpoints:** the full bf16 model excludes AdaLN from training
already (its `adaln_proj` is `[96768, 2688]`, and ComfyUI's pruned inference builds drop those
keys entirely, so training them is not deploy-consistent). The toggle therefore only does
anything on the pruned int8 checkpoint.

---

## What our own weight data says: nothing useful

Per-block `||dW||` measured on two finished H3 LoRAs (`tests/diag_minimax_block_energy.py` —
all 50 blocks have identical shapes, so raw norms are directly comparable and the 21 GB base
never has to be loaded):

```
                        early 0-15   middle 16-31   late 32-49
new88-2em4-3   (30 ep)     28.6%        33.5%         38.0%
new88-2em4-2p5 ( 9 ep)     27.9%        35.8%         36.3%
```

Essentially flat. Across individual blocks the spread is only **3x** quietest to loudest against
a 2% flat expectation, and the token refiner takes 2–3% for 8 of 258 modules — exactly its share.

The decisive part is that **the quietest blocks do not agree between runs**:

```
run 1:  1, 6, 7, 9, 11, 12, 14, 15, 21, 25, 40, 41
run 2:  1, 2, 3, 4,  5,  6,  9, 10, 39, 40, 41, 49
```

Five overlap; picking 12 of 50 at random would overlap ~3. Freeze one run's quiet blocks and you
freeze the other's busy ones.

Two caveats keep this from being read as "all blocks matter equally":

1. **Movement is not contribution.** The optimizer-eps bug produced the largest drift in the
   model in a layer that was simply broken. A map built from norms would have pointed straight at
   it.
2. **These LoRAs trained on all 50 blocks.** They show where learning *went* when everything was
   available, not what is *sufficient*. A flat profile with a peaked contribution profile is
   exactly the case where pruning pays.

---

## Prior art: none

Checked, 5 Aug 2026:

- **[ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3)** —
  despite the name, an inference accelerator. Forecasts the hidden feature after the final block
  and skips transformer evaluations wholesale. No per-layer analysis.
- **[MiniMax-H3-FineTuning](https://github.com/IAmIronMan42/MiniMax-H3-FineTuning)** — LoRAs
  `to_qkv, to_out.0, linear_1, linear_2` uniformly across every block.
- **ai-toolkit** — same, uniform across all blocks.

Nobody has published which H3 blocks do what.

---

## How to read an experiment

**Blocks to Train** and **Train AdaLN** are independent axes. Vary one at a time.

A selection that trains a good likeness means **its complement was not needed** — whichever end
that turns out to be. There is no result that disproves the idea; a surprising winner relocates
the answer. Both halves working would mean identity is redundantly represented, and you pick
whichever behaves better.

Two things that will mislead you if unaccounted for:

- **Fewer blocks is less total capacity.** A better-placed selection can still look worse at
  matched epochs. Give a narrow selection a few more epochs before calling it.
- **Judge in ComfyUI.** H3's single-frame training previews are known to be weaker than the same
  checkpoint rendered properly.

Every run records `ss_train_blocks` and `ss_train_adaln` in the LoRA's metadata, and the training
queue shows both, so a queued sweep stays readable afterwards.

### Why this is worth doing at all

A block that carries no identity still receives gradient, and what is left for it to learn is the
dataset's backgrounds, framing and lighting. Excluding it removes capacity that would otherwise
go into memorising the set. The upside is a *cleaner* likeness, not merely a faster run.

---

## Sources

- [MiniMax H3 is now open source](https://www.minimax.io/news/minimax-h3-open-source)
- [MiniMax H3 research blog](https://www.minimax.io/blog/minimax-h3)
- [MiniMaxAI/MiniMax-H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
