# Krea 2 performance — measured findings and plan

Branch: `perf/benchmark-and-backends`. Everything below is measured on an RTX 5090 (torch
2.10.0+cu128) with `scripts/bench_train.py`, on a 36-image dataset at 0.25 MP, batch 1.

This started from community reports that Fizgig trained Krea 2 more slowly than OneTrainer and
pegged the CPU (70–90% vs <5%). Every gap traced to Fizgig's own settings or to code that was
subtly inverted — not to missing kernels.

---

## Shipped on this branch

| Area | Before | After |
|---|---|---|
| Training, 16 GB cards | 3.09 s/it, 50% CPU | **0.70 s/it, 14% CPU** |
| Training, 24 GB+ | 0.85 s/it | **0.610 s/it** (INT8, auto-selected) |
| INT8 preview path | 1.02× vs bf16 (i.e. nothing) | **2.10×** |
| Krea 2 1024 preview | 5.15 s | **4.40 s** |
| Klein 1024 preview | 3.01 s | **2.74 s** |
| Optimizer choice (Krea 2) | AdamW8bit, hardcoded | **7 families + any module path** |
| Attention device syncs per step | 56 (every block, twice) | **1** |
| Training attention backend | — | unchanged (cuDNN wins only past ~1260 steps) |

### 1. Block swap was the whole story
`_auto_krea2_blocks_swap` picked a swap count from VRAM, which handed 16 GB cards the worst
available configuration: fp8 doesn't fit, so it swapped 20 of 28 blocks every step.

    fp8, no swap   0.85 s/it   20.1 GB   12.5% CPU
    fp8, swap 20   3.09 s/it   12.3 GB   49.9% CPU   <- what 16 GB cards got
    NF4, no swap   0.70 s/it   13.8 GB   14.0% CPU

Replaced by `utils/capabilities.py`, which probes the machine (running a real tiny matmul rather
than consulting an sm-version table, so it is right on hardware nobody here can test) and picks a
*strategy*: INT8-no-swap > NF4-no-swap > fp8-no-swap > swapping. Budgets from **free** VRAM, not
the card total — a "16 GB" card reports ~15.9 GiB and may have GBs held by a browser or ComfyUI.

    >17.7 GB free ..... INT8   0.637 s/it, forward err 1.3e-02
    >12.9 GB free ..... NF4    0.709 s/it, forward err 9.2e-02
    below ............. NF4 + swap
    no bitsandbytes ... fp8 + swap, with an install note

INT8 leads because it is both faster and more accurate than the NF4 it replaces; it costs ~5 GB,
which is why NF4 remains the 16 GB path. The probe targets `torch._int_mm`, which needs only
Turing-era int8 tensor cores — so a 3090 gets INT8 too, unlike fp8 `_scaled_mm` (sm_89+).

### 2. The INT8 fast path was doing nothing
`modules/int8.py` stored weights pre-transposed as (K,N) with a comment explaining this avoided
"a per-call transpose that would eat the speedup". Backwards: `.t()` on a contiguous (N,K) tensor
is a free view, and that layout is what int8 tensor cores want.

    weight (K,N) contiguous, _int_mm(x, W)      0.452 ms
    weight (N,K) contiguous, _int_mm(x, W.t())  0.131 ms   <- 3.4x faster

"INT8 fast inference" is on by default and used by Royale, Repair Studio and the Explorer, so
users were paying int8's quantisation error for no speed at all.

### 3. cuDNN attention — inference yes, training it depends (and NOT for the reason first given)
cuDNN builds a plan the first time it sees each shape, and that build is expensive. Measured on
the REAL INT8 Krea 2 model with real cached batches — fwd + bwd + optimizer, the whole step:

                          first sight    steady state    one-time, 36 shapes
        default backend     582.2 ms       572.2 ms            0.4 s
        cuDNN              1829.9 ms       536.6 ms           46.6 s

**Warm, cuDNN is ~6% FASTER.** It costs ~1.3 s per distinct shape to get there, so which one wins
is arithmetic: saves 37 ms/step, costs 46.6 s once, breaks even near **1260 steps** — about 35
epochs on a 36-image set. Inference is the unambiguous case for it (one resolution held for a
whole render, plan built once):

        Krea 2 1024 preview   5.15 s -> 4.40 s
        Klein  1024 preview   3.01 s -> 2.74 s

Selected on `torch.is_grad_enabled()`. Training keeps the default backend because it is the one
that **cannot lose**, not because cuDNN is bad; `FIZGIG_SDPA_BACKEND=cudnn` is there for long runs.

**Two earlier conclusions in this document were wrong.**

1. *"cuDNN is ~2x slower for training"* — a short-benchmark artifact. 46.6 s of plan building
   charged against a 108-step benchmark is +431 ms/step, which accounts for essentially all of the
   +370 ms gap that was measured. The 3-epoch A/B could not see past its own warm-up.
2. *"cudnn.benchmark is worth 68x"* — it makes **no measurable difference at all**. Plan building
   with autotuning off: 66.0 s, on: 65.9 s. Steady state 535.1 vs 536.6 ms/step. The original
   51.573-vs-0.757 ms figure never reproduced. `modules/sdpa.py` no longer sets the flag —
   setting a global torch flag with no measurable effect is worse than leaving it alone
   (`FIZGIG_CUDNN_BENCHMARK=1` opts back in).

**What the cost actually scales with is the number of DISTINCT shapes**, and that is self-inflicted.
The DiT pads the sequence to a multiple of 256 *explicitly to keep kernel shapes stable*, and the
attention trim immediately undoes it: the trimmed length carries each caption's own token count, so
36 images produce **30 distinct shapes, not the 7 image buckets** anyone would assume. `FIZGIG_ATTN_TRIM=0`
keeps the padded shapes (4 instead of 30) at the cost of attending over padding — measured 0.6289
vs 0.6098 s/it on the default backend, so the trim stays on by default.

Also settled while chasing this: **no mask ever reaches SDPA in training.** At batch size 1 the
"are all sequences the same length" test is trivially true, so attention always trims and passes
`attn_mask=None`. The earlier guess that variable-shaped masks were the cost was simply wrong.

VAE attention is deliberately left alone: single-head with head_dim = channels, which cuDNN
cannot run at all.

### 4. A device sync inside every block
`attention()` decided whether it could trim to a common sequence length by reading
`attn_params.seqlens[0].item()` — a CUDA-to-CPU sync — and it ran that check **per block**: 28
stalls per forward, 56 per step under gradient checkpointing. `seqlens` is fixed for the whole
forward, so the decision now happens once, in `AttentionParams.__post_init__` (Krea 2 and Klein
both). Behaviour is identical, covered by `tests/test_attention_trim.py`.

    int8                       0.625  s/it
    int8, no per-block sync    0.6098 s/it

That is ~2.5%, which is close enough to run-to-run noise (±2% across repeats) that I would not
claim it as a speed win on its own. It is worth having regardless: it removes 56 avoidable device
stalls per step, and it was the graph break that made torch.compile lose (see below).

It does **not** change the cuDNN verdict — the sync was present in both arms, so it could never
have explained the gap. Re-measured after the fix, cuDNN still came out at 1.17 vs 0.6098 on a
3-epoch benchmark, which is what finally forced the question of *why* an isolated kernel that
benchmarks 3x FASTER loses end to end. The answer is in section 3: warm-up, not the kernel.

### 5. Things measured and deliberately NOT shipped
- **Bucket-grouped batch ordering** (what OneTrainer's `AspectBatchSorting` does). Random
  shuffling changes shape on ~43% of steps. Grouping made **no** difference (0.7042 both) and did
  not rescue cuDNN. Kept behind `FIZGIG_BUCKET_ORDER=1` for a future torch.compile, but off:
  grouping correlates consecutive gradients, and that is a real if modest risk for nothing.
- **fp8 `_scaled_mm`** — 4.2× on the matmul, but requires fp8 activations, and NF4/int8 both beat
  fp8 end to end anyway.

---

## INT8 training (now auto-selected above ~19 GB free)

`--quant_int8 bf16|int8`. The base is frozen, so only grad-w.r.t.-input is needed — weight
gradients, where int8 does the most damage, are never computed. With gradient checkpointing the
forward runs twice per step, so an int8 forward pays even with a bf16 backward.

    int8 (int8 grads) .... 0.6098 s/it   18.3 GB
    int8 (bf16 grads) .... 0.6369 s/it   18.6 GB
    NF4 (default) ........ 0.7092 s/it   13.6 GB
    fp8 .................. 0.8264 s/it   20.3 GB

The expected precision trade runs the other way. Per-Linear forward error at 6144²:

    INT8 W8A8 ..... 1.306e-02
    NF4 ........... 9.229e-02      <- 7x worse (8-bit vs 4-bit)

So int8 is ~11% faster than NF4 **and** ~7× more accurate, with exact gradients. What it costs is
5 GB: at 18.6 GB it does not fit the 16 GB cards NF4 exists for. Independently corroborated —
SimpleTuner reports int8 LoRA training of Flux (~12B) at "a bit more than 18 GB".

**Now the default where it fits** (`recommend_krea2_strategy`), on the strength of being faster
*and* more accurate than the NF4 it displaces. The honest caveat: that rests on a per-Linear error
figure and a 3-epoch throughput run, **not** on a trained-LoRA comparison. The planned 40-epoch
int8-vs-NF4 A/B was cancelled before it ran. Every signal points one way, and this work has twice
produced isolated measurements that inverted end to end — so this is the one default on the branch
worth putting a real run behind before it reaches master.

---

## Plan

### Outstanding: validate int8 with a full-length run
Run int8 and NF4 at matched seed/epochs/dataset and compare the resulting LoRAs on likeness.
Confirms (or reverses) the default above.

### NVFP4 — prototyped, hypothesis WRONG, parked

Blackwell has fp4 tensor cores and `torch._scaled_mm` takes `float4_e2m1fn_x2` with blockwise
1x16 scales. The raw matmul is genuinely fast at 6144²:

    NVFP4 ..... 0.070 ms   <- 4-bit
    INT8 ...... 0.121 ms
    bf16 ...... 0.594 ms

I expected it to be *more accurate* than NF4 as well, since its scales cover 16 elements rather
than NF4's 64. It is not:

    INT8 W8A8 ..... 1.31e-02
    NF4 ........... 9.23e-02
    NVFP4 ......... 1.03e-01   <- slightly WORSE than NF4

The reason is the codebook. **NF4 is NormalFloat4** — its 16 levels are placed to be optimal for
normally-distributed values, which is what network weights are. NVFP4 is plain e2m1, whose eight
magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} are spread in exponent space and not matched to a Gaussian.
The better codebook beats the finer scaling.

So NVFP4 is a **pure speed play at NF4-level accuracy**, not the best-of-both it looked like. That
needs a custom quantiser with NVIDIA's swizzled blockwise scale layout (my prototype's `_scaled_mm`
path returns 51% error — a layout bug, though the weight-only comparison above is layout-independent
and stands), and it is Blackwell-only.

**Parked.** Revisit only if 16 GB users report speed as their blocker once the NF4 default lands.
`tests/test_nvfp4_accuracy.py` reproduces the comparison.

### Optimizer options for Krea 2 (Phase 3) — SHIPPED
`training/optimizers.py`, used by `krea2_train` via `--optimizer_type` / `--optimizer_args` and by
the GUI's Optimizer section (the dropdown re-populates per family, since Klein resolves names its
own way). Seven families where the packages are present — adamw8bit (default), adamw,
pagedadamw8bit, ademamix8bit, pagedademamix8bit, lion8bit, adafactor — plus prodigy/came if
installed, plus any `module.path.ClassName`. Entries whose package is missing are filtered out
rather than offered and then failing.

Two things shaped the implementation more than the list did:

- **LRs do not transfer between families.** Lion wants ~1/10 an AdamW LR (it applies the sign of
  the update); Prodigy wants `lr=1.0`. A dropdown without that warning is a LoRA-frying trap, so
  `create_optimizer` logs a loud warning when the LR looks wrong for the family.
- **Optimizer memory is nearly irrelevant here.** A LoRA's state is tens of MB against a 13-19 GB
  base, so the usual "8-bit saves VRAM" argument barely applies — what the choice buys is update
  behaviour. Worth saying out loud, because OneTrainer's optimizer count reads as a bigger
  advantage than it is for LoRA work specifically.

Construction failures fall back to plain AdamW with a warning rather than killing the run, and the
choice is recorded as `ss_optimizer` in the output LoRA.

### torch.compile — measured properly, NOT shipped (but it found a real bug)
`triton-windows==3.5.1.post24` installed (matches torch 2.10; pulls no dependencies and does not
touch torch), so the question could finally be measured instead of assumed. On a real
434M-parameter Krea 2 block, forward+backward:

    bf16, fixed shape, default mode   19.924 ms -> 18.837 ms   (1.06x, 14 s to compile)
    bf16, fixed shape, max-autotune   19.729 ms -> 16.705 ms   (1.18x, 94 s to compile)
    bf16, 5 bucket shapes            100.018 ms -> 94.693 ms   (1.06x)
    INT8, fixed shape                 13.980 ms -> 10.261 ms   (1.36x)
    INT8, 5 bucket shapes             70.539 ms -> 51.645 ms   (1.37x)

The INT8 number is the interesting one — inductor fuses exactly the quantise/dequantise
elementwise work that bounds that path, and `dynamic=True` holds the full gain while cutting the
first pass from 27 s to 11 s. That looked like a ship.

**End to end it is still slower**, on the same dataset and hardware as every other number here:

    int8              0.6098 s/it
    int8 + compile    0.7937 s/it     <- 30% WORSE

So it stays off, and there is no GUI toggle — a knob that measures worse should not be user
facing. `--compile_blocks` remains on the CLI, labelled experimental, as a hook for revisiting.
Two Windows-specific things it needed and now has: inductor requires MSVC for its host-side C++
(auto-discovered via vcvars64, since Visual Studio only exposes `cl.exe` inside a developer
prompt), and `torch._dynamo.config.suppress_errors` so a compile failure costs speed rather than
killing the run.

**What it found on the way** is worth more than the feature. The first end-to-end run was 1.5x
slower, and dynamo said why: a graph break at `attention.py:148`, on
`attn_params.seqlens[0].item()` — a CUDA-to-CPU sync in the trim check, running inside **every one
of the 28 blocks**, 56 times per step with gradient checkpointing's recompute. It is now resolved
once per forward in `AttentionParams.__post_init__` (both Krea 2 and Klein).

That is also the fourth time an isolated measurement pointed the wrong way, and this one has a
precise cause: my block-level benchmark built `AttentionParams` with no mask, so it never entered
the trim branch at all. **The microbenchmark was measuring a different code path than training
runs.** Fixing the break took compile from 0.943 to 0.794 s/it — a big improvement, still a loss.

---

## What was verified about OneTrainer

Read from a clone of the repo, not inferred:

- **Attention: the global default is SDP, but every shipped Krea 2 preset selects CUDNN.** Exactly
  4 of their 60 presets set `attention_mechanism: CUDNN`, and all 4 are the Krea 2 ones (LoRA and
  Finetune, 16 and 24 GB). So my earlier "the community claim that it defaults to cuDNN is wrong"
  was too broad: it is wrong about the global default and right about Krea 2 in practice, because
  anyone using their preset for this model is on cuDNN. That sits directly against our own
  measurement (cuDNN 1.6-1.9x slower for training even with stable shapes), which makes it the
  single most interesting thing to check in the comparison — either their attention path keeps
  shapes stable in a way ours does not, or the preset is inherited rather than measured.
- **`offload_fraction` defaults to 0.0**, and their 24 GB Krea 2 preset uses it — but the **16 GB
  preset offloads 0.3**. So they do offload on small cards; just far less than the 20-of-28 (71%)
  our old auto-swap was handing 16 GB users.
- **Their Krea 2 transformer recipe is `INT_W8A8` on BOTH 16 and 24 GB.** Independent corroboration
  of today's int8 default from a second implementation, on this exact model. The design difference
  is what happens when it does not fit: they keep int8 and offload 30%, we drop to NF4 with no
  offload. Worth measuring against each other — 0.3 offload is not the same animal as 71% swap.
- Their int8 path uses `torch._int_mm` (not `_scaled_mm`, which is fp8-only in torch 2.10).
- `AspectBatchSorting` groups batches by resolution.
- ~40 optimizers, default AdamW. Pinned to torch 2.12. The count is the headline, but most of that
  list is variants of a handful of families; Fizgig now covers the families that matter for LoRA
  work plus an escape hatch to any installed optimizer class, which closes the gap in practice
  without pretending 40 entries is 40 ideas.
- Their `LayerOffloadConductor` is better-shaped than our block swap: a continuous fraction rather
  than a block count, **activation offloading as a separate knob**, dedicated CUDA streams and
  pinned memory. Worth borrowing for 8–12 GB cards.

---

## Method note

Most of the wins were pre-existing code that was subtly wrong — a swap default, an inverted weight
layout, a backend never selected, a device sync inside a loop. Only the benchmark harness and the
optimizer factory were new code. Three conclusions had to be reversed after re-measuring (cuDNN
"slow backward" was actually shape churn; int8 "unavailable" was the wrong API — `_int_mm`, not
`_scaled_mm`; NVFP4's finer scales were assumed to beat NF4's better codebook and don't).

Isolated microbenchmarks pointed the wrong way about the end-to-end result **four times** — and
then, on the fifth, the end-to-end benchmark was the one that lied. cuDNN's kernel really is 3x
faster; the 3-epoch A/B just could not see past 46 s of one-time plan building. Both directions
have the same cure: know what your measurement includes. A microbenchmark tests the code path you
set up rather than the one that runs (the torch.compile case built `AttentionParams` without a
mask and never entered the branch training takes); a short end-to-end run charges warm-up costs
to steady state. Separate cold from warm, and measure the whole step.
