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
| Training, 32 GB | 0.85 s/it | **0.70 s/it** |
| INT8 preview path | 1.02× vs bf16 (i.e. nothing) | **2.10×** |
| 1024 preview | 5.15 s | **4.40 s** |
| Training attention | — | unchanged, deliberately |

### 1. Block swap was the whole story
`_auto_krea2_blocks_swap` picked a swap count from VRAM, which handed 16 GB cards the worst
available configuration: fp8 doesn't fit, so it swapped 20 of 28 blocks every step.

    fp8, no swap   0.85 s/it   20.1 GB   12.5% CPU
    fp8, swap 20   3.09 s/it   12.3 GB   49.9% CPU   <- what 16 GB cards got
    NF4, no swap   0.70 s/it   13.8 GB   14.0% CPU

Replaced by `utils/capabilities.py`, which probes the machine (running a real tiny matmul rather
than consulting an sm-version table, so it is right on hardware nobody here can test) and picks a
*strategy*: NF4-no-swap > fp8-no-swap > swapping. Budgets from **free** VRAM, not the card total —
a "16 GB" card reports ~15.9 GiB and may have GBs held by a browser or ComfyUI.

### 2. The INT8 fast path was doing nothing
`modules/int8.py` stored weights pre-transposed as (K,N) with a comment explaining this avoided
"a per-call transpose that would eat the speedup". Backwards: `.t()` on a contiguous (N,K) tensor
is a free view, and that layout is what int8 tensor cores want.

    weight (K,N) contiguous, _int_mm(x, W)      0.452 ms
    weight (N,K) contiguous, _int_mm(x, W.t())  0.131 ms   <- 3.4x faster

"INT8 fast inference" is on by default and used by Royale, Repair Studio and the Explorer, so
users were paying int8's quantisation error for no speed at all.

### 3. cuDNN attention — inference only
Fast forwards, but it plans a kernel per shape and re-plans on every change. Bucketed training
churns shapes constantly.

    fixed shape (fwd+bwd)     default 1.904 ms   cuDNN 0.656 ms
    varying shapes            default 2.248 ms   cuDNN 26.353 ms

Selected on `torch.is_grad_enabled()`: cuDNN when nothing is recording gradients (previews,
Royale, Repair Studio, sampling), PyTorch's default in a training step. Not a backward-pass
problem — cuDNN's backward is fine at a fixed shape.

`torch.backends.cudnn.benchmark` defaults to False, and that is the *bad* setting: the heuristic
path is 51.6 ms under shape churn against 0.757 ms when autotuning is allowed. Enabled with cuDNN.

Even so, cuDNN loses for training across three variants (plain 1.57, bucket-grouped 1.29,
autotuned 1.28 vs 0.704), so inference-only stands on measurement, not theory.

### 4. Things measured and deliberately NOT shipped
- **Bucket-grouped batch ordering** (what OneTrainer's `AspectBatchSorting` does). Random
  shuffling changes shape on ~43% of steps. Grouping made **no** difference (0.7042 both) and did
  not rescue cuDNN. Kept behind `FIZGIG_BUCKET_ORDER=1` for a future torch.compile, but off:
  grouping correlates consecutive gradients, and that is a real if modest risk for nothing.
- **fp8 `_scaled_mm`** — 4.2× on the matmul, but requires fp8 activations, and NF4/int8 both beat
  fp8 end to end anyway.

---

## INT8 training (wired, not default)

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

**Not made default.** A per-Linear error figure is not evidence about trained LoRA quality, and
today produced two cases where an isolated measurement inverted end to end.

---

## Plan

### Now: validate int8 with a full-length run
A 3-epoch benchmark proves throughput, not numerical stability over 40 epochs. Run int8 and NF4
at matched seed/epochs/dataset and compare the resulting LoRAs on likeness. Decides whether int8
becomes the default above ~20 GB free.

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

### Then: optimizer options for Krea 2 (Phase 3)
Krea 2 hardcodes AdamW8bit; OneTrainer offers ~40. Requested twice in the community thread. Not a
speed item — the one place "they have something we don't" still stands unqualified.

### Blocked: torch.compile
Triton is not installed in the venv, so `torch.compile` cannot run at all. Needs `triton-windows`
first. Would fuse the quantise/dequantise elementwise work that currently bounds these paths.
Note the known conflicts: bucketing (recompiles per shape), block swap (graph breaks), and
rotation fine-tuning (the trainable set changes every window).

---

## What was verified about OneTrainer

Read from a clone of the repo, not inferred:

- **Attention default is torch SDPA**, same as ours. cuDNN is one of three options, not the
  default — the community claim that it defaults to cuDNN is wrong, and anyone selecting it for
  bucketed training is likely taking the same shape-churn penalty we measured.
- **`offload_fraction` defaults to 0.0** — they never swap; they quantise until it fits. That was
  the entire speed difference.
- Their int8 path uses `torch._int_mm` (not `_scaled_mm`, which is fp8-only in torch 2.10).
- `AspectBatchSorting` groups batches by resolution.
- ~40 optimizers, default AdamW. Pinned to torch 2.12.
- Their `LayerOffloadConductor` is better-shaped than our block swap: a continuous fraction rather
  than a block count, **activation offloading as a separate knob**, dedicated CUDA streams and
  pinned memory. Worth borrowing for 8–12 GB cards.

---

## Method note

Three of the four wins were pre-existing code that was subtly wrong — a swap default, an inverted
weight layout, a backend never selected. Only the benchmark harness was new code. Two conclusions
also had to be reversed after re-measuring (cuDNN "slow backward" was actually shape churn; int8
"unavailable" was the wrong API — `_int_mm`, not `_scaled_mm`). Isolated microbenchmarks were
wrong about the end-to-end result twice. Measure the whole step.
