# Third-Party Notices

Fizgig is licensed under the Apache License, Version 2.0 (see `LICENSE`).

It includes code derived from the third-party projects listed below. Each is
adapted under its own license; all are permissive and compatible with
Apache-2.0. Where a file was modified, that is noted — Fizgig's changes are
themselves released under Apache-2.0.

---

## musubi-tuner — Apache License 2.0

Upstream: https://github.com/kohya-ss/musubi-tuner
Copyright the musubi-tuner authors (kohya-ss and contributors).

The following Krea 2 modules are adapted (and modified) from musubi-tuner:

- `src/fizgig/krea2/offloading.py` (block-swap offloader)
- `src/fizgig/krea2/fp8_optimization_utils.py` (fp8 quantization)
- `src/fizgig/krea2/safetensors_utils.py` (mmap safetensors I/O)
- `src/fizgig/krea2/lora_utils.py`
- `src/fizgig/krea2/attention.py` (attention dispatch)
- `src/fizgig/krea2/vae_loader.py` (Qwen-Image VAE loader/converter)
- training hooks (gradient checkpointing, block-swap wiring) in
  `src/fizgig/krea2/model.py` and the flow-matching training recipe in
  `src/fizgig/krea2/trainer.py`

Licensed under the Apache License, Version 2.0:
http://www.apache.org/licenses/LICENSE-2.0

---

## ai-toolkit (Ostris, LLC) — MIT License

Upstream: https://github.com/ostris/ai-toolkit

The Krea 2 single-stream MMDiT backbone (`src/fizgig/krea2/model.py`) and the
functional flow-matching sampler (`src/fizgig/krea2/sampling.py`) are ported
from ai-toolkit's `extensions_built_in/diffusion_models/{krea2,flux2}/src`,
then adapted for Fizgig.

```
MIT License

Copyright (c) 2024 Ostris, LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## FLUX (Black Forest Labs) — Apache License 2.0

Upstream: https://github.com/black-forest-labs/flux

The Klein 9B DiT (`src/fizgig/klein/model.py`) is a Fizgig-native implementation
based on the FLUX reference model code. Licensed under the Apache License,
Version 2.0. (Model weights are distributed separately under their own license —
see "Note on model weights" below.)

---

## Diffusers / Qwen-Image VAE — Apache License 2.0

Upstream: https://github.com/huggingface/diffusers

`src/fizgig/krea2/vae.py` is copied and modified from the Diffusers
`AutoencoderKLQwenImage` implementation.
Copyright 2025 The Qwen-Image Team, Wan Team, and The HuggingFace Team.
All rights reserved. Licensed under the Apache License, Version 2.0.

---

## Note on model weights

The third-party notices above cover **source code** only. Krea 2 / FLUX.2 model
weights, the Qwen-Image VAE weights, and the Qwen3-VL text-encoder weights are
distributed by their respective publishers under their own model licenses, which
the user accepts when downloading them. Fizgig does not redistribute any model
weights.
