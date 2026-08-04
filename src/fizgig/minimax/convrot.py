"""ConvRot int8 decode — reads ComfyUI's pre-quantized MiniMax H3 checkpoints.

`minimax_h3_*_pruned_int8_convrot.safetensors` stores every big block Linear as int8 in a
ROTATED basis, marked by a `<module>.comfy_quant` blob:

    {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}

ConvRot (arXiv:2512.03673) folds a block regular-Hadamard rotation into the weight offline and
applies the same rotation to the activation at runtime; being orthogonal it cancels in the
matmul, and it spreads outliers so one scale per output row is safe (the SmoothQuant failure
mode is what it avoids). Storage is therefore:

    weight        int8   [out, in]   codes of  rotate(W, G)
    weight_scale  fp32   [out, 1]    per-output-row symmetric scale
    comfy_quant   uint8  [n]         the JSON above

Fizgig doesn't run an int8 kernel — it NF4s the base and trains LoRA on top — so all we need is
the inverse: undo the rotation and get W back. The regular Hadamard is symmetric and orthogonal,
i.e. its own inverse, so "unrotate" is the same operation as "rotate".

Verified against the bf16 release of the same model: 0.9% relative error per tensor (the ~1%
the method claims), vs cosine 0.06 if the rotation is skipped — the failure is loud, not subtle.
"""

import json

import torch

_hadamard_cache = {}


def regular_hadamard(rot_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Kronecker powers of the 4x4 regular Hadamard, orthonormal — symmetric and orthogonal,
    so it is its own inverse.

    Built in fp32 on purpose: entries stay exactly +-1 through the krons and rot_size is a power
    of 4, so dividing by its (power-of-two) root is exact."""
    key = (rot_size, str(device), dtype)
    if key not in _hadamard_cache:
        r4 = torch.tensor([[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                          dtype=torch.float32)
        h = r4.clone()
        while h.shape[0] < rot_size:
            h = torch.kron(h, r4)
        if h.shape[0] != rot_size:
            raise ValueError(f"convrot group size {rot_size} is not a power of 4")
        _hadamard_cache[key] = (h / rot_size ** 0.5).to(device=device, dtype=dtype)
    return _hadamard_cache[key]


def rotate(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Block regular-Hadamard rotation along the LAST dim. Self-inverse, so this both applies
    and undoes it."""
    if rot_size <= 1:
        return x
    if x.shape[-1] % rot_size:
        raise ValueError(f"last dim {x.shape[-1]} is not a multiple of the rotation block {rot_size}")
    h = regular_hadamard(rot_size, x.device, x.dtype)
    return torch.matmul(x.reshape(-1, x.shape[-1] // rot_size, rot_size), h).reshape(x.shape)


def parse_comfy_quant(blob: torch.Tensor) -> dict:
    """The `<module>.comfy_quant` marker: a uint8 tensor holding JSON."""
    return json.loads(bytes(blob.to(torch.uint8).cpu().numpy().tobytes()).decode("utf-8"))


def dequantize_int8_convrot(qweight: torch.Tensor, scale: torch.Tensor, conf: dict,
                            out_dtype=torch.bfloat16) -> torch.Tensor:
    """int8 codes + per-row scale (+ the marker's config) -> the dense weight.

    scale arrives as [out, 1] (or [out]); the rotation is undone in fp32 and the result cast
    once, so no intermediate is held at full precision longer than one tensor."""
    fmt = conf.get("format")
    if fmt != "int8_tensorwise":
        raise ValueError(f"unsupported comfy quant format {fmt!r} (this decodes int8_tensorwise)")
    group = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1
    w = qweight.to(torch.float32) * scale.to(torch.float32).reshape(-1, 1)
    return rotate(w, group).to(out_dtype)
