"""Load the real MiniMax H3 bf16 checkpoint into MiniMaxH3DiT, NF4-quantizing the big block
linears so the ~33B model fits a 24-32 GB card.

Strategy (never holds 66 GB whole): build the model on `meta`, then for each parameter read
just that tensor from the safetensors file and place it —
  * big block/refiner Linear weights (attn qkv/out, mlp fc1/fc2)  -> NF4 (bitsandbytes)
  * everything else (norms, adaln, patch/condition/time proj, heads, rope) -> bf16 on GPU
The frozen base is what LoRA/FT trains on top of; the fp32 output-head island stays fp32.
"""

import torch
import torch.nn as nn

# Not the official safe_open mmap path: on Windows, streaming a 66 GB file via
# safe_open(device="cpu").get_tensor() hard-crashes (access violation) in torch's storage
# mmap-slicing. MemoryEfficientSafeOpen reads each tensor with a plain np.fromfile instead —
# the same reader every other large-model loader in the repo uses. See embedder.py.
from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen

from .model import MiniMaxH3DiT, MiniMaxH3Config

# Which Linear weights get NF4'd: the per-block matmul bulk AND the per-block AdaLN
# modulation projection (adaln_proj.linear is [96768, 2688] — ~0.5 GB/block bf16, ~26 GB
# across 50 blocks, the single biggest VRAM sink; it's ~40% of the model, the "overstated"
# modulation weight mass). All are frozen base weights, so NF4 is fine for LoRA/FT on top.
_NF4_SUBSTRINGS = (".attn.qkv_proj.weight", ".attn.out_proj.weight",
                   ".mlp.fc1.weight", ".mlp.fc2.weight", ".adaln_proj.linear.weight")


def _is_nf4_target(name: str) -> bool:
    return name.startswith(("blocks.", "token_refiner.blocks.")) and name.endswith(_NF4_SUBSTRINGS)


def load_minimax_h3_dit(path: str, device="cuda", compute_dtype=torch.bfloat16,
                        quantize=True) -> MiniMaxH3DiT:
    """Return a MiniMaxH3DiT with the real weights loaded; block matmul linears NF4 (frozen)."""
    from bitsandbytes.nn import Linear4bit, Params4bit

    with torch.device("meta"):
        model = MiniMaxH3DiT(MiniMaxH3Config())

        # Swap the NF4-target Linears for Linear4bit shells — INSIDE the meta context. Outside it,
        # each Linear4bit constructor eagerly allocates a full fp32 weight on CPU (nn.Linear's
        # default), which across the 33B of NF4-target Linears is ~118 GB of throwaway tensors
        # that the process allocator then holds for the whole run. On meta the shells are 0 bytes;
        # the real weights are streamed in below.
        if quantize:
            for mod_name, module in list(model.named_modules()):
                for child_name, child in list(module.named_children()):
                    full = f"{mod_name}.{child_name}" if mod_name else child_name
                    if isinstance(child, nn.Linear) and _is_nf4_target(f"{full}.weight"):
                        q = Linear4bit(child.in_features, child.out_features, bias=child.bias is not None,
                                       compute_dtype=compute_dtype, quant_type="nf4")
                        setattr(module, child_name, q)

    dev = torch.device(device)
    with MemoryEfficientSafeOpen(path) as f:
        keys = set(f.keys())
        for name, param in model.named_parameters():
            if name not in keys:
                continue
            w = f.get_tensor(name)
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            leaf = name.rsplit(".", 1)[1]
            if quantize and _is_nf4_target(name):
                # NF4-quantize this weight onto the GPU; frozen (no grad).
                # NF4 quantization happens on the .to(cuda) move (Params4bit.cuda()).
                p = Params4bit(w.to(compute_dtype), requires_grad=False, quant_type="nf4").to(dev)
                setattr(parent, leaf, p)
            else:
                keep = w.to(torch.float32) if w.dtype == torch.float32 else w.to(compute_dtype)
                setattr(parent, leaf, nn.Parameter(keep.to(dev), requires_grad=False))
        # buffers (rope.inv_freq)
        for name, _ in model.named_buffers():
            if name in keys:
                parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
                leaf = name.rsplit(".", 1)[1]
                parent.register_buffer(leaf, f.get_tensor(name).to(dev))
    model.eval()
    return model
