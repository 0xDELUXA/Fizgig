"""Load a real MiniMax H3 checkpoint into MiniMaxH3DiT, NF4-quantizing the big block linears.

Two checkpoint variants load through the same path (auto-detected from the file's own keys):

  * **bf16 full** (`minimax_h3_fl2va_bf16.safetensors`, 66 GB) — the released weights. ~33 B
    params, of which the full-width AdaLN modulation ([96768, 2688] x50) is ~40%.
  * **pruned int8 ConvRot** (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, 21 GB) — what
    ComfyUI ships and what users actually run at inference. The AdaLN MLP is replaced by a
    sampled curve table (`adaln_t_table`, so `adaln_proj.linear` is [96768, **8**]) and the 200
    block matmul Linears are stored as int8 in a rotated basis. Training this one means the LoRA
    trains against the same weights it will be deployed on, and AdaLN becomes a sane LoRA target
    instead of an unusable one. See convrot.py for the decode.

Strategy (never holds the file whole): build the model on `meta`, then per parameter read just
that tensor, decode it if quantized, and place it —
  * big block/refiner Linear weights (attn qkv/out, mlp fc1/fc2, adaln)  -> NF4 (bitsandbytes)
  * everything else (norms, patch/condition/time proj, heads, rope, table) -> bf16 on GPU
The frozen base is what LoRA/FT trains on top of; the fp32 output-head island stays fp32.
"""

import torch
import torch.nn as nn

# Not the official safe_open mmap path: on Windows, streaming a 66 GB file via
# safe_open(device="cpu").get_tensor() hard-crashes (access violation) in torch's storage
# mmap-slicing. MemoryEfficientSafeOpen reads each tensor with a plain np.fromfile instead —
# the same reader every other large-model loader in the repo uses. See embedder.py.
from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen

from .convrot import dequantize_int8_convrot, parse_comfy_quant
from .model import MiniMaxH3DiT, MiniMaxH3Config

# Which Linear weights get NF4'd: the per-block matmul bulk AND the per-block AdaLN modulation
# projection. On the bf16 model adaln_proj.linear is [96768, 2688] — ~0.5 GB/block, the single
# biggest VRAM sink; on the pruned model it is [96768, 8] and NF4 barely matters either way.
# All are frozen base weights, so NF4 is fine for LoRA/FT on top.
_NF4_SUBSTRINGS = (".attn.qkv_proj.weight", ".attn.out_proj.weight",
                   ".mlp.fc1.weight", ".mlp.fc2.weight", ".adaln_proj.linear.weight")


def _is_nf4_target(name: str) -> bool:
    return name.startswith(("blocks.", "token_refiner.blocks.")) and name.endswith(_NF4_SUBSTRINGS)


def _owner_and_leaf(model, name: str):
    """(module that holds `name`, attribute name) — for a TOP-LEVEL entry the owner is the
    model itself. The pruned checkpoint's `adaln_t_table` is exactly that case: every other
    tensor in either file is nested, so a naive rsplit('.') works everywhere else and fails
    only here."""
    parent_path, _, leaf = name.rpartition(".")
    return (model.get_submodule(parent_path) if parent_path else model), leaf


def config_from_checkpoint(keys, table_shape=None) -> MiniMaxH3Config:
    """Build the config the file actually describes.

    The only structural difference between the two releases is the timestep path: a pruned file
    carries `adaln_t_table` and no `time_embedder.*`."""
    if "adaln_t_table" in keys:
        if table_shape is None:
            raise ValueError("pruned checkpoint: adaln_t_table shape is required")
        size, dim = int(table_shape[0]), int(table_shape[1])
        return MiniMaxH3Config(adaln_t_table_size=size, time_embed_dim=dim)
    return MiniMaxH3Config()


def load_minimax_h3_dit(path: str, device="cuda", compute_dtype=torch.bfloat16,
                        quantize=True, blocks_to_swap: int = 0) -> MiniMaxH3DiT:
    """Return a MiniMaxH3DiT with the real weights loaded; block matmul linears NF4 (frozen).
    blocks_to_swap parks the LAST n blocks on CPU at load (see the loop below) — pair it with
    model.enable_block_swap(n) so the forward moves them just-in-time."""
    from bitsandbytes.nn import Linear4bit, Params4bit

    with MemoryEfficientSafeOpen(path) as f:
        keys = set(f.keys())
        table_shape = None
        if "adaln_t_table" in keys:
            table_shape = tuple(f.get_tensor("adaln_t_table").shape)
        cfg = config_from_checkpoint(keys, table_shape)
        # `<module>.comfy_quant` marks a pre-quantized Linear; the config rides in the blob.
        quant_conf = {k[: -len(".comfy_quant")]: parse_comfy_quant(f.get_tensor(k))
                      for k in keys if k.endswith(".comfy_quant")}

        with torch.device("meta"):
            model = MiniMaxH3DiT(cfg)

            # Swap the NF4-target Linears for Linear4bit shells — INSIDE the meta context.
            # Outside it, each Linear4bit constructor eagerly allocates a full fp32 weight on CPU
            # (nn.Linear's default), which across the 33 B of NF4-target Linears is ~118 GB of
            # throwaway tensors that the process allocator then holds for the whole run. On meta
            # the shells are 0 bytes; the real weights are streamed in below.
            if quantize:
                for mod_name, module in list(model.named_modules()):
                    for child_name, child in list(module.named_children()):
                        full = f"{mod_name}.{child_name}" if mod_name else child_name
                        if isinstance(child, nn.Linear) and _is_nf4_target(f"{full}.weight"):
                            q = Linear4bit(child.in_features, child.out_features,
                                           bias=child.bias is not None,
                                           compute_dtype=compute_dtype, quant_type="nf4")
                            setattr(module, child_name, q)

        dev = torch.device(device)
        if quant_conf:
            print(f"[load] streaming the pre-quantized MiniMax H3 base ({len(quant_conf)} int8 "
                  "ConvRot linears decoded to bf16, then NF4) — a quiet minute here is normal "
                  "(as is a bitsandbytes 'expandable_segments not supported' warning on Windows).",
                  flush=True)
        else:
            print("[load] streaming the 66 GB MiniMax H3 base and quantizing to NF4 — a couple of "
                  "quiet minutes here is normal (as is a bitsandbytes 'expandable_segments not "
                  "supported' warning on Windows).", flush=True)

        # Block swap: params of the LAST n blocks are parked on CPU at load time — quantized on
        # the GPU (Params4bit quantizes on the .to(cuda) move) then immediately moved off, so the
        # packed weights never accumulate. Loading everything resident first and parking
        # afterwards would transiently need the full ~17 GB, OOMing exactly the cards that asked
        # for swap.
        n_layers = len(model.blocks)
        swap_from = n_layers - max(0, min(int(blocks_to_swap or 0), n_layers - 2))

        def _parked(name: str) -> bool:
            if not name.startswith("blocks."):
                return False
            return int(name.split(".")[1]) >= swap_from

        def _read_weight(name: str):
            """The dense tensor for `name`, decoding a pre-quantized Linear if that's what it is."""
            module_path = name[: -len(".weight")]
            conf = quant_conf.get(module_path) if name.endswith(".weight") else None
            if conf is None:
                return f.get_tensor(name)
            return dequantize_int8_convrot(f.get_tensor(name),
                                           f.get_tensor(f"{module_path}.weight_scale"),
                                           conf, out_dtype=compute_dtype)

        for name, param in model.named_parameters():
            if name not in keys:
                continue
            w = _read_weight(name)
            parent, leaf = _owner_and_leaf(model, name)
            if quantize and _is_nf4_target(name):
                # NF4-quantize this weight onto the GPU; frozen (no grad).
                # NF4 quantization happens on the .to(cuda) move (Params4bit.cuda()).
                p = Params4bit(w.to(compute_dtype), requires_grad=False, quant_type="nf4").to(dev)
                if _parked(name):
                    p = p.to("cpu")            # stays packed (uint8) + keeps quant_state
                setattr(parent, leaf, p)
            else:
                keep = w.to(torch.float32) if w.dtype == torch.float32 else w.to(compute_dtype)
                target = torch.device("cpu") if _parked(name) else dev
                setattr(parent, leaf, nn.Parameter(keep.to(target), requires_grad=False))
        # buffers (rope.inv_freq, and adaln_t_table on a pruned file)
        for name, _ in model.named_buffers():
            if name in keys:
                parent, leaf = _owner_and_leaf(model, name)
                parent.register_buffer(leaf, f.get_tensor(name).to(torch.float32).to(dev))
    model.eval()
    return model
