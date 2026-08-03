"""MiniMax H3 text encoder: Qwen3-VL-32B language model (truncated to 50 layers).

The DiT is conditioned on the **unnormalized hidden state after language layer 50** — the
last-layer output of the truncated checkpoint, with NO final norm (comfy applies none). We
build a transformers Qwen3Model (its layout matches the checkpoint's `model.layers.*` 1:1),
strip the `model.` prefix, skip the vision tower (`visual.*` — unused for text-only captions),
NF4-quantize the big linears so the ~32B fits a 24-32 GB card, and replace the final norm with
Identity so last_hidden_state IS the raw layer-50 output.

Image LoRA training uses text-only captions (no vision blocks), tokenized raw with NO special
tokens — the H3 convention. The tokenizer is the Qwen3-VL one Fizgig already bundles
(src/fizgig/assets/qwen3vl_tokenizer — shared vocab across the 4B/32B sizes).

Output: [1, L, 5120] bf16, exactly what the DiT's condition_proj expects.
"""

import os

import torch
import torch.nn as nn

# The official safetensors safe_open(device="cpu") + get_tensor path memory-maps the file and
# slices the torch storage per tensor — on Windows, reading a 48 GB file that way hard-crashes
# (access violation, exit 0xC0000005) deep in torch.storage.__getitem__. The repo's own
# MemoryEfficientSafeOpen reads each tensor with a plain np.fromfile (no torch mmap-view), which
# is exactly why every large-model loader (krea2 / klein) uses it. Use it here too.
from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen

# Derived from the checkpoint tensor shapes (U8 weights are 4-bit-packed: real in-dim = 2x).
_QWEN3_32B_TRUNC50 = dict(
    hidden_size=5120, num_hidden_layers=50, num_attention_heads=64, num_key_value_heads=8,
    head_dim=128, intermediate_size=25600, vocab_size=151936, max_position_embeddings=262144,
    rms_norm_eps=1e-6, rope_theta=5000000.0, attention_bias=False, tie_word_embeddings=False,
)

# The Qwen3 decoder Linears to NF4 (the matmul bulk). embed_tokens stays as-is (int8 in the
# checkpoint / bf16 after load); the tiny q/k norms + layernorms stay bf16.
_NF4_SUFFIXES = (".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                 ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
                 ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight")


def _bundled_tokenizer_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "qwen3vl_tokenizer")


def build_qwen3_te(config_overrides=None):
    """A Qwen3Model text encoder with no final norm (returns the raw layer-50 output)."""
    from transformers import Qwen3Config, Qwen3Model
    cfg = dict(_QWEN3_32B_TRUNC50)
    if config_overrides:
        cfg.update(config_overrides)
    model = Qwen3Model(Qwen3Config(**cfg))
    model.norm = nn.Identity()          # comfy applies NO final norm to the layer-50 conditioning
    return model


class MiniMaxH3TextEncoder:
    """Loads the bf16 TE, NF4 on GPU, and encodes captions to [1, L, 5120] bf16."""

    def __init__(self, model, tokenizer, device="cuda", compute_dtype=torch.bfloat16):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.compute_dtype = compute_dtype

    @torch.no_grad()
    def encode(self, caption: str, max_length: int = 512) -> torch.Tensor:
        # H3: raw prompt text, NO special tokens (no chat template).
        ids = self.tokenizer(caption, add_special_tokens=False, return_tensors="pt",
                             truncation=True, max_length=max_length)["input_ids"].to(self.device)
        if ids.shape[1] == 0:                              # empty caption -> single pad token
            ids = torch.tensor([[self.tokenizer.pad_token_id or 151643]], device=self.device)
        out = self.model(input_ids=ids)                    # norm=Identity -> raw layer-50 output
        return out.last_hidden_state.to(self.compute_dtype)   # [1, L, 5120]


def load_minimax_h3_te(path: str, device="cuda", compute_dtype=torch.bfloat16,
                       quantize=True, tokenizer_dir=None) -> MiniMaxH3TextEncoder:
    """Build + NF4-load the Qwen3-VL-32B language TE from the bf16 checkpoint (visual.* skipped)."""
    from bitsandbytes.nn import Linear4bit, Params4bit
    from transformers import AutoTokenizer

    with torch.device("meta"):
        model = build_qwen3_te()

    if quantize:
        for mod_name, module in list(model.named_modules()):
            for child_name, child in list(module.named_children()):
                full = f"{mod_name}.{child_name}" if mod_name else child_name
                if isinstance(child, nn.Linear) and (full + ".weight").endswith(_NF4_SUFFIXES):
                    q = Linear4bit(child.in_features, child.out_features, bias=child.bias is not None,
                                   compute_dtype=compute_dtype, quant_type="nf4")
                    setattr(module, child_name, q)

    dev = torch.device(device)
    model_keys = {n for n, _ in model.named_parameters()}
    with MemoryEfficientSafeOpen(path) as f:
        ckpt = set(f.keys())
        for name in model_keys:
            src = "model." + name                          # checkpoint prefixes the LM with model.
            if src not in ckpt:
                continue                                   # e.g. norm (Identity) has no params
            w = f.get_tensor(src)
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            leaf = name.rsplit(".", 1)[1]
            if quantize and (name.endswith(_NF4_SUFFIXES)):
                p = Params4bit(w.to(compute_dtype), requires_grad=False, quant_type="nf4").to(dev)
                setattr(parent, leaf, p)
            else:
                keep = w.to(torch.float32) if w.dtype == torch.float32 else w.to(compute_dtype)
                setattr(parent, leaf, nn.Parameter(keep.to(dev), requires_grad=False))

    # Computed (non-checkpoint) buffers stayed on meta from the meta-build. The rotary
    # embedding's inv_freq is the load-bearing one — rebuild it on the real device. A general
    # sweep materializes any other stray meta buffers to be safe.
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    model.rotary_emb = Qwen3RotaryEmbedding(model.config).to(dev)
    for mod in model.modules():
        for bname, buf in list(mod.named_buffers(recurse=False)):
            if buf is not None and buf.is_meta:
                mod.register_buffer(bname, torch.zeros(buf.shape, dtype=buf.dtype, device=dev))
    model.requires_grad_(False)

    tok = AutoTokenizer.from_pretrained(tokenizer_dir or _bundled_tokenizer_dir())
    return MiniMaxH3TextEncoder(model, tok, device=device, compute_dtype=compute_dtype)
