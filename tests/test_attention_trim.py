"""Sequence trimming must behave exactly as it did when the decision lived inside attention().

The trim decision moved into AttentionParams.__post_init__ because it reads a CUDA tensor on the
CPU, and running it per block meant a device sync in every one of the 28 blocks (56 with gradient
checkpointing) plus a graph break for torch.compile. This checks the behaviour is unchanged:
uniform sequence lengths trim, ragged ones do not, and split-attn never trims.
"""
import sys

sys.path.insert(0, r"W:\Peter\Documents\Development\Fizgig\src")
import torch

from fizgig.krea2.attention import AttentionParams as K2Params, attention as k2_attention
from fizgig.modules.attention import AttentionParams as KleinParams, attention as klein_attention

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, L, H, D, IMG = 2, 16, 8, 64, 32

for name, Params, attention, from_mask in (
        ("krea2", K2Params, k2_attention, "create_attention_params_from_mask"),
        ("klein", KleinParams, klein_attention, "create_from_mask")):
    mk = getattr(Params, from_mask)

    uniform = mk("torch", False, IMG, torch.ones(B, L, device=DEV))
    assert uniform.uniform_seqlen == L + IMG, uniform.uniform_seqlen

    ragged = mk("torch", False, IMG, torch.tensor([[1.] * L, [1.] * 8 + [0.] * 8], device=DEV))
    assert ragged.uniform_seqlen is None, ragged.uniform_seqlen

    split = mk("torch", True, IMG, torch.ones(B, L, device=DEV))
    assert split.uniform_seqlen is None, "split-attn has its own per-sequence path"

    # flash/sageattn take the mask natively and must not be trimmed out from under them.
    assert mk("flash", False, IMG, torch.ones(B, L, device=DEV)).uniform_seqlen is None

    # A full-length mask trims to the full length, so the result must match the no-mask path.
    q = torch.randn(B, L + IMG, H, D, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        trimmed = attention(q, q, q, attn_params=uniform)
        plain = attention(q, q, q, attn_params=Params.create_attention_params("torch", False)
                          if name == "krea2" else Params.create("torch", False))
    assert trimmed.shape == plain.shape, (trimmed.shape, plain.shape)
    assert torch.allclose(trimmed.float(), plain.float(), atol=1e-2)
    print(f"  {name}: uniform trims to {uniform.uniform_seqlen}, ragged/split/flash do not, "
          f"output matches the untrimmed path")
