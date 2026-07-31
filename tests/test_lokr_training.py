"""Trainable LoKR (Phase 1): module math, init, and the save->reload round-trip.

Everything here runs on CPU with toy dims — the point is exactness against the dense
kron reference and compatibility with the loaders the rest of the app already uses,
not speed. GPU behaviour is covered by the smoke training run in Phase 6.
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from fizgig.networks.lora import (LoKRModule, factorization, create_network,  # noqa: E402
                                  create_network_from_weights, detect_lora_format,
                                  ensure_kohya_lora_state_dict)

FAILS = []


def ck(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


torch.manual_seed(0)

# --- 1. factorization ---------------------------------------------------------------------
ck("factorization: exact split at the factor", factorization(6144, 8) == (8, 768))
ck("  factor larger than sqrt clamps to a divisor", factorization(64, 8) == (8, 8))
ck("  non-divisor factor takes the largest divisor below", factorization(24, 5) == (4, 6))
ck("  prime dims degenerate to (1, n), not an error", factorization(13, 8) == (1, 13))
ck("  factor 1", factorization(100, 1) == (1, 100))
for n, f in ((6144, 8), (24576, 16), (30, 4), (7, 3)):
    a, b = factorization(n, f)
    ck(f"  product invariant {n}/{f}", a * b == n and a <= f, (a, b))

# --- 2. LoKRModule math -------------------------------------------------------------------
lin = torch.nn.Linear(24, 16, bias=False)
mod = LoKRModule("lora_unet_toy_lin", lin, multiplier=1.0, lora_dim=4, alpha=1, factor=4)

ck("w1/w2 shapes obey a*c==out, b*d==in",
   mod.a * mod.c == 16 and mod.b * mod.d == 24, (mod.a, mod.b, mod.c, mod.d))
ck("alpha buffer is 1.0 and scale is 1.0",
   float(mod.alpha) == 1.0 and mod.scale == 1.0)
ck("delta is exactly zero at init (w2 zeroed)",
   torch.all(mod.lokr_w2 == 0) and not torch.all(mod.lokr_w1 == 0))

x = torch.randn(3, 24)
base_out = lin(x)
mod.apply_to()
ck("apply_to removed org_module (frozen base stays out of state_dict)",
   not hasattr(mod, "org_module") and
   all("org_module" not in k for k in mod.state_dict().keys()), list(mod.state_dict().keys()))
ck("zero-init forward == base forward exactly", torch.equal(mod.forward(x), base_out))

# Give the factors real values and check against the dense kron reference.
with torch.no_grad():
    mod.lokr_w1.copy_(torch.randn_like(mod.lokr_w1))
    mod.lokr_w2.copy_(torch.randn_like(mod.lokr_w2))
mod.multiplier = 0.7
ref = base_out + 0.7 * (x @ torch.kron(mod.lokr_w1, mod.lokr_w2).T)
got = mod.forward(x)
ck("forward matches dense kron reference",
   torch.allclose(got, ref, atol=1e-5), f"max diff {(got - ref).abs().max():.2e}")

mod.enabled = False
ck("enabled=False returns pure base output", torch.equal(mod.forward(x), base_out))
mod.enabled = True
mod.multiplier = 0.0
ck("multiplier=0 returns pure base output", torch.equal(mod.forward(x), base_out))
mod.multiplier = 1.0

# Gradients: w2 learns immediately; w1 unlocks once w2 is nonzero (same staging as
# LoRA's zeroed lora_up — only one side has grad at the very first step).
lin2 = torch.nn.Linear(24, 16, bias=False)
m2 = LoKRModule("lora_unet_toy_lin2", lin2, 1.0, 4, 1, factor=4)
m2.apply_to()
m2.forward(torch.randn(2, 24)).sum().backward()
ck("step-0 grads: w2 nonzero, w1 zero (w2 is the zeroed factor)",
   m2.lokr_w2.grad is not None and m2.lokr_w2.grad.abs().sum() > 0
   and (m2.lokr_w1.grad is None or torch.all(m2.lokr_w1.grad == 0)))
with torch.no_grad():
    m2.lokr_w2.add_(torch.randn_like(m2.lokr_w2))
m2.zero_grad()
m2.forward(torch.randn(2, 24)).sum().backward()
ck("once w2 is nonzero both factors receive grads",
   m2.lokr_w1.grad.abs().sum() > 0 and m2.lokr_w2.grad.abs().sum() > 0)

# --- 3. network build -> save -> reload round-trip ----------------------------------------
class ToyDiT(torch.nn.Module):
    """Two 'blocks' of Linears with dotted paths, mimicking the DiT walk."""
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        for _ in range(2):
            blk = torch.nn.Module()
            blk.attn = torch.nn.Module()
            blk.attn.qkv = torch.nn.Linear(24, 48, bias=False)
            blk.attn.out = torch.nn.Linear(16, 24, bias=False)
            self.blocks.append(blk)

    def forward(self, x):  # unused; networks patch the Linears directly
        return x


dit = ToyDiT()
net = create_network(None, "lora_unet", 1.0, 4, 1.0, None, [], dit,
                     module_class=LoKRModule, module_kwargs={"factor": 4})
net.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
ck("network built one LoKR module per Linear", len(net.unet_loras) == 4,
   [m.lora_name for m in net.unet_loras])
ck("  all modules are LoKRModule", all(isinstance(m, LoKRModule) for m in net.unet_loras))

# Real (nonzero) weights so the round-trip comparison is meaningful.
with torch.no_grad():
    for m in net.unet_loras:
        m.lokr_w1.copy_(torch.randn_like(m.lokr_w1))
        m.lokr_w2.copy_(torch.randn_like(m.lokr_w2))

with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "toy_lokr.safetensors")
    net.save_weights(p, torch.float32, {"ss_test": "1"})
    sd = load_file(p)

    ck("saved keys are native lokr suffixes",
       any(k.endswith(".lokr_w1") for k in sd) and any(k.endswith(".lokr_w2") for k in sd)
       and any(k.endswith(".alpha") for k in sd), sorted(sd.keys())[:4])
    ck("  no lora_up/lora_down keys",
       not any("lora_up" in k or "lora_down" in k for k in sd))
    ck("  detect_lora_format says lokr", detect_lora_format(sd) == "lokr")
    ck("  ensure_kohya passes native lokr through unchanged",
       ensure_kohya_lora_state_dict(dict(sd)).keys() == sd.keys())

    # Reload path — the exact chain previews and the context LoRA use.
    dit2 = ToyDiT()
    x = torch.randn(3, 24)
    ref_deltas = {}
    for m in net.unet_loras:
        w = torch.kron(m.lokr_w1, m.lokr_w2)
        ref_deltas[m.lora_name] = w

    inf_net = create_network_from_weights(None, 1.0, dict(sd), None, dit2, for_inference=True)
    inf_net.apply_to(text_encoders=None, unet=dit2, apply_text_encoder=False, apply_unet=True)
    missing = inf_net.load_state_dict(dict(sd), strict=False)
    ck("reloaded via create_network_from_weights: 4 inf modules",
       len(inf_net.unet_loras) == 4, [m.lora_name for m in inf_net.unet_loras])
    ok = True
    for m in inf_net.unet_loras:
        w_inf = m._w1() if hasattr(m, "_w1") else None
        kron_inf = torch.kron(m._w1(), m._w2()) * m.scale * m.multiplier
        if not torch.allclose(kron_inf, ref_deltas[m.lora_name], atol=1e-6):
            ok = False
    ck("  reloaded deltas match trained deltas to 1e-6 (scale round-trips)", ok)

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
