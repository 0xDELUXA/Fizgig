"""Krea 2 weight-only LoRA profiler.

No pipeline / no activation probes (the block-role map for Krea 2 doesn't exist yet, so there
are no semantic buckets to weight against). Just the static per-block weight signature:
for each block, sum ||lora_up||_F * ||lora_down||_F over its modules — a cheap, honest read of
where the LoRA put its weight. This is the instrument for *discovering* Krea 2's block roles.

Emits a flat HTML report (per-block bars, ranked by depth) + a `<name>.json` sidecar matching
the Klein profiler's schema so the Repair Studio cross-link works for Krea 2 too.
"""

import datetime
import json
import os
import re
from typing import Optional

logger = __import__("logging").getLogger(__name__)

_MAIN = re.compile(r"lora_unet_blocks_(\d+)_")
_TXT = re.compile(r"txtfusion_(layerwise|refiner)_blocks_(\d+)_")


def _krea2_block_of(lora_name: str) -> str:
    """Map a LoRA module name to a Krea 2 block id, or 'io' for the single I/O linears
    (first / last / tmlp / tproj / txtmlp / projector) that aren't transformer blocks."""
    t = _TXT.search(lora_name)
    if t:
        return f"txt_{'lw' if t.group(1) == 'layerwise' else 'rf'}_{int(t.group(2))}"
    m = _MAIN.search(lora_name)
    if m:
        return f"block_{int(m.group(1))}"
    return "io"


def _block_sort_key(bid: str):
    """Order for the report: main blocks 0..27, then txtfusion, then io."""
    if bid.startswith("block_"):
        return (0, int(bid.split("_")[1]))
    if bid.startswith("txt_"):
        kind, n = bid.split("_")[1], int(bid.split("_")[2])
        return (1, (0 if kind == "lw" else 1) * 100 + n)
    return (2, 0)


def _block_label(bid: str) -> str:
    if bid.startswith("block_"):
        return f"Block {bid.split('_')[1]}"
    if bid.startswith("txt_"):
        _, kind, n = bid.split("_")
        return f"Txt {kind.upper()} {n}"
    return "I/O linears"


def profile_krea2_weight_only(lora_path: str, output_html: str,
                              profiles_dir: Optional[str] = None):
    """Compute per-block weight norms, write the HTML report + sidecar. Returns (html, sidecar)."""
    from safetensors.torch import load_file
    from fizgig.networks.lora import ensure_kohya_lora_state_dict
    from fizgig.profiler.visualize import compute_lora_hash

    sd = ensure_kohya_lora_state_dict(load_file(lora_path))
    block_norms = {}
    n_modules = 0
    for key in list(sd.keys()):
        if not key.endswith(".lora_down.weight"):
            continue
        name = key[: -len(".lora_down.weight")]
        up = sd.get(name + ".lora_up.weight")
        if up is None:
            continue
        norm = float(up.float().norm().item()) * float(sd[key].float().norm().item())
        block_norms[_krea2_block_of(name)] = block_norms.get(_krea2_block_of(name), 0.0) + norm
        n_modules += 1

    if not block_norms:
        raise RuntimeError("No standard lora_down/up modules found — is this a krea2 LoRA?")

    total = sum(block_norms.values()) or 1.0
    by_depth = sorted(block_norms.items(), key=lambda kv: _block_sort_key(kv[0]))
    ranked = sorted(block_norms.items(), key=lambda kv: kv[1], reverse=True)
    lora_hash = compute_lora_hash(lora_path)

    html = _build_html(os.path.basename(lora_path), by_depth, ranked, total, n_modules, lora_hash)
    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    sidecar = output_html.rsplit(".", 1)[0] + ".json"
    payload = {
        "version": 1,
        "hash": lora_hash,
        "lora_path": lora_path,
        "lora_name": os.path.basename(lora_path),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "architecture": "krea2",
        "profile_kind": "weight-only",
        "html_report": os.path.basename(output_html),
        # Reuse the Klein cross-link schema: top blocks by weight norm, no bucket category yet.
        "top_active_blocks": [{"name": b, "category": "", "pct": round(100 * n / total, 1)}
                              for b, n in ranked[:8]],
        "top_static_blocks": [{"name": b, "norm": round(n, 5)} for b, n in ranked[:8]],
    }
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        logger.exception("Failed to write krea2 profile sidecar")
    return output_html, sidecar


def _build_html(name, by_depth, ranked, total, n_modules, lora_hash) -> str:
    hash_meta = f'<meta name="fizgig-lora-hash" content="{lora_hash}">' if lora_hash else ""
    maxn = max((n for _, n in by_depth), default=1.0) or 1.0
    rows = []
    for bid, n in by_depth:
        pct = 100 * n / total
        w = max(1, round(100 * n / maxn))
        rows.append(
            f'<div class="row"><div class="lbl">{_block_label(bid)}</div>'
            f'<div class="barwrap"><div class="bar" style="width:{w}%"></div></div>'
            f'<div class="val">{pct:.1f}%</div></div>')
    ranked_rows = "".join(
        f"<tr><td>{i+1}</td><td>{_block_label(b)}</td><td>{100*n/total:.1f}%</td>"
        f"<td>{n:.4f}</td></tr>" for i, (b, n) in enumerate(ranked))
    return f"""<!doctype html><html><head><meta charset="utf-8">{hash_meta}
<title>Krea 2 weight profile — {name}</title><style>
body{{font-family:Segoe UI,system-ui,sans-serif;background:#1b1d23;color:#e6e6e6;margin:0;padding:28px;}}
h1{{font-size:20px;margin:0 0 4px;}} .sub{{color:#9aa0aa;font-size:13px;margin-bottom:20px;}}
.card{{background:#23262e;border:1px solid #333;border-radius:10px;padding:18px 20px;margin-bottom:18px;}}
.row{{display:flex;align-items:center;gap:10px;margin:3px 0;}}
.lbl{{width:120px;font-size:12px;color:#cfd3da;}} .val{{width:54px;text-align:right;font-size:12px;color:#9aa0aa;}}
.barwrap{{flex:1;background:#2c3038;border-radius:4px;height:14px;overflow:hidden;}}
.bar{{height:100%;background:linear-gradient(90deg,#4a90d9,#6db3f2);}}
table{{width:100%;border-collapse:collapse;font-size:13px;}} td,th{{padding:5px 8px;text-align:left;border-bottom:1px solid #2c3038;}}
th{{color:#9aa0aa;font-weight:600;}} .note{{color:#9aa0aa;font-size:12px;margin-top:10px;}}
</style></head><body>
<h1>Krea 2 LoRA weight profile</h1>
<div class="sub">{name} &middot; {n_modules} modules &middot; weight-only (no semantic block map yet)</div>
<div class="card"><h3 style="margin:0 0 12px">Per-block weight signature (by depth)</h3>{''.join(rows)}
<div class="note">||lora_up||&middot;||lora_down|| summed per block, as % of the LoRA's total. Krea 2 block
roles aren't mapped yet, so this is shown flat (no Identity/Style/Details buckets) — it's the data to map them.</div></div>
<div class="card"><h3 style="margin:0 0 12px">Top blocks by weight</h3>
<table><tr><th>#</th><th>Block</th><th>Share</th><th>Norm</th></tr>{ranked_rows}</table></div>
</body></html>"""
