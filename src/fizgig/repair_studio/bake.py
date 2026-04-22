"""Bake a SliderState into a new safetensors LoRA.

Supports:
- Primary-only bake: disabled blocks dropped; enabled blocks rescaled by
  `primary_strength` (scale + mult baked into `lora_up`; output alpha = rank
  so new scale = 1.0, equivalent to the live preview's contribution).
- Donor-blended bake (standard LoRA + standard LoRA): rank-concatenation.
  For a block with both primary and donor enabled, the output module's
  `lora_up` is `cat([up_p * mp * scale_p, up_d * md * scale_d], dim=-1)` and
  `lora_down` is `cat([down_p, down_d], dim=0)`. Alpha set to the new rank so
  the file's effective scale is 1.0; the original primary/donor multipliers
  and alphas are fully baked in.
  Verification: `baked_up @ baked_down` equals the live inference sum
  `mp * scale_p * up_p @ down_p + md * scale_d * up_d @ down_d`.
- LyCORIS primary or donor: refused with `UnsupportedLoRAFormat`. Phase E will
  add SVD-merge fallback for LyCORIS-involved bakes.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

from fizgig.repair_studio.state import SliderState

logger = logging.getLogger(__name__)


_BLOCK_KEY_RE = re.compile(r"(?:lora_unet_)?(double_blocks|single_blocks)_(\d+)_")


def _block_id_from_key(key: str) -> Optional[str]:
    m = _BLOCK_KEY_RE.search(key)
    if m:
        kind = m.group(1).replace("_blocks", "")  # "double" / "single"
        return f"{kind}_{int(m.group(2))}"
    return None


def _group_by_module(sd: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """{module_name: {suffix: tensor, ...}} — splits state dict keys on the
    first dot after the lora_unet_... prefix."""
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in sd.items():
        if "." not in key:
            continue
        mod, _, suffix = key.partition(".")
        out.setdefault(mod, {})[suffix] = tensor
    return out


def _module_alpha(mod_keys: Dict[str, torch.Tensor], fallback_rank: int) -> float:
    """Extract alpha as a float; fallback to rank so scale = 1.0."""
    alpha = mod_keys.get("alpha")
    if alpha is None:
        return float(fallback_rank)
    try:
        return float(alpha.item())
    except Exception:
        return float(fallback_rank)


_ALPHA_SENTINEL_THRESHOLD = 1e8  # Comfy-Realtime-Lora sentinel: alpha > 1e8 means scale=1.0


def _materialize_lycoris_module(mod_keys: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
    """Convert a single LoKR or LoHa module to standard lora_up/lora_down via SVD.

    Returns a new dict with lora_up.weight, lora_down.weight, alpha — or None
    if the module isn't a recognised LyCORIS variant.
    """
    # --- LoKR ---
    if mod_keys.get("lokr_w1") is not None or mod_keys.get("lokr_w1_a") is not None:
        if mod_keys.get("lokr_w1_a") is not None:
            w1 = (mod_keys["lokr_w1_a"].float() @ mod_keys["lokr_w1_b"].float())
        else:
            w1 = mod_keys["lokr_w1"].float()
        if mod_keys.get("lokr_w2_a") is not None:
            w2 = (mod_keys["lokr_w2_a"].float() @ mod_keys["lokr_w2_b"].float())
        else:
            w2 = mod_keys["lokr_w2"].float()
        alpha_t = mod_keys.get("alpha")
        dim = w1.shape[0] * w2.shape[0]
        alpha = float(alpha_t.item()) if alpha_t is not None else float(dim)
        scale = 1.0 if alpha > _ALPHA_SENTINEL_THRESHOLD else alpha / max(dim, 1)
        W = torch.kron(w1, w2) * scale

    # --- LoHa ---
    elif mod_keys.get("hada_w1_a") is not None:
        W1 = mod_keys["hada_w1_a"].float() @ mod_keys["hada_w1_b"].float()
        W2 = mod_keys["hada_w2_a"].float() @ mod_keys["hada_w2_b"].float()
        alpha_t = mod_keys.get("alpha")
        dim = W1.shape[0]
        alpha = float(alpha_t.item()) if alpha_t is not None else float(dim)
        scale = 1.0 if alpha > _ALPHA_SENTINEL_THRESHOLD else alpha / max(dim, 1)
        W = (W1 * W2) * scale

    else:
        return None

    # SVD the dense delta to a standard LoRA at a reasonable rank
    # Use min(64, min_dim) — preserves most information without bloating
    min_dim = min(W.shape)
    target_rank = min(64, min_dim)

    try:
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    except Exception:
        return None

    R = min(target_rank, S.shape[0])
    sqrt_S = torch.sqrt(S[:R])
    lora_up = (U[:, :R] * sqrt_S.unsqueeze(0)).to(torch.float16)
    lora_down = (sqrt_S.unsqueeze(1) * Vt[:R, :]).to(torch.float16)

    return {
        "lora_up.weight": lora_up.contiguous(),
        "lora_down.weight": lora_down.contiguous(),
        "alpha": torch.tensor(float(R), dtype=torch.float16),
    }


def _materialize_lycoris_modules(modules: Dict[str, Dict[str, torch.Tensor]]) -> int:
    """In-place: replace any LoKR/LoHa modules with SVD-materialized standard LoRA.
    Returns the number of modules converted."""
    count = 0
    for mod_name, mod_keys in list(modules.items()):
        # Skip if already standard LoRA
        if mod_keys.get("lora_up.weight") is not None:
            continue
        result = _materialize_lycoris_module(mod_keys)
        if result is not None:
            modules[mod_name] = result
            count += 1
    return count


def _bake_single_contribution(
    mod_keys: Dict[str, torch.Tensor],
    multiplier: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Bake one contribution (primary OR donor) into up/down tensors with
    scale + multiplier absorbed into up. Returns (new_up, down, rank) or None
    if the module isn't a standard LoRA (no lora_up/lora_down keys)."""
    up = mod_keys.get("lora_up.weight")
    down = mod_keys.get("lora_down.weight")
    if up is None or down is None:
        return None
    rank = int(down.shape[0])  # lora_down is (rank, in_dim)
    alpha = _module_alpha(mod_keys, rank)
    scale = alpha / max(rank, 1)
    factor = multiplier * scale
    if factor == 1.0:
        new_up = up.clone()
    else:
        new_up = (up.to(torch.float32) * factor).to(up.dtype)
    return new_up, down.clone(), rank


def save_repaired_lora(
    primary_path: str,
    state: SliderState,
    out_path: str,
    donor_path: Optional[str] = None,
) -> dict:
    """Bake state into a new .safetensors. Returns a summary dict with
    dropped_blocks, rescaled_blocks, blended_blocks, keys_in, keys_out."""
    from fizgig.networks.lora import ensure_kohya_lora_state_dict, detect_lora_format, UnsupportedLoRAFormat

    if not os.path.isfile(primary_path):
        raise FileNotFoundError(primary_path)

    # Load primary.
    sd_p_raw: Dict[str, torch.Tensor] = load_file(primary_path)
    sd_p = ensure_kohya_lora_state_dict(sd_p_raw)
    fmt_p = detect_lora_format(sd_p)
    if fmt_p == "glora":
        raise UnsupportedLoRAFormat(
            f"Primary LoRA format is GLoRA (4-matrix form); not supported for bake."
        )

    # Metadata from the primary file.
    metadata: Dict[str, str] = {}
    try:
        from safetensors import safe_open
        with safe_open(primary_path, framework="pt") as f:
            md = f.metadata() or {}
            metadata = {str(k): str(v) for k, v in md.items()}
    except Exception:
        logger.exception("Could not read primary metadata; saving without it")

    # Load donor (optional).
    sd_d: Optional[Dict[str, torch.Tensor]] = None
    if donor_path and os.path.isfile(donor_path):
        sd_d_raw = load_file(donor_path)
        sd_d = ensure_kohya_lora_state_dict(sd_d_raw)
        fmt_d = detect_lora_format(sd_d)
        if fmt_d == "glora":
            raise UnsupportedLoRAFormat(
                f"Donor LoRA format is GLoRA (4-matrix form); not supported for bake."
            )

    modules_p = _group_by_module(sd_p)
    modules_d = _group_by_module(sd_d) if sd_d is not None else {}

    # Materialize any LyCORIS modules (LoKR/LoHa) to standard LoRA via SVD
    lycoris_converted = 0
    if fmt_p in ("lokr", "loha"):
        lycoris_converted += _materialize_lycoris_modules(modules_p)
        logger.info("Bake: materialized %d LyCORIS primary modules to standard LoRA via SVD", lycoris_converted)
    if sd_d is not None and fmt_d in ("lokr", "loha"):
        n = _materialize_lycoris_modules(modules_d)
        lycoris_converted += n
        logger.info("Bake: materialized %d LyCORIS donor modules to standard LoRA via SVD", n)

    all_modules = sorted(set(modules_p) | set(modules_d))

    sd_out: Dict[str, torch.Tensor] = {}
    dropped_blocks: set = set()
    rescaled_blocks: set = set()
    blended_blocks: set = set()
    combined_ranks: Dict[str, int] = {}  # module_name → new rank

    keys_in = len(sd_p) + (len(sd_d) if sd_d is not None else 0)

    for mod_name in all_modules:
        block_id = _block_id_from_key(mod_name)
        if block_id is None:
            # Not a transformer-block module — pass primary's keys through if present.
            if mod_name in modules_p:
                for suffix, tensor in modules_p[mod_name].items():
                    sd_out[f"{mod_name}.{suffix}"] = tensor
            continue

        bs = state.blocks.get(block_id)
        if bs is None:
            # No slider state for this block — keep primary as-is.
            if mod_name in modules_p:
                for suffix, tensor in modules_p[mod_name].items():
                    sd_out[f"{mod_name}.{suffix}"] = tensor
            continue

        has_p = mod_name in modules_p
        has_d = mod_name in modules_d
        p_on = bs.primary_enabled and has_p
        d_on = bs.donor_enabled and has_d

        if not p_on and not d_on:
            dropped_blocks.add(block_id)
            continue

        if p_on and d_on:
            # Rank-concat: both contributions baked and concatenated.
            p_baked = _bake_single_contribution(modules_p[mod_name], bs.primary_strength)
            d_baked = _bake_single_contribution(modules_d[mod_name], bs.donor_strength)
            if p_baked is None or d_baked is None:
                # Fall back to whichever side has valid keys (shouldn't happen for well-formed std LoRA).
                logger.warning(f"Concat bake skipped for {mod_name} — missing up/down in one side")
                if p_baked is not None:
                    new_up, down, rank = p_baked
                    sd_out[f"{mod_name}.lora_up.weight"] = new_up
                    sd_out[f"{mod_name}.lora_down.weight"] = down
                    sd_out[f"{mod_name}.alpha"] = torch.tensor(float(rank))
                    rescaled_blocks.add(block_id)
                elif d_baked is not None:
                    new_up, down, rank = d_baked
                    sd_out[f"{mod_name}.lora_up.weight"] = new_up
                    sd_out[f"{mod_name}.lora_down.weight"] = down
                    sd_out[f"{mod_name}.alpha"] = torch.tensor(float(rank))
                    rescaled_blocks.add(block_id)
                continue

            up_p, down_p, rank_p = p_baked
            up_d, down_d, rank_d = d_baked

            # Normalize dtypes so cat doesn't choke if donor was saved in a different dtype.
            common_dtype = up_p.dtype
            if up_d.dtype != common_dtype:
                up_d = up_d.to(common_dtype)
                down_d = down_d.to(common_dtype)
            if down_p.dtype != common_dtype:
                down_p = down_p.to(common_dtype)

            new_rank = rank_p + rank_d
            new_up = torch.cat([up_p, up_d], dim=-1)     # (out, rank_p + rank_d)
            new_down = torch.cat([down_p, down_d], dim=0)  # (rank_p + rank_d, in)
            sd_out[f"{mod_name}.lora_up.weight"] = new_up
            sd_out[f"{mod_name}.lora_down.weight"] = new_down
            # alpha = rank so inference scale = 1.0; mult + original scales already in new_up.
            sd_out[f"{mod_name}.alpha"] = torch.tensor(float(new_rank))
            blended_blocks.add(block_id)
            combined_ranks[mod_name] = new_rank

        elif p_on:
            baked = _bake_single_contribution(modules_p[mod_name], bs.primary_strength)
            if baked is None:
                # Non-standard module in primary — pass raw keys (future-proofing).
                for suffix, tensor in modules_p[mod_name].items():
                    sd_out[f"{mod_name}.{suffix}"] = tensor
                continue
            new_up, down, rank = baked
            sd_out[f"{mod_name}.lora_up.weight"] = new_up
            sd_out[f"{mod_name}.lora_down.weight"] = down
            sd_out[f"{mod_name}.alpha"] = torch.tensor(float(rank))
            if bs.primary_strength != 1.0:
                rescaled_blocks.add(block_id)

        elif d_on:
            baked = _bake_single_contribution(modules_d[mod_name], bs.donor_strength)
            if baked is None:
                continue
            new_up, down, rank = baked
            sd_out[f"{mod_name}.lora_up.weight"] = new_up
            sd_out[f"{mod_name}.lora_down.weight"] = down
            sd_out[f"{mod_name}.alpha"] = torch.tensor(float(rank))
            rescaled_blocks.add(block_id)

    # Record slider config + donor reference in metadata.
    try:
        metadata["ss_repair_studio_config"] = json.dumps(state.to_json(), separators=(",", ":"))
    except Exception:
        logger.exception("Could not serialize SliderState into metadata")
    if donor_path:
        metadata["ss_repair_studio_donor_path"] = os.path.basename(donor_path)
    if combined_ranks:
        metadata["ss_repair_studio_combined_ranks"] = json.dumps(
            {k: v for k, v in sorted(combined_ranks.items())}, separators=(",", ":"))
    if lycoris_converted:
        metadata["ss_repair_studio_lycoris_svd"] = str(lycoris_converted)

    # safetensors requires contiguous tensors
    sd_out = {k: v.contiguous() if v.is_floating_point() else v for k, v in sd_out.items()}
    save_file(sd_out, out_path, metadata=metadata)

    summary = {
        "dropped_blocks": sorted(dropped_blocks),
        "rescaled_blocks": sorted(rescaled_blocks - dropped_blocks),
        "blended_blocks": sorted(blended_blocks),
        "keys_in": keys_in,
        "keys_out": len(sd_out),
        "donor_path": donor_path,
    }
    logger.info(
        "save_repaired_lora: primary_in=%d donor_in=%d out=%d dropped=%d rescaled=%d blended=%d → %s",
        len(sd_p), len(sd_d) if sd_d is not None else 0, summary["keys_out"],
        len(summary["dropped_blocks"]), len(summary["rescaled_blocks"]), len(summary["blended_blocks"]),
        out_path,
    )
    return summary
