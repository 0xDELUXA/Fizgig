"""Visualization for LoRA profiling results — interactive HTML."""

import datetime
import hashlib
import json
import logging
import os
import re
from typing import Optional

import numpy as np

from fizgig.profiler.profiler import ProfileResult

logger = logging.getLogger(__name__)


def compute_lora_hash(lora_path: str) -> Optional[str]:
    """SHA-256 of the file bytes — a stable LoRA identifier for cross-tab lookups
    (Profiler writes it into its sidecar; Repair Studio hashes the loaded primary
    and scans profiles_dir for a match).

    Returns None on any IO error or missing path. Typical ~50 MB LoRA hashes in
    well under 1 second.
    """
    if not lora_path or not os.path.isfile(lora_path):
        return None
    h = hashlib.sha256()
    try:
        with open(lora_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except Exception:
        logger.exception("Failed to compute LoRA hash for %s", lora_path)
        return None
    return h.hexdigest()


def _short_name(block_name: str) -> str:
    return block_name.replace("_blocks.", " ")


def _get_category(block_name: str) -> str:
    """Report categorization with dual-class overlap regions.

    Reflects the honest block map:
      - double 0-7:        style_composition
      - single 0:          style_composition
      - single 1:          style_ident_overlap  (belongs to BOTH style_composition and identity)
      - single 2-11:       identity
      - single 12-16:      ident_details_overlap (belongs to BOTH identity and details)
      - single 17-23:      details
    """
    if block_name.startswith("double"):
        return "style_composition"
    idx = int(block_name.split(".")[1])
    if idx == 0:
        return "style_composition"
    if idx == 1:
        return "style_ident_overlap"
    if 2 <= idx <= 11:
        return "identity"
    if 12 <= idx <= 16:
        return "ident_details_overlap"
    return "details"  # 17-23


CATEGORY_ORDER = [
    "style_composition",
    "style_ident_overlap",
    "identity",
    "ident_details_overlap",
    "details",
]


def _category_totals(result: ProfileResult) -> dict:
    totals = {c: 0.0 for c in CATEGORY_ORDER}
    for i, name in enumerate(result.block_names):
        totals[_get_category(name)] += result.magnitudes[i].sum()
    return totals


# ==============================================================================
# Static analysis — instant file-only inspection, no GPU
# ==============================================================================

def _detect_lora_type_from_keys(keys) -> str:
    """Classify LoRA variant from key patterns (LoRA / LoKR / LoHa / GLoRA).
    Ported from the ComfyUI realtime-lora analyzer — pure string matching, no GPU.
    """
    for key in keys:
        kl = key.lower()
        if ".lokr_w1" in kl:
            return "LoKR"
        if ".hada_w1_a" in kl or ".hada_w2_a" in kl:
            return "LoHa"
        if ".glora_a" in kl or ".glora_b" in kl:
            return "GLoRA"
    return "LoRA"


_BLOCK_NAME_RE = re.compile(r"(?:lora_unet_)?(double_blocks|single_blocks)_(\d+)_")


def _block_name_from_lora_name(lora_name: str) -> Optional[str]:
    """Extract block name like 'double.5' or 'single.14' from a lora_name key prefix.
    Uses the same dotted convention as ProfileResult.block_names so _get_category matches.
    """
    m = _BLOCK_NAME_RE.search(lora_name)
    if m:
        return f"{m.group(1).replace('_blocks', '_blocks')}.{int(m.group(2))}"
    return None


def _compute_static_analysis(lora_path: str) -> dict:
    """Read a LoRA file and compute static (non-GPU) analysis.

    Returns a dict suitable for JSON injection into the HTML report. Every field
    is optional — fields sourced from missing metadata or missing LoRA modules
    are left empty and silently skipped by the renderer.
    """
    result = {
        "lora_type": "LoRA",
        "is_peft_converted": False,
        "title": "",
        "base_model": "",
        "network_module": "",
        "trained_by": "",           # human-friendly: Fizgig / Kohya / ai-toolkit / PEFT / etc.
        "rank": 0,
        "alpha": 0.0,
        "num_modules": 0,
        "num_keys": 0,
        "file_size_mb": 0.0,
        "trained_at": "",
        "static_top_blocks": [],
        "top_tags": [],
        "sample_prompts": [],
        "training_comment": "",
        "model_description": "",
        "model_tags": "",
        "usage_hint": "",
        "dataset_dirs": [],
        "datasets_json": "",        # raw ss_datasets JSON excerpt (truncated)
        "context_lora": "",
        "context_lora_strength": "",
        "learning_rate": "",
        "optimizer": "",
        "num_epochs": "",
        "steps": "",
    }

    if not lora_path or not os.path.exists(lora_path):
        return result

    try:
        result["file_size_mb"] = round(os.path.getsize(lora_path) / (1024 * 1024), 2)
    except OSError:
        pass

    # Read safetensors metadata header (no tensor loads)
    try:
        from fizgig.training.metadata import load_metadata_from_safetensors
        meta = load_metadata_from_safetensors(lora_path) or {}
    except Exception as e:
        logger.info(f"[static] metadata read failed for {lora_path}: {e}")
        meta = {}

    # Base model + network identity + display title
    result["base_model"] = meta.get("ss_base_model_version") or meta.get("modelspec.architecture") or ""
    result["network_module"] = meta.get("ss_network_module") or ""
    result["title"] = meta.get("modelspec.title") or meta.get("name") or meta.get("ss_output_name") or ""

    # Trained-by heuristic — check software key (PEFT), ss_network_module (Fizgig/Kohya)
    _software = meta.get("software", "")
    if _software:
        try:
            s = json.loads(_software) if _software.startswith("{") else {}
            sw_name = s.get("name", _software) if isinstance(s, dict) else _software
            result["trained_by"] = sw_name
        except (json.JSONDecodeError, TypeError):
            result["trained_by"] = str(_software)[:64]
    elif "fizgig" in result["network_module"].lower():
        result["trained_by"] = "Fizgig"
    elif result["network_module"]:
        result["trained_by"] = result["network_module"]
    try:
        if meta.get("ss_network_dim"):
            result["rank"] = int(meta["ss_network_dim"])
    except (ValueError, TypeError):
        pass
    try:
        if meta.get("ss_network_alpha"):
            result["alpha"] = float(meta["ss_network_alpha"])
    except (ValueError, TypeError):
        pass

    # Timestamps + free-text notes
    _ts = meta.get("ss_training_finished_at", "")
    if _ts:
        try:
            from datetime import datetime
            result["trained_at"] = datetime.fromtimestamp(float(_ts)).strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError, OSError):
            result["trained_at"] = str(_ts)

    def _clean(v):
        """Filter out literal 'None' strings that Kohya sometimes writes for absent fields."""
        s = (v or "").strip()
        return "" if s.lower() == "none" else s
    result["training_comment"] = _clean(meta.get("ss_training_comment"))
    result["model_description"] = _clean(meta.get("modelspec.description"))
    result["model_tags"] = _clean(meta.get("modelspec.tags"))
    result["usage_hint"] = _clean(meta.get("modelspec.usage_hint"))
    result["context_lora"] = meta.get("ss_context_lora", "")
    result["context_lora_strength"] = meta.get("ss_context_lora_strength", "")
    result["learning_rate"] = meta.get("ss_learning_rate", "")
    result["optimizer"] = meta.get("ss_optimizer", "")
    result["num_epochs"] = meta.get("ss_num_epochs", "")
    result["steps"] = meta.get("ss_steps", "")

    # Parse JSON-valued metadata
    try:
        tag_freq_raw = meta.get("ss_tag_frequency", "")
        if tag_freq_raw:
            tf = json.loads(tag_freq_raw)
            # ss_tag_frequency can be {dir: {tag: count}} OR {tag: count} depending on trainer
            flat = {}
            if isinstance(tf, dict):
                for k, v in tf.items():
                    if isinstance(v, dict):
                        for tag, count in v.items():
                            flat[tag] = flat.get(tag, 0) + int(count)
                    elif isinstance(v, (int, float)):
                        flat[k] = flat.get(k, 0) + int(v)
            top = sorted(flat.items(), key=lambda kv: kv[1], reverse=True)[:15]
            result["top_tags"] = [{"tag": t, "count": c} for t, c in top]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    try:
        dd_raw = meta.get("ss_dataset_dirs", "")
        if dd_raw:
            dd = json.loads(dd_raw)
            if isinstance(dd, dict):
                # Values can be dicts with 'img_count' / 'n_repeats', or ints
                items = []
                for k, v in dd.items():
                    if isinstance(v, dict):
                        items.append({"dir": k, "count": int(v.get("img_count", v.get("n_repeats", 0)))})
                    else:
                        items.append({"dir": k, "count": int(v)})
                items.sort(key=lambda x: x["count"], reverse=True)
                result["dataset_dirs"] = items[:10]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    _sp = meta.get("ss_sample_prompts", "")
    if _sp:
        result["sample_prompts"] = [line.strip() for line in _sp.splitlines() if line.strip()][:5]

    # ss_datasets — Kohya JSON array with resolution/batch/num_repeats per dataset
    _ds = meta.get("ss_datasets", "")
    if _ds:
        try:
            ds = json.loads(_ds)
            if isinstance(ds, list) and ds:
                d0 = ds[0]
                parts = []
                if "resolution" in d0:
                    parts.append(f"res {d0['resolution']}")
                if "num_repeats" in d0:
                    parts.append(f"repeats {d0['num_repeats']}")
                if "batch_size_per_device" in d0:
                    parts.append(f"batch {d0['batch_size_per_device']}")
                if "caption_extension" in d0:
                    parts.append(f"captions {d0['caption_extension']}")
                result["datasets_json"] = " · ".join(parts)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # PEFT-format "training_info" field seen in ai-toolkit exports
    _ti = meta.get("training_info", "")
    if _ti and not result["steps"]:
        try:
            ti = json.loads(_ti)
            if isinstance(ti, dict):
                if "step" in ti:
                    result["steps"] = str(ti["step"])
                if "epoch" in ti and not result["num_epochs"]:
                    result["num_epochs"] = str(ti["epoch"])
        except (json.JSONDecodeError, TypeError):
            pass

    # ---- Tensor-level analysis: LoRA type + per-block static norm ----
    try:
        from safetensors.torch import load_file
        from fizgig.networks.lora import detect_lora_format, ensure_kohya_lora_state_dict
        weights_sd = load_file(lora_path)
    except Exception as e:
        logger.info(f"[static] weights read failed for {lora_path}: {e}")
        return result

    raw_keys = list(weights_sd.keys())
    result["num_keys"] = len(raw_keys)
    result["lora_type"] = _detect_lora_type_from_keys(raw_keys)

    # PEFT conversion detection: if format is peft, normalise for analysis
    try:
        if detect_lora_format(weights_sd) == "peft":
            result["is_peft_converted"] = True
            weights_sd = ensure_kohya_lora_state_dict(weights_sd)
    except Exception:
        pass

    # Only compute per-block norms for standard LoRA (lora_down/up). Other variants skip.
    if result["lora_type"] != "LoRA":
        return result

    # Group keys by lora_name (strip .lora_down/up/alpha suffixes)
    modules = {}  # lora_name -> {"down": tensor, "up": tensor, "alpha": tensor}
    for key, tensor in weights_sd.items():
        if key.endswith(".lora_down.weight"):
            name = key[: -len(".lora_down.weight")]
            modules.setdefault(name, {})["down"] = tensor
        elif key.endswith(".lora_up.weight"):
            name = key[: -len(".lora_up.weight")]
            modules.setdefault(name, {})["up"] = tensor
        elif key.endswith(".alpha"):
            name = key[: -len(".alpha")]
            modules.setdefault(name, {})["alpha"] = tensor

    complete = {n: m for n, m in modules.items() if "down" in m and "up" in m}
    result["num_modules"] = len(complete)

    # Infer rank / alpha from the first module if metadata didn't provide them
    if complete and result["rank"] == 0:
        first = next(iter(complete.values()))
        try:
            result["rank"] = int(first["down"].shape[0])
        except (AttributeError, IndexError):
            pass
    if complete and result["alpha"] == 0.0:
        for m in complete.values():
            if "alpha" in m:
                try:
                    result["alpha"] = float(m["alpha"].item())
                    break
                except Exception:
                    pass

    # Per-block static norm = sum over modules inside that block of ||up||_F * ||down||_F
    block_norms = {}  # block_name like "double.5" -> summed norm
    for lora_name, m in complete.items():
        try:
            up_norm = m["up"].float().norm().item()
            down_norm = m["down"].float().norm().item()
            mod_norm = up_norm * down_norm
        except Exception:
            continue
        block = _block_name_from_lora_name(lora_name)
        if block is None:
            continue
        block_norms[block] = block_norms.get(block, 0.0) + mod_norm

    if block_norms:
        max_norm = max(block_norms.values())
        ranked = sorted(block_norms.items(), key=lambda kv: kv[1], reverse=True)
        result["static_top_blocks"] = [
            {
                "name": _short_name(name),
                "category": _get_category(name),
                "norm_pct": round((norm / max_norm) * 100, 1) if max_norm > 0 else 0.0,
                "raw_norm": round(norm, 4),
            }
            for name, norm in ranked[:10]
        ]

    return result


def plot_profile_heatmap(
    result: ProfileResult,
    output_path: str,
    title: Optional[str] = None,
    **kwargs,
):
    """Generate an interactive HTML profile visualization.

    Despite the name (kept for backward compat), this generates HTML not PNG.
    """
    if not output_path.endswith(".html"):
        output_path = output_path.rsplit(".", 1)[0] + ".html"

    if title is None:
        lora_name = os.path.basename(result.metadata.get("lora_path", "Unknown"))
        title = lora_name

    # Build data for JS
    blocks = []
    for i, name in enumerate(result.block_names):
        cat = _get_category(name)
        blocks.append({
            "name": _short_name(name),
            "fullName": name,
            "category": cat,
            "values": result.magnitudes[i].tolist(),
        })

    bin_labels = [f"{lo:.1f}-{hi:.1f}" for lo, hi in result.timestep_bins]
    cat_totals = _category_totals(result)
    grand_total = sum(cat_totals.values()) or 1.0

    # Top blocks
    top_blocks = []
    for name, total in result.get_top_blocks(10):
        top_blocks.append({
            "name": _short_name(name),
            "category": _get_category(name),
            "pct": round(total / grand_total * 100, 1),
        })

    # Static (file-only) analysis — runs in a fraction of a second, no GPU
    static = _compute_static_analysis(result.metadata.get("lora_path", ""))

    # Hash the source LoRA once — embedded in the HTML meta + the sidecar JSON
    # so the Repair Studio can cross-link to the matching report by content hash.
    lora_path = result.metadata.get("lora_path", "")
    lora_hash = compute_lora_hash(lora_path)

    js_data = json.dumps({
        "title": title,
        "blocks": blocks,
        "binLabels": bin_labels,
        "numBins": result.num_bins,
        "categoryPcts": {
            c: round(cat_totals[c] / grand_total * 100, 1) for c in CATEGORY_ORDER
        },
        "topBlocks": top_blocks,
        "metadata": {
            "prompt": result.metadata.get("prompt", ""),
            "resolution": f"{result.metadata.get('width', '?')}x{result.metadata.get('height', '?')}",
        },
        "static": static,
    })

    html = _build_html(js_data, lora_hash=lora_hash)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Profile visualization saved: {output_path}")

    # Sidecar JSON next to the HTML — Repair Studio reads this for quick lookup
    # (top active blocks + static top-weight blocks + prompt + timestamp). We
    # keep both the HTML meta tag (single-file portability) and the sidecar
    # (trivially parseable without HTML scraping).
    _write_profile_sidecar(output_path, result, top_blocks, static, lora_hash)


def _write_profile_sidecar(
    report_path: str, result: ProfileResult,
    top_blocks: list, static: dict, lora_hash: Optional[str],
) -> None:
    """Emit a compact JSON summary next to the HTML report for cross-tab lookup."""
    sidecar_path = report_path.rsplit(".", 1)[0] + ".json"
    lora_path = result.metadata.get("lora_path", "")
    payload = {
        "version": 1,
        "hash": lora_hash,
        "lora_path": lora_path,
        "lora_name": os.path.basename(lora_path) if lora_path else "",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "prompt": result.metadata.get("prompt", ""),
        "resolution": f"{result.metadata.get('width', '?')}x{result.metadata.get('height', '?')}",
        "num_bins": result.num_bins,
        "html_report": os.path.basename(report_path),
        "top_active_blocks": top_blocks,  # already [{name, category, pct}]
        "top_static_blocks": static.get("static_top_blocks", []) if static else [],
    }
    try:
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Profile sidecar written: {sidecar_path}")
    except Exception:
        logger.exception(f"Failed to write profile sidecar at {sidecar_path}")


def _build_html(js_data: str, lora_hash: Optional[str] = None) -> str:
    # Embed the source LoRA's SHA-256 hash so the Repair Studio can locate
    # this report by content hash even if the filename changes.
    hash_meta = f'<meta name="fizgig-lora-hash" content="{lora_hash}">' if lora_hash else ""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="fizgig-profile-version" content="1">
{hash_meta}
<title>LoRA Profile</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', sans-serif; background: #1B2028; color: #ECF0F1; padding: 20px; min-height: 100vh; }}
h1 {{ font-size: 22px; margin-bottom: 4px; }}
.subtitle {{ color: #95A5A6; font-size: 14px; margin-bottom: 20px; }}
.category-bar {{ display: flex; gap: 20px; margin-bottom: 20px; font-size: 15px; font-weight: 600; }}
.cat-comp {{ color: #5B9BD5; }}
.cat-sio   {{ color: #5BB3A6; }}   /* style↔identity overlap (teal: between blue and green) */
.cat-ident {{ color: #70AD47; }}
.cat-ido   {{ color: #B8A547; }}   /* identity↔details overlap (olive: between green and orange) */
.cat-details {{ color: #ED7D31; }}
.slider-row {{ display: flex; align-items: center; gap: 15px; margin-bottom: 15px; background: #2C3040; padding: 12px 18px; border-radius: 8px; }}
.slider-row label {{ font-size: 13px; color: #BDC3C7; white-space: nowrap; }}
.slider-row input[type=range] {{ flex: 1; accent-color: #3498DB; }}
.slider-label {{ font-size: 14px; font-weight: bold; color: #ECF0F1; min-width: 80px; text-align: center; }}
.blocks-container {{ display: flex; align-items: flex-end; gap: 2px; height: 320px; margin-bottom: 0; padding: 0 4px; }}
.block-col {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; min-width: 0; }}
.block-bar {{ width: 100%; border-radius: 3px 3px 0 0; transition: height 0.3s ease; min-height: 2px; }}
.block-bar.style_composition {{ background: #5B9BD5; }}
.block-bar.style_ident_overlap {{ background: #5BB3A6; }}
.block-bar.identity {{ background: #70AD47; }}
.block-bar.ident_details_overlap {{ background: #B8A547; }}
.block-bar.details {{ background: #ED7D31; }}
.block-bar:hover {{ filter: brightness(1.3); }}
.labels-container {{ display: flex; gap: 2px; padding: 0 4px; height: 40px; }}
.label-col {{ flex: 1; display: flex; align-items: flex-start; justify-content: center; min-width: 0; overflow: hidden; }}
.block-label {{ font-size: 10px; color: #999; transform: rotate(-45deg); transform-origin: top left; white-space: nowrap; margin-top: 4px; margin-left: 10px; }}
.separator {{ width: 1px; background: #555; height: 100%; margin: 0 1px; flex-shrink: 0; }}
.section-label {{ display: flex; gap: 2px; margin-bottom: 4px; padding: 0 4px; }}
.section-label span {{ flex: 1; text-align: center; font-size: 11px; font-weight: 600; padding: 4px 0; border-radius: 4px; }}
.section-label .sl-comp {{ background: rgba(91,155,213,0.15); color: #5B9BD5; }}
.section-label .sl-sio  {{ background: rgba(91,179,166,0.18); color: #5BB3A6; }}
.section-label .sl-ident {{ background: rgba(112,173,71,0.15); color: #70AD47; }}
.section-label .sl-ido  {{ background: rgba(184,165,71,0.18); color: #B8A547; }}
.section-label .sl-details {{ background: rgba(237,125,49,0.15); color: #ED7D31; }}
.summary {{ background: #2C3040; border-radius: 8px; padding: 18px 22px; margin-top: 20px; }}
.summary h2 {{ font-size: 16px; margin-bottom: 12px; color: #ECF0F1; }}
.summary h3 {{ font-size: 13px; margin-bottom: 8px; color: #BDC3C7; text-transform: uppercase; letter-spacing: 0.6px; }}
.summary-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
.static-header {{ background: #2C3040; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; font-size: 13px; line-height: 1.7; }}
.static-header .row {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: baseline; }}
.static-header .label {{ color: #95A5A6; }}
.static-header .value {{ color: #ECF0F1; }}
.static-header .divider {{ color: #555; }}
.lora-badge {{ display: inline-block; padding: 2px 10px; border-radius: 10px; background: #3498DB; color: white; font-size: 11px; font-weight: 600; margin-left: 8px; vertical-align: middle; }}
.lora-badge.peft {{ background: #9B59B6; }}
.lora-badge.lokr {{ background: #E67E22; }}
.lora-badge.loha {{ background: #E67E22; }}
.lora-badge.glora {{ background: #E67E22; }}
.training-content {{ background: #252936; border-radius: 6px; padding: 10px 14px; margin-top: 10px; font-size: 12px; line-height: 1.6; color: #BDC3C7; }}
.training-content strong {{ color: #ECF0F1; }}
.training-content .tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }}
.training-content .tag {{ background: #3A3F52; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-family: monospace; color: #ECF0F1; }}
.training-content .tag .cnt {{ color: #95A5A6; margin-left: 4px; }}
.top-blocks {{ font-size: 13px; line-height: 1.8; }}
.top-blocks .entry {{ display: flex; align-items: center; gap: 8px; }}
.top-blocks .bar {{ height: 10px; border-radius: 2px; }}
.top-blocks .bar.style_composition {{ background: #5B9BD5; }}
.top-blocks .bar.style_ident_overlap {{ background: #5BB3A6; }}
.top-blocks .bar.identity {{ background: #70AD47; }}
.top-blocks .bar.ident_details_overlap {{ background: #B8A547; }}
.top-blocks .bar.details {{ background: #ED7D31; }}
.top-blocks .name {{ min-width: 90px; font-family: monospace; color: #BDC3C7; }}
.top-blocks .pct {{ min-width: 40px; text-align: right; color: #ECF0F1; }}
.info-section {{ font-size: 13px; color: #95A5A6; line-height: 1.8; }}
.info-section strong {{ color: #BDC3C7; }}
.tooltip {{ position: fixed; background: #2C3040; border: 1px solid #555; border-radius: 6px; padding: 8px 12px; font-size: 12px; pointer-events: none; display: none; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }}
</style>
</head>
<body>

<h1 id="title"></h1>
<div class="subtitle" id="subtitle"></div>
<div class="static-header" id="staticHeader"></div>
<div class="category-bar" id="catBar"></div>

<div class="slider-row">
    <button id="btnOverall" onclick="setView('overall')" style="padding:6px 16px; border:1px solid #555; background:#2C3040; color:#ccc; border-radius:4px; cursor:pointer; font-size:13px;">Overall</button>
    <button id="btnPerstep" onclick="setView('perstep')" style="padding:6px 16px; border:1px solid #555; background:#3498DB; color:#fff; border-radius:4px; cursor:pointer; font-size:13px;">By Stage</button>
    <span style="width:20px;"></span>
    <label style="font-size:13px; color:#666;" id="sliderLabelLeft">Step 1</label>
    <input type="range" id="tsSlider" min="0" max="4" value="0" step="1">
    <label style="font-size:13px; color:#666;" id="sliderLabelRight"></label>
    <span class="slider-label" id="tsLabel" style="min-width: 100px; color: #3498DB;"></span>
</div>

<div class="section-label" id="sectionLabels"></div>
<div class="blocks-container" id="blocks"></div>
<div class="labels-container" id="blockLabels"></div>

<div class="summary" id="summary"></div>
<div class="tooltip" id="tooltip"></div>

<script>
const DATA = {js_data};

let viewMode = 'perstep'; // 'perstep', 'overall', 'timestep'
let currentSlider = 0; // slider position

function init() {{
    document.getElementById('title').textContent = 'LoRA Profile: ' + DATA.title;
    document.getElementById('subtitle').textContent =
        'Prompt: "' + DATA.metadata.prompt + '"  |  ' + DATA.metadata.resolution + ' inference';

    renderStaticHeader();

    const catBar = document.getElementById('catBar');
    catBar.innerHTML = `
        <span class="cat-comp">Style+Comp ${{DATA.categoryPcts.style_composition}}%</span>
        <span class="cat-sio">Style/ID ${{DATA.categoryPcts.style_ident_overlap}}%</span>
        <span class="cat-ident">Identity ${{DATA.categoryPcts.identity}}%</span>
        <span class="cat-ido">ID/Detail ${{DATA.categoryPcts.ident_details_overlap}}%</span>
        <span class="cat-details">Details ${{DATA.categoryPcts.details}}%</span>
    `;

    document.getElementById('tsSlider').max = DATA.numBins - 1;

    buildBlocks();
    buildSummary();
    setView('perstep'); // Default to Per Step — must be after buildBlocks
}}

function buildBlocks() {{
    const container = document.getElementById('blocks');
    const sectionLabels = document.getElementById('sectionLabels');
    const blockLabels = document.getElementById('blockLabels');
    container.innerHTML = '';
    sectionLabels.innerHTML = '';
    blockLabels.innerHTML = '';

    let prevCat = null;
    let catCounts = {{ style_composition: 0, style_ident_overlap: 0, identity: 0, ident_details_overlap: 0, details: 0 }};
    DATA.blocks.forEach(b => catCounts[b.category]++);

    // Section labels (5 categories — empty sections are auto-hidden via flex:0)
    sectionLabels.innerHTML = `
        <span class="sl-comp" style="flex:${{catCounts.style_composition}}; ${{catCounts.style_composition?'':'display:none;'}}">Style+Comp</span>
        <span class="sl-sio" style="flex:${{catCounts.style_ident_overlap}}; ${{catCounts.style_ident_overlap?'':'display:none;'}}">Style/ID</span>
        <span class="sl-ident" style="flex:${{catCounts.identity}}; ${{catCounts.identity?'':'display:none;'}}">Identity</span>
        <span class="sl-ido" style="flex:${{catCounts.ident_details_overlap}}; ${{catCounts.ident_details_overlap?'':'display:none;'}}">ID/Detail</span>
        <span class="sl-details" style="flex:${{catCounts.details}}; ${{catCounts.details?'':'display:none;'}}">Details</span>
    `;

    DATA.blocks.forEach((block, i) => {{
        if (prevCat && block.category !== prevCat) {{
            // Separator in bars
            const sep = document.createElement('div');
            sep.className = 'separator';
            container.appendChild(sep);
            // Matching spacer in labels
            const sepL = document.createElement('div');
            sepL.style.cssText = 'width:1px; flex-shrink:0; margin:0 1px;';
            blockLabels.appendChild(sepL);
        }}
        prevCat = block.category;

        // Bar column (no label inside — keeps heights consistent)
        const col = document.createElement('div');
        col.className = 'block-col';

        const bar = document.createElement('div');
        bar.className = 'block-bar ' + block.category;
        bar.id = 'bar-' + i;
        bar.addEventListener('mouseenter', (e) => showTooltip(e, block));
        bar.addEventListener('mouseleave', hideTooltip);

        col.appendChild(bar);
        container.appendChild(col);

        // Label in separate container below
        const lblCol = document.createElement('div');
        lblCol.className = 'label-col';
        const lbl = document.createElement('div');
        lbl.className = 'block-label';
        lbl.textContent = block.name.replace('single ', 's').replace('double ', 'd');
        lblCol.appendChild(lbl);
        blockLabels.appendChild(lblCol);
    }});
}}

function getBinForSlider(pos) {{
    // Slider 0 = early (high noise, last bin), slider max = late (low noise, first bin)
    return (DATA.numBins - 1) - pos;
}}

function updateBars() {{
    const binIdx = getBinForSlider(currentSlider);

    let maxVal = 0;
    DATA.blocks.forEach(block => {{
        let val = viewMode === 'overall' ? block.values.reduce((a, b) => a + b, 0) : block.values[binIdx];
        maxVal = Math.max(maxVal, val);
    }});
    if (maxVal === 0) maxVal = 1;

    DATA.blocks.forEach((block, i) => {{
        let val = viewMode === 'overall' ? block.values.reduce((a, b) => a + b, 0) : block.values[binIdx];
        const pct = (val / maxVal) * 100;
        const bar = document.getElementById('bar-' + i);
        bar.style.height = Math.max(2, pct) + '%';
    }});

    // Update label
    if (viewMode === 'overall') {{
        document.getElementById('tsLabel').textContent = '';
    }} else {{
        document.getElementById('tsLabel').textContent = 'Stage ' + currentSlider + '  (t=' + DATA.binLabels[binIdx] + ')';
    }}
}}

function setView(mode) {{
    viewMode = mode;
    const slider = document.getElementById('tsSlider');
    const btnMap = {{'perstep': 'btnPerstep', 'overall': 'btnOverall'}};
    Object.entries(btnMap).forEach(([m, id]) => {{
        const btn = document.getElementById(id);
        btn.style.background = m === mode ? '#3498DB' : '#2C3040';
        btn.style.color = m === mode ? '#fff' : '#ccc';
    }});

    const sliderEnabled = mode !== 'overall';
    slider.disabled = !sliderEnabled;
    slider.style.opacity = sliderEnabled ? '1' : '0.3';

    const leftLabel = document.getElementById('sliderLabelLeft');
    const rightLabel = document.getElementById('sliderLabelRight');
    if (sliderEnabled) {{
        leftLabel.textContent = 'Early';
        rightLabel.textContent = 'Late';
    }} else {{
        leftLabel.textContent = '';
        rightLabel.textContent = '';
    }}

    updateBars();
}}

document.getElementById('tsSlider').addEventListener('input', (e) => {{
    currentSlider = parseInt(e.target.value);
    updateBars();
}});

function showTooltip(e, block) {{
    const tip = document.getElementById('tooltip');
    const total = block.values.reduce((a, b) => a + b, 0);
    let html = `<strong>${{block.name}}</strong> [${{block.category}}]<br>`;
    html += `Total activity: ${{total.toFixed(1)}}<br>`;
    block.values.forEach((v, i) => {{
        html += `t=${{DATA.binLabels[i]}}: ${{v.toFixed(1)}}<br>`;
    }});
    tip.innerHTML = html;
    tip.style.display = 'block';
    tip.style.left = (e.clientX + 15) + 'px';
    tip.style.top = (e.clientY - 10) + 'px';
}}

function hideTooltip() {{
    document.getElementById('tooltip').style.display = 'none';
}}

function renderStaticHeader() {{
    const hdr = document.getElementById('staticHeader');
    const s = DATA.static || {{}};
    // If we got basically nothing, hide the whole header
    if (!s.base_model && !s.num_keys && !s.file_size_mb) {{
        hdr.style.display = 'none';
        return;
    }}
    const esc = t => (t == null ? '' : String(t)).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    // Badge class
    let badgeCls = 'lora-badge';
    let badgeTxt = s.lora_type || 'LoRA';
    if (s.lora_type === 'LoKR') badgeCls += ' lokr';
    else if (s.lora_type === 'LoHa') badgeCls += ' loha';
    else if (s.lora_type === 'GLoRA') badgeCls += ' glora';
    else if (s.is_peft_converted) {{ badgeCls += ' peft'; badgeTxt = 'LoRA (PEFT-converted)'; }}

    // Factual line
    const parts1 = [];
    if (s.title) parts1.push(`<span class="label">Title:</span> <span class="value"><strong>${{esc(s.title)}}</strong></span>`);
    if (s.base_model) parts1.push(`<span class="label">Base:</span> <span class="value">${{esc(s.base_model)}}</span>`);
    if (s.trained_by) parts1.push(`<span class="label">Trained by:</span> <span class="value">${{esc(s.trained_by)}}</span>`);
    if (s.rank) parts1.push(`<span class="label">Rank/Alpha:</span> <span class="value">${{s.rank}}/${{s.alpha}}</span>`);
    const row1 = parts1.join(' <span class="divider">·</span> ') + ` <span class="${{badgeCls}}">${{esc(badgeTxt)}}</span>`;

    const parts2 = [];
    if (s.num_keys) parts2.push(`<span class="label">Keys:</span> <span class="value">${{s.num_keys}}</span>`);
    if (s.num_modules) parts2.push(`<span class="label">Modules:</span> <span class="value">${{s.num_modules}}</span>`);
    if (s.file_size_mb) parts2.push(`<span class="label">Size:</span> <span class="value">${{s.file_size_mb}} MB</span>`);
    if (s.trained_at) parts2.push(`<span class="label">Trained:</span> <span class="value">${{esc(s.trained_at)}}</span>`);
    if (s.steps) parts2.push(`<span class="label">Steps:</span> <span class="value">${{esc(s.steps)}}</span>`);
    if (s.num_epochs) parts2.push(`<span class="label">Epochs:</span> <span class="value">${{esc(s.num_epochs)}}</span>`);
    if (s.learning_rate) parts2.push(`<span class="label">LR:</span> <span class="value">${{esc(s.learning_rate)}}</span>`);
    if (s.optimizer) {{
        let optShort = String(s.optimizer).split('.').pop();
        parts2.push(`<span class="label">Optimizer:</span> <span class="value">${{esc(optShort)}}</span>`);
    }}
    if (s.datasets_json) parts2.push(`<span class="label">Dataset:</span> <span class="value">${{esc(s.datasets_json)}}</span>`);
    const row2 = parts2.join(' <span class="divider">·</span> ');

    // Training content block (triggers, tags, sample prompts, notes)
    const tcParts = [];
    if (s.top_tags && s.top_tags.length) {{
        const tagHtml = s.top_tags.map(t => `<span class="tag">${{esc(t.tag)}}<span class="cnt">${{t.count}}</span></span>`).join('');
        tcParts.push(`<div><strong>Tags / triggers:</strong></div><div class="tags">${{tagHtml}}</div>`);
    }}
    if (s.sample_prompts && s.sample_prompts.length) {{
        const sp = s.sample_prompts.map(p => `<div>• ${{esc(p)}}</div>`).join('');
        tcParts.push(`<div style="margin-top:6px;"><strong>Sample prompts:</strong></div>${{sp}}`);
    }}
    if (s.model_description) tcParts.push(`<div style="margin-top:6px;"><strong>Description:</strong> ${{esc(s.model_description)}}</div>`);
    if (s.usage_hint) tcParts.push(`<div style="margin-top:6px;"><strong>Usage hint:</strong> ${{esc(s.usage_hint)}}</div>`);
    if (s.training_comment) tcParts.push(`<div style="margin-top:6px;"><strong>Training notes:</strong> ${{esc(s.training_comment)}}</div>`);
    if (s.model_tags) tcParts.push(`<div style="margin-top:6px;"><strong>Tags (modelspec):</strong> ${{esc(s.model_tags)}}</div>`);
    if (s.context_lora) tcParts.push(`<div style="margin-top:6px;"><strong>Context LoRA:</strong> ${{esc(s.context_lora)}} @ ${{esc(s.context_lora_strength || '1.0')}} <em>(use same pairing at inference)</em></div>`);
    if (s.dataset_dirs && s.dataset_dirs.length) {{
        const dd = s.dataset_dirs.map(d => `${{esc(d.dir)}} (${{d.count}})`).join(', ');
        tcParts.push(`<div style="margin-top:6px;"><strong>Dataset dirs:</strong> ${{dd}}</div>`);
    }}
    const tc = tcParts.length ? `<div class="training-content">${{tcParts.join('')}}</div>` : '';

    hdr.innerHTML = `<div class="row">${{row1}}</div><div class="row" style="margin-top:4px;">${{row2}}</div>${{tc}}`;
}}

function buildSummary() {{
    const summary = document.getElementById('summary');
    // Activation-based top blocks (existing)
    let topHtml = DATA.topBlocks.map(b => {{
        const barWidth = Math.max(4, b.pct * 3);
        return `<div class="entry">
            <span class="name">${{b.name}}</span>
            <span class="pct">${{b.pct}}%</span>
            <div class="bar ${{b.category}}" style="width:${{barWidth}}px"></div>
        </div>`;
    }}).join('');

    // Static weight-norm top blocks (new)
    const staticTop = (DATA.static && DATA.static.static_top_blocks) || [];
    let staticHtml = '';
    if (staticTop.length) {{
        staticHtml = staticTop.map(b => {{
            const barWidth = Math.max(4, b.norm_pct * 3);
            return `<div class="entry">
                <span class="name">${{b.name}}</span>
                <span class="pct">${{b.norm_pct}}%</span>
                <div class="bar ${{b.category}}" style="width:${{barWidth}}px"></div>
            </div>`;
        }}).join('');
    }} else {{
        staticHtml = '<div style="color:#95A5A6; font-style:italic; font-size:12px;">(no standard LoRA modules found — weight-norm analysis unavailable)</div>';
    }}

    let infoHtml = `
        <div style="margin-bottom:12px;"><strong>Where this LoRA's energy lives:</strong></div>
        <div><span class="cat-comp">■</span> <strong>Style+Composition ${{DATA.categoryPcts.style_composition}}%</strong> — overall look, colour palette, artistic style</div>
        <div><span class="cat-sio">■</span> <strong>Style↔Identity overlap ${{DATA.categoryPcts.style_ident_overlap}}%</strong> — where style meets subject</div>
        <div><span class="cat-ident">■</span> <strong>Identity ${{DATA.categoryPcts.identity}}%</strong> — face, body, subject recognition</div>
        <div><span class="cat-ido">■</span> <strong>Identity↔Details overlap ${{DATA.categoryPcts.ident_details_overlap}}%</strong> — fine features of the subject</div>
        <div><span class="cat-details">■</span> <strong>Details ${{DATA.categoryPcts.details}}%</strong> — textures, hair, skin, fabric detail</div>
        <br>
        <div style="color:#95A5A6; font-size:12px; line-height:1.6;">
            <strong>Most Active Blocks</strong> shows which blocks contribute most during image generation — this is what the LoRA actually <em>does</em> when you use it.<br>
            <strong>Highest Weight Norms</strong> shows which blocks have the largest stored weights — this is where the LoRA packs the most learned information.<br>
            When these two lists differ, it means some blocks store a lot but don't activate strongly (or vice versa) — a sign the LoRA may benefit from repair or extraction.
        </div>
    `;

    // Find blocks that appear in BOTH lists (activate AND store)
    const activeNames = new Set(DATA.topBlocks.map(b => b.name));
    const staticNames = new Set(staticTop.map(b => b.name));
    const bothNames = [...activeNames].filter(n => staticNames.has(n));
    let bothHtml = '';
    if (bothNames.length) {{
        bothHtml = `<div style="margin-top:16px; padding:10px 14px; background:#252936; border-radius:6px; border-left:3px solid #3498DB;">
            <strong style="color:#3498DB;">Key blocks</strong>
            <span style="color:#95A5A6; font-size:12px;"> — high in both activation and stored weight (the LoRA's core signal):</span><br>
            <span style="font-size:14px; font-weight:600; color:#ECF0F1;">${{bothNames.join(' &nbsp;·&nbsp; ')}}</span>
        </div>`;
    }}

    summary.innerHTML = `
        <h2>Summary</h2>
        <div class="summary-grid">
            <div><h3>Most Active Blocks</h3><div class="top-blocks">${{topHtml}}</div></div>
            <div><h3>Highest Weight Norms</h3><div class="top-blocks">${{staticHtml}}</div></div>
            <div class="info-section">${{infoHtml}}</div>
        </div>
        ${{bothHtml}}
    `;
}}

init();
</script>
</body>
</html>'''


def print_profile_summary(result: ProfileResult):
    """Print a text summary of the profiling result."""
    cat_totals = _category_totals(result)
    grand_total = sum(cat_totals.values()) or 1.0

    print(f"\n{'='*60}")
    print(f"LoRA Profile: {os.path.basename(result.metadata.get('lora_path', 'Unknown'))}")
    print(f"{'='*60}")

    # Static (file-only) analysis block — fast, no GPU
    static = _compute_static_analysis(result.metadata.get("lora_path", ""))
    if static.get("base_model") or static.get("num_keys"):
        print("Static analysis:")
        badge = static.get("lora_type", "LoRA")
        if static.get("is_peft_converted"):
            badge = "LoRA (PEFT-converted)"
        row = []
        if static.get("title"): row.append(f"Title: {static['title']}")
        if static.get("base_model"): row.append(f"Base: {static['base_model']}")
        if static.get("trained_by"): row.append(f"Trained by: {static['trained_by']}")
        if static.get("rank"): row.append(f"Rank/Alpha: {static['rank']}/{static['alpha']}")
        row.append(f"Type: {badge}")
        print("  " + " | ".join(row))
        row2 = []
        if static.get("num_keys"): row2.append(f"{static['num_keys']} keys")
        if static.get("num_modules"): row2.append(f"{static['num_modules']} modules")
        if static.get("file_size_mb"): row2.append(f"{static['file_size_mb']} MB")
        if static.get("trained_at"): row2.append(f"trained {static['trained_at']}")
        if row2:
            print("  " + " | ".join(row2))
        if static.get("top_tags"):
            tags_str = ", ".join(f"{t['tag']}({t['count']})" for t in static["top_tags"][:8])
            print(f"  Top tags: {tags_str}")
        if static.get("context_lora"):
            print(f"  Context LoRA: {static['context_lora']} @ {static.get('context_lora_strength','1.0')}")
        print()

    print("Category breakdown (activation):")
    print(f"  Style+Composition: {cat_totals['style_composition']/grand_total*100:5.1f}%  (double 0-7 + single 0)")
    print(f"  ↔ style↔identity:  {cat_totals['style_ident_overlap']/grand_total*100:5.1f}%  (single 1)")
    print(f"  Identity:          {cat_totals['identity']/grand_total*100:5.1f}%  (single 2-11)")
    print(f"  ↔ identity↔details:{cat_totals['ident_details_overlap']/grand_total*100:5.1f}%  (single 12-16)")
    print(f"  Details:           {cat_totals['details']/grand_total*100:5.1f}%  (single 17-23)")
    print()

    print("Most active blocks (activation):")
    for name, total in result.get_top_blocks(8):
        cat = _get_category(name)
        pct = total / grand_total * 100
        bar = "|" * min(40, int(pct * 4))
        print(f"  {_short_name(name):14s}  {pct:5.1f}%  {bar}  [{cat}]")

    if static.get("static_top_blocks"):
        print("\nTop weight-norm blocks (static):")
        for b in static["static_top_blocks"][:8]:
            bar = "|" * min(40, int(b["norm_pct"] / 2.5))
            print(f"  {b['name']:14s}  {b['norm_pct']:5.1f}%  {bar}  [{b['category']}]")

    print(f"{'='*60}\n")
