"""Optimizer selection, shared by the trainers.

Krea 2 hardcoded `bnb.optim.AdamW8bit` — a good default, but the only choice, which is the one
place the community comparison against OneTrainer (~40 optimizers) landed a fair hit.

Two things matter for a LoRA and pull against the "more optimizers is better" instinct:

* **Optimizer state is tiny here.** Krea 2 trains 264 small factors, so AdamW's two moments cost
  ~tens of MB against a 13-19 GB base. Choosing an 8-bit or factored optimizer to save memory is
  nearly pointless — the base dominates. What the choice actually buys is *update behaviour*.
* **Learning rates are not comparable across families.** Lion's update is a sign, so it wants
  roughly a tenth of AdamW's LR; Prodigy estimates the LR itself and wants `lr=1.0`. Handing
  someone a dropdown without saying that is how you produce a fried LoRA and a bug report, so
  `create_optimizer` warns loudly on the record when the LR looks wrong for the family.

Free-form `module.path.ClassName` is accepted too, so a user who pip-installs something exotic
does not need a Fizgig release to use it.
"""

from __future__ import annotations

import ast
import importlib
import logging

import torch

logger = logging.getLogger(__name__)


# name -> (import to test, one-line description shown in the GUI/CLI help)
_CATALOG = {
    "adamw8bit":          ("bitsandbytes", "AdamW, 8-bit state (default — the validated recipe)"),
    "adamw":              (None,           "AdamW, fp32 state, CUDA-fused where available"),
    "pagedadamw8bit":     ("bitsandbytes", "AdamW8bit that pages state to CPU under pressure"),
    "ademamix8bit":       ("bitsandbytes", "AdEMAMix — second slow EMA, aimed at long runs"),
    "pagedademamix8bit":  ("bitsandbytes", "AdEMAMix8bit with CPU paging"),
    "lion8bit":           ("bitsandbytes", "Lion — sign updates; use ~1/10 the AdamW LR"),
    "adafactor":          (None,           "Adafactor — factored state, lowest memory"),
    "prodigy":            ("prodigyopt",   "Prodigy — estimates the LR itself; set LR to 1.0"),
    "came":               ("came_pytorch", "CAME — confidence-guided Adafactor variant"),
}

DEFAULT_OPTIMIZER = "adamw8bit"


def available_optimizers() -> list[str]:
    """Catalog entries whose backing package is actually importable on this machine."""
    out = []
    for name, (module, _desc) in _CATALOG.items():
        if module is None:
            out.append(name)
            continue
        try:
            importlib.import_module(module)
            out.append(name)
        except Exception:
            pass
    return out


def describe(name: str) -> str:
    return _CATALOG.get(name.lower(), (None, "custom optimizer"))[1]


def parse_optimizer_args(raw: str) -> dict:
    """`"weight_decay=0.01 betas=0.9,0.99"` -> `{"weight_decay": 0.01, "betas": (0.9, 0.99)}`.

    Values go through `ast.literal_eval`, so tuples, bools and None all survive; anything that
    isn't a literal is kept as a plain string rather than being executed.
    """
    kwargs = {}
    for tok in (raw or "").split():
        if "=" not in tok:
            raise ValueError(f"optimizer arg {tok!r} is not key=value")
        key, value = tok.split("=", 1)
        try:
            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kwargs[key] = value
    return kwargs


def _bnb(cls_name: str):
    import bitsandbytes as bnb
    return getattr(bnb.optim, cls_name)


def _warn_lr(name: str, lr: float) -> None:
    """A wrong-family LR is silent at step 1 and obvious only hours later. Say it now."""
    if name == "lion8bit" and lr > 5e-5:
        logger.warning("[optimizer] Lion applies the SIGN of the update, so it needs roughly a "
                       "TENTH of an AdamW LR. %.2e will likely overbake — try %.2e.", lr, lr / 10)
    elif name == "prodigy" and lr < 0.5:
        logger.warning("[optimizer] Prodigy estimates the LR itself and expects lr=1.0 as a "
                       "multiplier. At %.2e it will barely train.", lr)
    elif name not in ("lion8bit", "prodigy") and lr > 1e-2:
        logger.warning("[optimizer] LR %.2e is very high for %s.", lr, name)


def create_optimizer(name: str, params, lr: float, args_str: str = "") -> tuple:
    """Build an optimizer. Returns `(optimizer, label)`; the label goes into LoRA metadata.

    Falls back to plain AdamW if the requested one cannot be constructed — a training run should
    not die at minute one over a dropdown, but the substitution is logged as a warning, never
    silently.
    """
    name = (name or DEFAULT_OPTIMIZER).strip()
    kwargs = parse_optimizer_args(args_str)
    key = name.lower()
    _warn_lr(key, lr)

    try:
        if key == "adamw8bit":
            opt = _bnb("AdamW8bit")(params, lr=lr, **kwargs)
        elif key == "pagedadamw8bit":
            opt = _bnb("PagedAdamW8bit")(params, lr=lr, **kwargs)
        elif key == "ademamix8bit":
            opt = _bnb("AdEMAMix8bit")(params, lr=lr, **kwargs)
        elif key == "pagedademamix8bit":
            opt = _bnb("PagedAdEMAMix8bit")(params, lr=lr, **kwargs)
        elif key == "lion8bit":
            opt = _bnb("Lion8bit")(params, lr=lr, **kwargs)
        elif key == "adamw":
            # Fused AdamW is one CUDA kernel over all 264 factors instead of a Python loop —
            # the "CUDA optimizer" the comparison thread was pointing at. Requires every param
            # on CUDA and floating point, which LoRA factors are.
            kwargs.setdefault("fused", torch.cuda.is_available())
            opt = torch.optim.AdamW(params, lr=lr, **kwargs)
        elif key == "adafactor":
            opt = torch.optim.Adafactor(params, lr=lr, **kwargs)
        elif key == "prodigy":
            from prodigyopt import Prodigy
            opt = Prodigy(params, lr=lr, **kwargs)
        elif key == "came":
            from came_pytorch import CAME
            opt = CAME(params, lr=lr, **kwargs)
        elif "." in name:
            module_path, cls_name = name.rsplit(".", 1)
            opt = getattr(importlib.import_module(module_path), cls_name)(params, lr=lr, **kwargs)
        else:
            raise ValueError(f"unknown optimizer {name!r} — use one of {available_optimizers()} "
                             "or a full module.path.ClassName")
    except Exception as e:
        logger.warning("[optimizer] could not create %s (%s) — falling back to AdamW", name, e)
        return torch.optim.AdamW(params, lr=lr), "adamw (fallback)"

    label = name + (f"({args_str.strip()})" if args_str.strip() else "")
    logger.info("optimizer: %s — %s", label, describe(key))
    return opt, label
