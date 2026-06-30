"""Krea 2 LoRA training: full-model LoRA + flow-matching loss over a bucketed dataloader.

Trains on the RAW model. The LoRA wraps all 264 Linears (no layer-targeting presets yet — Krea2's
block semantics aren't mapped, so Identity/Style/Details presets come later). The base is frozen
(optionally fp8, QLoRA-style); only the LoRA trains in bf16. Uses Fizgig's bucketed multi-resolution
dataloader (same framework as Klein) over the krea2 latent/TE caches.
"""

import argparse
import gc
import json
import logging
import os
from multiprocessing import Value

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fizgig.dataset.config import (
    BlueprintGenerator,
    ConfigSanitizer,
    generate_dataset_group_by_blueprint,
    load_user_config,
)
from fizgig.krea2.utils import load_krea2_dit
from fizgig.krea2.sampling import gather_valid_text, prepare
from fizgig.networks.lora import create_network
from fizgig.training.metadata import ARCHITECTURE_KREA2
from fizgig.training.train_utils import LossRecorder

logger = logging.getLogger(__name__)


def load_dit_for_training(
    raw_path: str,
    *,
    network_dim: int = 32,
    network_alpha: float = 32,
    fp8_scaled: bool = True,
    blocks_to_swap: int = 0,
    gradient_checkpointing: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load the RAW DiT (frozen base, optionally fp8) and apply a trainable full-model LoRA."""
    loading_device = "cpu" if blocks_to_swap > 0 else device
    dit = load_krea2_dit(raw_path, device=device, dtype=dtype, fp8_scaled=fp8_scaled,
                         loading_device=loading_device)
    dit.requires_grad_(False)  # frozen base (QLoRA-style)
    if gradient_checkpointing:
        dit.enable_gradient_checkpointing()

    network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit)
    network.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    network.requires_grad_(True)
    network.to(device=device, dtype=dtype)
    return dit, network


def sample_shifted_timesteps(bsize: int, shift: float, device) -> torch.Tensor:
    """Flow-matching 'shift' timestep sampling: u~U(0,1), t = shift*u / (1 + (shift-1)*u)."""
    u = torch.rand(bsize, device=device)
    return shift * u / (1.0 + (shift - 1.0) * u)


def compute_loss(dit, latent, hidden_states, attention_mask, *, shift=2.5, dtype=torch.bfloat16):
    """Flow-matching training loss for Krea 2.

    latent:        (B, 16, h, w)         — cached Qwen-Image VAE latent
    hidden_states: (B, seq, layers, dim) — cached Qwen3-VL multi-layer stack
    attention_mask:(B, seq) bool         — cached validity mask
    """
    device = next(p for p in dit.parameters()).device
    B = latent.shape[0]
    latent = latent.to(device=device, dtype=dtype)

    noise = torch.randn_like(latent)
    t = sample_shifted_timesteps(B, shift, device)
    t_ = t.view(B, 1, 1, 1).to(dtype)
    noised = (1.0 - t_) * latent + t_ * noise
    target = noise - latent  # flow-matching velocity

    txt, txtmask = gather_valid_text(hidden_states.to(device=device, dtype=dtype), attention_mask.to(device))
    patch = dit.config.patch
    img_tokens, pos, mask = prepare(noised, txt.shape[1], patch, txtmask)
    target_tokens, _, _ = prepare(target, txt.shape[1], patch, txtmask)

    with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
        pred = dit(img=img_tokens, context=txt, t=t.to(dtype), pos=pos, mask=mask)
    return F.mse_loss(pred.float(), target_tokens.float())


class _Krea2Collator:
    """DataLoader batch_size is always 1 (the dataset batches internally by bucket)."""

    def __init__(self, shared_epoch, dataset):
        self.shared_epoch = shared_epoch
        self.dataset = dataset

    def __call__(self, examples):
        wi = torch.utils.data.get_worker_info()
        ds = wi.dataset if wi is not None else self.dataset
        ds.set_current_epoch(self.shared_epoch.value)
        return examples[0]


def _save_lora(network, path, network_dim, network_alpha, dtype):
    metadata = {
        "ss_network_module": "fizgig.krea2 (lora_unet, all-Linear)",
        "ss_network_dim": str(network_dim),
        "ss_network_alpha": str(network_alpha),
        "ss_architecture": ARCHITECTURE_KREA2,
    }
    network.save_weights(path, dtype, metadata)


# --- in-training previews (sample the fp8 Turbo with the live LoRA) -----------
def encode_sample_prompts(te_path, prompts, *, device="cuda"):
    """Pre-encode the sample prompts once (Qwen3-VL), freeing the encoder afterwards.
    Returns a list of (txt, txtmask) on CPU, fed straight to sampling.sample at preview time."""
    from fizgig.krea2.utils import load_krea2_text_encoder
    from fizgig.krea2 import sampling

    enc = load_krea2_text_encoder(te_path, dtype=torch.bfloat16, device=device)
    out = []
    for p in prompts:
        txt, txtmask, _, _ = sampling.encode_prompts(enc, [p], cfg=False)
        out.append((txt.cpu(), txtmask.cpu()))
    del enc
    torch.cuda.empty_cache()
    return out


def _read_sample_override(output_dir):
    """Live sample override written by the GUI to <output_dir>/.sample_override.json.

    Returns {prompt, seed, width, height} while active, else None. Model-agnostic fields
    only — the override's ref_image (Klein edit conditioning) is ignored, since Krea 2 isn't
    an edit model and generates previews from scratch."""
    path = os.path.join(output_dir, ".sample_override.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if str(d.get("prompt", "")).strip():
            return {"prompt": str(d["prompt"]).strip(),
                    "seed": int(d.get("seed", 1234)),
                    "width": int(d.get("width", 1024)),
                    "height": int(d.get("height", 1024))}
    except Exception:
        pass
    return None


def sample_previews(turbo_path, ae, encoded_prompts, lora_sd, out_dir, epoch, *,
                    output_name="krea2", steps=8, cfg_scale=1.0, width=512, height=512,
                    seed=42, device="cuda"):
    """Load the (clean) pre-quant fp8 Turbo, apply the current LoRA LIVE (no merge -> no grid),
    and render each pre-encoded prompt. Turbo is freed afterwards.

    Filenames follow the Fizgig samples-gallery pattern
    `{name}_e{epoch:06d}_{idx:02d}_{timestamp:14d}_{seed}.png` so the live preview gallery
    (which parses that exact format) picks them up — same as the Klein training path."""
    import datetime
    from fizgig.krea2.utils import load_krea2_dit
    from fizgig.networks.lora import create_network_from_weights
    from fizgig.krea2 import sampling

    turbo = load_krea2_dit(turbo_path, device=device, dtype=torch.bfloat16)  # prequant fp8 auto-detected
    net = create_network_from_weights(None, 1.0, lora_sd, None, turbo, for_inference=True)
    net.apply_to(text_encoders=None, unet=turbo, apply_text_encoder=False, apply_unet=True)
    # create_network_from_weights only builds the module STRUCTURE (sizes from dims/alphas);
    # the trained values must be loaded in, or the LoRA stays at its zero init (lora_up=0) and
    # contributes nothing — which made every epoch's preview identical. Mirrors the Klein path
    # (inference.py: apply_to -> load_state_dict(strict=False)).
    net.load_state_dict(lora_sd, strict=False)
    net.to(device=device, dtype=torch.bfloat16).eval()
    turbo.eval()
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 14-digit timestamp
    paths = []
    for i, (txt, txtmask) in enumerate(encoded_prompts):
        with torch.no_grad():
            imgs = sampling.sample(turbo, ae, txt, txtmask, untxt=None, untxtmask=None,
                                   device=device, dtype=torch.bfloat16, width=width, height=height,
                                   steps=steps, cfg_scale=cfg_scale, mu=1.15, seed=seed + i)
        p = os.path.join(out_dir, f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{seed + i}.png")
        imgs[0].save(p)
        paths.append(p)
    del turbo, net
    torch.cuda.empty_cache()
    return paths


def train_krea2(
    raw_path: str,
    dataset_config: str,
    output_dir: str,
    output_name: str,
    *,
    network_dim: int = 32,
    network_alpha: float = 32,
    learning_rate: float = 1e-4,
    max_train_epochs: int = 10,
    save_every_n_epochs: int = 0,
    fp8_scaled: bool = True,
    blocks_to_swap: int = 0,
    shift: float = 2.5,
    seed: int = 42,
    # in-training previews (sample the fp8 Turbo with the live LoRA)
    sample_prompts: list = None,
    turbo_path: str = None,
    vae_path: str = None,
    te_path: str = None,
    sample_every_n_epochs: int = 0,
    sample_width: int = 512,
    sample_height: int = 512,
    sample_steps: int = 8,
    sample_seed: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Native Krea 2 LoRA training: bucketed multi-resolution dataloader over the krea2 caches ->
    flow-matching loss -> AdamW -> save a ComfyUI-compatible LoRA. In-training Turbo previews +
    GUI wiring are layered on elsewhere."""
    torch.manual_seed(seed)

    shared_epoch = Value("i", 0)
    user_config = load_user_config(dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        user_config, argparse.Namespace(), architecture=ARCHITECTURE_KREA2)
    group = generate_dataset_group_by_blueprint(
        blueprint.dataset_group, training=True, num_timestep_buckets=None, shared_epoch=shared_epoch)
    if group.num_train_items == 0:
        raise RuntimeError("No training items — run the krea2 cache scripts first.")
    logger.info(f"Krea 2 training: {group.num_train_items} items, {max_train_epochs} epochs")

    # Preview setup: pre-encode prompts (frees the 8GB encoder) + load the VAE BEFORE the RAW DiT,
    # so the encoder never coexists with the resident base.
    do_previews = bool(sample_every_n_epochs and sample_prompts and turbo_path and vae_path and te_path)
    encoded_prompts = sample_ae = sample_dir = None
    if do_previews:
        from fizgig.krea2.vae_loader import load_vae
        logger.info(f"pre-encoding {len(sample_prompts)} sample prompt(s)...")
        encoded_prompts = encode_sample_prompts(te_path, sample_prompts, device=device)
        sample_ae = load_vae(vae_path, input_channels=3, device="cpu", disable_mmap=True)
        sample_dir = os.path.join(output_dir, "sample")

    dit, network = load_dit_for_training(
        raw_path, network_dim=network_dim, network_alpha=network_alpha,
        fp8_scaled=fp8_scaled, blocks_to_swap=blocks_to_swap, device=device, dtype=dtype)
    if blocks_to_swap > 0:
        from fizgig.krea2.offloading import BlockSwapConfig
        dit.enable_block_swap(blocks_to_swap, BlockSwapConfig(torch.device(device), supports_backward=True))
        dit.move_to_device_except_swap_blocks(torch.device(device))
        dit.switch_block_swap_for_training()
    dit.train()
    network.train()
    network.requires_grad_(True)

    params = list(network.get_trainable_params())
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(params, lr=learning_rate)
        logger.info("optimizer: AdamW8bit")
    except Exception:
        optimizer = torch.optim.AdamW(params, lr=learning_rate)
        logger.info("optimizer: AdamW (bitsandbytes unavailable)")

    collator = _Krea2Collator(shared_epoch, group)
    loader = DataLoader(group, batch_size=1, shuffle=True, collate_fn=collator, num_workers=0)

    os.makedirs(output_dir, exist_ok=True)
    global_step = 0
    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = group.num_train_items
    # Progress + loss display exactly as Klein: one continuous tqdm bar over all steps with
    # a smoothed avr_loss in the postfix (the raw per-step loss is very noisy — batch size 1
    # plus a random flow-matching timestep each step — so the moving average is the signal).
    loss_recorder = LossRecorder()
    progress_bar = tqdm(total=steps_per_epoch * max_train_epochs, desc="steps", smoothing=0)
    for epoch in range(max_train_epochs):
        shared_epoch.value = epoch + 1
        for i, batch in enumerate(loader):
            loss = compute_loss(dit, batch["latents"], batch["hidden_states"], batch["attention_mask"],
                                shift=shift, dtype=dtype)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            loss_recorder.add(epoch=epoch, step=i, loss=loss.item())
            progress_bar.set_postfix(avr_loss=f"{loss_recorder.moving_average:.4f}")
            progress_bar.update(1)
        logger.info(f"epoch {epoch + 1}/{max_train_epochs}  avr_loss={loss_recorder.moving_average:.4f}  step={global_step}")

        if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0 and (epoch + 1) < max_train_epochs:
            _save_lora(network, os.path.join(output_dir, f"{output_name}-{epoch + 1:06d}.safetensors"),
                       network_dim, network_alpha, dtype)

        if do_previews and (epoch + 1) % sample_every_n_epochs == 0:
            from safetensors.torch import load_file
            tmp = os.path.join(output_dir, "_sample_lora.safetensors")
            _save_lora(network, tmp, network_dim, network_alpha, dtype)
            logger.info(f"rendering previews (epoch {epoch + 1}) on the fp8 Turbo...")
            # The preview loads the fp8 Turbo (~13 GB) on top of the resident training DiT
            # (~14 GB fp8) + the VAE — two full models won't fit (OOMs ~30 GB on a 32 GB card).
            # Park the training DiT on CPU for the preview, then restore it (and its block-swap
            # placement) before the next epoch. Costs one CPU<->GPU round-trip per preview.
            dit.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            try:
                # Live sample override (GUI status-bar panel) — model-agnostic prompt/seed/res
                # for the next preview. Encoded here (after the training DiT is on CPU) so the
                # text encoder has room. No override -> the configured pre-encoded prompts.
                ov = _read_sample_override(output_dir)
                if ov:
                    logger.info(f"[sample override] active — '{ov['prompt'][:60]}' "
                                f"seed={ov['seed']} {ov['width']}x{ov['height']}")
                    prev_enc = encode_sample_prompts(te_path, [ov["prompt"]], device=device)
                    prev_w, prev_h, prev_seed = ov["width"], ov["height"], ov["seed"]
                else:
                    prev_enc, prev_w, prev_h, prev_seed = encoded_prompts, sample_width, sample_height, sample_seed
                sample_previews(turbo_path, sample_ae, prev_enc, load_file(tmp), sample_dir, epoch + 1,
                                output_name=output_name, steps=sample_steps, width=prev_w,
                                height=prev_h, seed=prev_seed, device=device)
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                if blocks_to_swap > 0:
                    # Re-establish the training placement (non-swap blocks -> GPU, swap blocks -> CPU).
                    dit.move_to_device_except_swap_blocks(torch.device(device))
                    dit.switch_block_swap_for_training()
                else:
                    dit.to(device)
            dit.train()
            network.train()

    progress_bar.close()
    out = os.path.join(output_dir, f"{output_name}.safetensors")
    _save_lora(network, out, network_dim, network_alpha, dtype)
    logger.info(f"saved final LoRA -> {out}")
    return out
