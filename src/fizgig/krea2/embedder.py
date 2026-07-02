"""Krea 2 (K2) text encoder: Qwen3-VL-4B conditioner.

Returns the stacked selected hidden states (b, seq, num_select_layers, dim) plus the
attention mask; the layerwise fusion lives inside the DiT (TextFusionTransformer), so
the raw stack is what gets cached during training.

Loading follows musubi conventions (cf. qwen_image's load_qwen2_5_vl): the model config
is vendored here so it is built without fetching config.json from the Hub, weights are
loaded directly from a local safetensors file (ComfyUI-style `model.`/`visual.` keys are
accepted as well as the official HF layout), and only the tokenizer is still pulled by
repo id. This lets K2 share the same Qwen3-VL-4B weights a user already has for ComfyUI,
instead of requiring a separate transformers/Diffusers checkpoint.
"""

import logging
from dataclasses import dataclass

import torch
from accelerate import init_empty_weights
from torch import Tensor
from transformers import (
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from fizgig.krea2.safetensors_utils import load_split_weights

logger = logging.getLogger(__name__)


# Only the tokenizer is still fetched by repo id (small, HF-cached after first use).
QWEN3_VL_4B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"

# Vendored copy of the Qwen3-VL-4B-Instruct config.json so the text encoder is built
# without fetching the config from the Hugging Face Hub. Qwen3-VL is natively supported by
# transformers (no auto_map / remote code), so Qwen3VLConfig.from_dict reproduces
# AutoConfig.from_pretrained exactly. Mirror upstream config.json if Qwen ever revises it.
QWEN3_VL_4B_INSTRUCT_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000,
        "tie_word_embeddings": True,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": True,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}


@dataclass
class TextEncoderConfig:
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID


def _convert_comfyui_qwen3vl_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map a ComfyUI-style (bare ``model.`` / ``visual.``) Qwen3-VL state dict onto the HF
    ``Qwen3VLForConditionalGeneration`` layout. Official HF checkpoints already use the
    ``model.language_model.`` / ``model.visual.`` layout and pass through unchanged.
    """
    converted: dict[str, Tensor] = {}
    for key, value in sd.items():
        if key.startswith("model.language_model.") or key.startswith("model.visual."):
            new_key = key
        elif key.startswith("visual."):
            new_key = "model.visual." + key[len("visual.") :]
        elif key.startswith("language_model."):
            new_key = "model." + key
        elif key.startswith("model."):
            new_key = "model.language_model." + key[len("model.") :]
        else:
            new_key = key
        converted[new_key] = value
    return converted


def _load_qwen3_vl_model(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    disable_mmap: bool = True,
) -> Qwen3VLForConditionalGeneration:
    """Build Qwen3-VL-4B from the vendored config and load weights from a local safetensors."""
    config = Qwen3VLConfig.from_dict(QWEN3_VL_4B_INSTRUCT_CONFIG)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)

    logger.info(f"Loading Krea 2 text encoder (Qwen3-VL) weights from {model_path}")
    sd = load_split_weights(model_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    sd = _convert_comfyui_qwen3vl_state_dict(sd)

    info = model.load_state_dict(sd, strict=False, assign=True)
    # Qwen3-VL-4B ties the LM head to the input embeddings (tie_word_embeddings=true), so the
    # checkpoint omits lm_head.weight; re-tie after loading to materialize it.
    model.tie_weights()

    unexpected = list(info.unexpected_keys)
    missing = [k for k in info.missing_keys if k != "lm_head.weight"]
    if unexpected or missing:
        raise RuntimeError(
            f"Qwen3-VL text encoder checkpoint did not match the model: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    model.to(device)
    if dtype is not None:
        model.to(dtype)
    return model.eval().requires_grad_(False)


def load_qwen3_vl_conditioner(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple[int, ...] = TextEncoderConfig.select_layers,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
    disable_mmap: bool = True,
) -> "Qwen3VLConditioner":
    """Load the Qwen3-VL-4B conditioner used by K2: weights from ``model_path`` (safetensors),
    tokenizer from ``tokenizer_repo`` (Hub id or local dir)."""
    qwen = _load_qwen3_vl_model(model_path, dtype=dtype, device=device, disable_mmap=disable_mmap)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, max_length=max_length)
    processor = Qwen2TokenizerFast.from_pretrained(tokenizer_repo, max_length=max_length)
    conditioner = Qwen3VLConditioner(qwen, tokenizer, processor, max_length=max_length,
                                     select_layers=select_layers, tokenizer_repo=tokenizer_repo)
    return conditioner.eval().requires_grad_(False)


CAPTION_INSTRUCTION = (
    "Write one factual training caption for this image as a single sentence. Describe the subject, "
    "their pose and clothing, the camera viewpoint (e.g. 'viewed from behind', 'side profile', "
    "'close-up'), whether the face is visible, and the setting. State only what is visible — no "
    "speculation, no names, no style commentary."
)

# Second-attempt instruction: if the standard caption didn't unstick the image, the miss is
# probably something salient the short caption skipped — go exhaustive so every visible element
# that could contradict the conditioning gets named.
DETAILED_CAPTION_INSTRUCTION = (
    "Write a detailed factual training caption for this image, 2-4 sentences. Cover: the subject "
    "and exactly how much of them is visible (state the camera viewpoint and explicitly whether "
    "the face is visible or hidden), their pose and body position, every visible clothing item "
    "with colors, hair style and color, any objects they hold or touch, anything partially "
    "blocking or cropping the subject, the lighting, and the background/setting with its main "
    "objects. State only what is visible — no speculation, no names, no style commentary."
)


def generate_caption(conditioner: "Qwen3VLConditioner", image_path: str, *,
                     max_new_tokens: int = 120, megapixels: float = 1.0,
                     detailed: bool = False, seed: int = None) -> str:
    """Caption an image with the SAME Qwen3-VL the trainer conditions on (its LM head is
    legitimately tied to the embeddings — unlike Klein's stripped Qwen3-8B — so generation is
    real). Used by auto-recaption to rewrite a stuck image's caption from what's actually in it,
    with an instruction tuned to Peter's captioning doctrine: name the viewpoint / visibility.

    Decoding is SAMPLED with a random seed (seed=None) so repeated attempts on the same image get
    fresh phrasings instead of the identical greedy caption — attempt 2 varies by wording as well
    as by instruction. Sampling uses the global torch RNG, so the state is saved and restored
    around the call: caption generation must never perturb the training noise stream."""
    import random as _random
    from PIL import Image

    proc = conditioner._get_image_processor()
    im = conditioner._cap_image(Image.open(image_path), megapixels)
    instruction = DETAILED_CAPTION_INSTRUCTION if detailed else CAPTION_INSTRUCTION
    if detailed:
        max_new_tokens = max(max_new_tokens, 240)
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": instruction}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[prompt], images=[im], return_tensors="pt").to(conditioner.qwen.device)

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed if seed is not None else _random.randint(1, 2**31 - 1))
        with torch.no_grad():
            out = conditioner.qwen.generate(**inputs, max_new_tokens=max_new_tokens,
                                            do_sample=True, temperature=0.7, top_p=0.9)
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return " ".join(text.split()).strip()


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(
        self,
        qwen: Qwen3VLForConditionalGeneration,
        tokenizer,
        processor,
        max_length: int = 512,
        select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
        tokenizer_repo: str | None = None,
    ):
        super().__init__()
        self.qwen = qwen.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.processor = processor
        self.tokenizer_repo = tokenizer_repo
        self._image_processor = None  # lazily-loaded full Qwen3-VL processor (for image refs)
        self.max_length = max_length
        self.select_layers = select_layers
        self.system_descriptor = ("Describe the image by detailing the color, shape, size, texture, "
                                  "quantity, text, spatial relationships of the objects and background:")
        self.prompt_template_encode_prefix = "<|im_start|>system\n" + self.system_descriptor + "<|im_end|>\n<|im_start|>user\n"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_suffix_start_idx = 5

    def forward(self, text: list[str], images: list | None = None,
                vision_megapixels: float = 1.0) -> tuple[Tensor, Tensor]:
        """Encode prompts to the K2 multi-layer hidden stack + mask.

        `images`, when given, is a per-prompt list (same length as `text`); each entry is a
        list of PIL.Image references (or None). When any prompt has references they are fed
        through Qwen3-VL's *vision* path under the same descriptor template, so the conditioning
        becomes "visually aware" of the image (a prompt-from-a-picture effect — Krea 2's DiT has
        no reference-latent slot, so this is the only reference mechanism). Requires the bf16 TE:
        ComfyUI's Qwen3-VL vision tower can't run in fp8.
        """
        has_imgs = bool(images) and any(images[i] for i in range(min(len(images), len(text))))
        if has_imgs:
            return self._forward_with_images(text, images, vision_megapixels)
        return self._forward_text(text)

    def _forward_text(self, text: list[str]) -> tuple[Tensor, Tensor]:
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        suffix_inputs = self.processor(text=suffix_text, return_tensors="pt").to(self.qwen.device, non_blocking=True)
        suffix_ids, suffix_mask = (
            suffix_inputs["input_ids"],
            suffix_inputs["attention_mask"].bool(),
        )

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                max_length=self.max_length + prefix_idx - self.prompt_template_encode_suffix_start_idx,
                return_tensors="pt",
            ).to(self.qwen.device, non_blocking=True)
            input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
            mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)
            states = self.qwen(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)

            hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
            hiddens = hiddens[:, prefix_idx:]
            mask = mask[:, prefix_idx:]

            return hiddens, mask

    def _get_image_processor(self):
        """Lazily load the full Qwen3-VL processor (text + image). Only needed for image refs,
        so text-only training never pays for it."""
        if self._image_processor is None:
            from transformers import AutoProcessor
            repo = self.tokenizer_repo or QWEN3_VL_4B_INSTRUCT_REPO_ID
            self._image_processor = AutoProcessor.from_pretrained(repo)
        return self._image_processor

    @staticmethod
    def _cap_image(im, megapixels: float):
        """RGB + downscale an image so its pixel area is <= megapixels (never upscale)."""
        from PIL import Image
        im = im.convert("RGB")
        cap = int(megapixels * 1024 * 1024)
        w, h = im.size
        if w * h > cap and w > 0 and h > 0:
            scale = (cap / (w * h)) ** 0.5
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        return im

    def _forward_with_images(self, text, images, vision_megapixels) -> tuple[Tensor, Tensor]:
        """Vision-aware encode: build the descriptor template with vision placeholders, run the
        Qwen3-VL processor (text + pixel_values), and extract the same select-layer stack.

        Mirrors the ComfyUI Text-Encode-(Krea2) node: forced Krea 2 descriptor system prompt,
        image tokens in the user turn, vision_megapixels as a downscale cap. The system prefix
        (start_idx tokens) is trimmed exactly as in the text path.
        """
        proc = self._get_image_processor()
        prefix_idx = self.prompt_template_encode_start_idx
        full_texts, flat_images = [], []
        for i, prompt in enumerate(text):
            imgs = (images[i] if images and i < len(images) and images[i] else [])
            imgs = [self._cap_image(im, vision_megapixels) for im in imgs]
            vis = "".join("<|vision_start|><|image_pad|><|vision_end|>" for _ in imgs)
            full_texts.append(self.prompt_template_encode_prefix + vis + prompt
                              + self.prompt_template_encode_suffix)
            flat_images.extend(imgs)

        with torch.no_grad():
            inputs = proc(text=full_texts, images=flat_images or None,
                          padding=True, return_tensors="pt").to(self.qwen.device)
            states = self.qwen(**inputs, output_hidden_states=True)
            hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
            mask = inputs["attention_mask"].bool()
            # Trim the system descriptor prefix (same fixed prefix as the text path; the image
            # tokens live in the user turn, after it).
            hiddens = hiddens[:, prefix_idx:]
            mask = mask[:, prefix_idx:]
            return hiddens, mask
