import gc
import logging
import math
import os

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.sd
import comfy.utils
import folder_paths
from comfy.ldm.modules.attention import attention_basic, get_attention_function


TAG = "crt-marigold-v2"

QWEN_MODEL_REPO = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
QWEN_MODEL_FILE = "split_files/diffusion_models/qwen_image_edit_2509_fp8mixed.safetensors"
QWEN_VAE_REPO = "Comfy-Org/Qwen-Image_ComfyUI"
QWEN_VAE_FILE = "split_files/vae/qwen_image_vae.safetensors"
QWEN_VAE_NAME = "qwen_image_vae.safetensors"
MARIGOLD_REPO = "huawei-bayerlab/marigold-v2-0"

EMBED_PREFIX = "qwen_edit_2509_qwen_depth_realimg512"
TIMESTEP = 499.0 / 1000.0

DEPTH_CHECKPOINTS = [
    "Log-stage2",
    "Log-stage1",
    "Log-layered",
    "Uniform-base",
    "Disparity-base",
    "Disparity-layered",
    "Uniform-layered",
]

ATTENTION_METHODS = [
    "comfy_kitchen_int8",
    "sage",
    "flash",
    "xformers",
    "pytorch",
    "sub_quad",
    "split",
    "basic",
]

# Marigold's trainables are stored with diffusers VAE key names; the ComfyUI VAE
# is the same Wan-style architecture under different names.
_RESNET_KEYS = {
    "norm1.gamma": "residual.0.gamma",
    "conv1.weight": "residual.2.weight",
    "conv1.bias": "residual.2.bias",
    "norm2.gamma": "residual.3.gamma",
    "conv2.weight": "residual.6.weight",
    "conv2.bias": "residual.6.bias",
    "conv_shortcut.weight": "shortcut.weight",
    "conv_shortcut.bias": "shortcut.bias",
}
_UP_RESNET_BASE = (0, 4, 8, 12)
_UP_UPSAMPLER_BASE = (3, 7, 11)


def _assets_dir():
    path = os.path.join(folder_paths.models_dir, "marigold_v2")
    os.makedirs(path, exist_ok=True)
    return path


def _hf_download(repo_id, filename, local_dir):
    from huggingface_hub import hf_hub_download

    os.makedirs(local_dir, exist_ok=True)
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)


def _download_snapshot(repo_id, destination, allow_patterns):
    from huggingface_hub import snapshot_download

    os.makedirs(destination, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
        allow_patterns=allow_patterns,
        max_workers=4,
    )


def _ensure_qwen_model():
    target = os.path.join(_assets_dir(), QWEN_MODEL_FILE.replace("/", os.sep))
    if os.path.exists(target):
        return target
    print(f"[{TAG}] Downloading {QWEN_MODEL_FILE} (~20 GB, one time)...")
    return _hf_download(QWEN_MODEL_REPO, QWEN_MODEL_FILE, _assets_dir())


def _ensure_vae():
    existing = folder_paths.get_full_path("vae", QWEN_VAE_NAME)
    if existing is not None:
        return existing
    target = os.path.join(_assets_dir(), QWEN_VAE_FILE.replace("/", os.sep))
    if os.path.exists(target):
        return target
    print(f"[{TAG}] Downloading {QWEN_VAE_NAME} (~253 MB, one time)...")
    return _hf_download(QWEN_VAE_REPO, QWEN_VAE_FILE, _assets_dir())


def _ensure_embeddings():
    marigold = os.path.join(_assets_dir(), "Marigold-V2")
    embeds = os.path.join(marigold, "qwen_text_embeddings")
    needed = [f"{EMBED_PREFIX}_prompt_embeds.pt", f"{EMBED_PREFIX}_prompt_mask.pt"]
    if all(os.path.exists(os.path.join(embeds, name)) for name in needed):
        return embeds
    print(f"[{TAG}] Downloading Marigold V2 depth prompt embeddings...")
    _download_snapshot(
        MARIGOLD_REPO, marigold, [f"qwen_text_embeddings/{EMBED_PREFIX}_prompt_*"]
    )
    return embeds


def _ensure_checkpoint(model_name):
    marigold = os.path.join(_assets_dir(), "Marigold-V2")
    relative = f"depth/{model_name}/trainables.safetensors"
    target = os.path.join(marigold, "depth", model_name, "trainables.safetensors")
    if os.path.exists(target):
        return target
    print(f"[{TAG}] Downloading Marigold V2 depth/{model_name} checkpoint...")
    _download_snapshot(MARIGOLD_REPO, marigold, [relative])
    return target


def _map_vae_key(key):
    if key.startswith("decoder.conv_in."):
        return key.replace("decoder.conv_in.", "decoder.conv1.")
    if key.startswith("decoder.conv_out."):
        return key.replace("decoder.conv_out.", "decoder.head.2.")
    if key.startswith("decoder.norm_out."):
        return key.replace("decoder.norm_out.", "decoder.head.0.")
    if key.startswith("decoder.mid_block.resnets.0."):
        rest = key.split("decoder.mid_block.resnets.0.", 1)[1]
        return "decoder.middle.0." + _RESNET_KEYS.get(rest, rest)
    if key.startswith("decoder.mid_block.resnets.1."):
        rest = key.split("decoder.mid_block.resnets.1.", 1)[1]
        return "decoder.middle.2." + _RESNET_KEYS.get(rest, rest)
    if key.startswith("decoder.mid_block.attentions.0."):
        return key.replace("decoder.mid_block.attentions.0.", "decoder.middle.1.")
    if key.startswith("decoder.up_blocks."):
        parts = key.split(".")
        block = int(parts[2])
        kind = parts[3]
        rest = ".".join(parts[5:])
        if kind == "resnets":
            index = _UP_RESNET_BASE[block] + int(parts[4])
            return f"decoder.upsamples.{index}.{_RESNET_KEYS.get(rest, rest)}"
        if kind == "upsamplers":
            return f"decoder.upsamples.{_UP_UPSAMPLER_BASE[block]}.{rest}"
    if key.startswith("post_quant_conv."):
        return key.replace("post_quant_conv.", "conv2.")
    return None


def _build_vae(checkpoint):
    sd = comfy.utils.load_torch_file(_ensure_vae())
    for key, value in checkpoint.items():
        if not key.startswith("VAE."):
            continue
        native_key = _map_vae_key(key[len("VAE.") :])
        if native_key is not None:
            sd[native_key] = value
    vae = comfy.sd.VAE(sd=sd)
    vae.not_video = True
    return vae


def _load_prompt(embeds_dir):
    embeds = torch.load(
        os.path.join(embeds_dir, f"{EMBED_PREFIX}_prompt_embeds.pt"),
        map_location="cpu",
        weights_only=False,
    ).contiguous()
    mask = torch.load(
        os.path.join(embeds_dir, f"{EMBED_PREFIX}_prompt_mask.pt"),
        map_location="cpu",
        weights_only=False,
    ).contiguous()
    if mask.dtype != torch.bool:
        mask = mask > 0
    return embeds, mask


def _apply_attention(model, method):
    if method == "basic":
        attention = attention_basic
    else:
        attention = get_attention_function(method, None)
        if attention is None:
            logging.warning(
                f"[{TAG}] Attention backend '{method}' is not installed; using the ComfyUI default."
            )
            return
    model.set_model_optimized_attention(attention)


def _build(model_name, attention_method):
    model_path = _ensure_qwen_model()
    embeds_dir = _ensure_embeddings()
    checkpoint_path = _ensure_checkpoint(model_name)
    checkpoint = comfy.utils.load_torch_file(checkpoint_path)

    print(f"[{TAG}] Loading {model_name}...")
    model = comfy.sd.load_diffusion_model(model_path)
    if model is None:
        raise RuntimeError(f"[{TAG}] Failed to load Qwen-Image-Edit from {model_path}")

    lora = {
        key[len("Diffuser.") :]: value
        for key, value in checkpoint.items()
        if key.startswith("Diffuser.")
    }
    model, _ = comfy.sd.load_lora_for_models(model, None, lora, 1.0, 0.0)
    _apply_attention(model, attention_method)

    vae = _build_vae(checkpoint)
    embeds, mask = _load_prompt(embeds_dir)
    return model, vae, embeds, mask


def _mp_to_hw(megapixels, height, width):
    target = float(megapixels) * 1_000_000.0
    scale = math.sqrt(target / max(1.0, height * width))
    resized_h = max(16, int(round(height * scale / 16.0)) * 16)
    resized_w = max(16, int(round(width * scale / 16.0)) * 16)
    return resized_h, resized_w


def _run_step(model, latents, embeds, mask):
    device = latents.device
    batch_size = latents.shape[0]
    context = embeds[:1].expand(batch_size, *embeds.shape[1:]).to(
        device=device, dtype=torch.bfloat16
    )
    attention_mask = mask[:1].expand(batch_size, *mask.shape[1:]).to(
        device=device, dtype=torch.int64
    )
    timestep = torch.full((batch_size,), TIMESTEP, device=device, dtype=torch.bfloat16)
    prediction = model.model.diffusion_model(
        latents,
        timestep,
        context=context,
        attention_mask=attention_mask,
        transformer_options=model.model_options.get("transformer_options", {}),
    )
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
    return latents - prediction


def _normalize_depth(depth, mode):
    if mode == "raw":
        return depth
    low = depth.amin(dim=(-2, -1), keepdim=True)
    high = depth.amax(dim=(-2, -1), keepdim=True)
    normalized = ((depth - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
    return 1.0 - normalized


class CRT_MarigoldV2Depth:
    def __init__(self):
        self._cache_key = None
        self._cache = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (DEPTH_CHECKPOINTS, {"default": "Log-stage2"}),
                "image": ("IMAGE",),
                "megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05},
                ),
                "normalization": (["min_max", "raw"], {"default": "min_max"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "attention_method": (
                    ATTENTION_METHODS,
                    {
                        "default": "comfy_kitchen_int8",
                        "tooltip": "Attention backend for the Qwen-Image-Edit transformer. Falls back to the ComfyUI default if the selected backend is not installed.",
                    },
                ),
                "max_batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 64,
                        "tooltip": "Images per forward (0 = the whole batch). Marigold V2 needs a lot of VRAM at high megapixels; lower it if you run out.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "CRT/Depth"
    DESCRIPTION = """
Marigold V2 monocular depth estimation on ComfyUI's native Qwen-Image-Edit-2509 transformer.
https://huggingface.co/huawei-bayerlab/marigold-v2-0
Downloads the fp8 Qwen-Image-Edit base, the Qwen VAE and the selected depth checkpoint on first use.
"""

    def _release(self):
        if self._cache is not None:
            model = self._cache[0]
            self._cache = None
            self._cache_key = None
            try:
                mm.unload_model_and_clones(model, unload_additional_models=True)
            except Exception:
                pass
        gc.collect()
        mm.soft_empty_cache()

    def execute(
        self,
        model_name,
        image,
        megapixels,
        normalization,
        keep_model_loaded,
        attention_method="comfy_kitchen_int8",
        max_batch_size=1,
    ):
        if self._cache is not None and self._cache_key == model_name:
            model, vae, embeds, mask = self._cache
            _apply_attention(model, attention_method)
        else:
            self._release()
            model, vae, embeds, mask = _build(model_name, attention_method)
            self._cache = (model, vae, embeds, mask)
            self._cache_key = model_name

        mm.load_model_gpu(model)

        batch_size, height, width, _ = image.shape
        resized_h, resized_w = _mp_to_hw(megapixels, height, width)
        chunk = max_batch_size if max_batch_size and max_batch_size > 0 else batch_size
        device = mm.get_torch_device()
        pbar = comfy.utils.ProgressBar(batch_size)

        outputs = []
        with torch.no_grad():
            for start in range(0, batch_size, chunk):
                batch = image[start : start + chunk]
                pixels = comfy.utils.common_upscale(
                    batch.movedim(-1, 1), resized_w, resized_h, "lanczos", "disabled"
                )
                latents = vae.encode(pixels.movedim(1, -1))
                latents = model.model.process_latent_in(
                    latents.to(device=device, dtype=torch.float32)
                ).to(torch.bfloat16)
                latents = _run_step(model, latents, embeds, mask)
                latents = model.model.process_latent_out(latents)
                decoded = vae.decode(latents)
                depth = (
                    (decoded[:, 0].mean(dim=-1, keepdim=True) * 2.0 - 1.0)
                    .movedim(-1, 1)
                    .float()
                )
                depth = F.interpolate(
                    depth, size=(height, width), mode="bilinear", align_corners=False
                )
                outputs.append(depth)
                pbar.update(batch.shape[0])

        depth = torch.cat(outputs, dim=0)
        depth = _normalize_depth(depth, normalization)
        result = depth.movedim(1, -1).repeat(1, 1, 1, 3).contiguous()

        if not keep_model_loaded:
            self._release()

        return (result.to(mm.intermediate_device()).float(),)


NODE_CLASS_MAPPINGS = {
    "CRT_MarigoldV2Depth": CRT_MarigoldV2Depth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_MarigoldV2Depth": "Heavy Depth Marigold v2 (CRT)",
}
