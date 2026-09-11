import math
import os
import logging

import torch
import comfy.sd
import comfy.model_management as mm
import comfy.utils
import folder_paths
from comfy.ldm.modules.attention import get_attention_function, attention_basic
from comfy.ldm.depth_anything_3 import preprocess as da3_preprocess

from .download_progress import download_url_with_progress
from comfy_extras.nodes_depth_anything_3 import DA3Inference, DA3Render


TAG = "crt-da3"

MODEL_OPTIONS = [
    "depth_anything_3_small.safetensors",
    "depth_anything_3_base.safetensors",
    "depth_anything_3_mono_large.safetensors",
    "depth_anything_3_metric_large.safetensors",
]

_ATTENTION_METHODS = [
    "comfy_kitchen_int8",
    "sage",
    "sage3",
    "flash",
    "xformers",
    "pytorch",
    "sub_quad",
    "split",
    "basic",
]


def _resolve_attention(method):
    if method == "basic":
        return attention_basic
    func = get_attention_function(method, None)
    if func is None:
        logging.warning(f"[{TAG}] Attention backend '{method}' is not installed; using ComfyUI default.")
    return func


def _geometry_estimation_dir():
    return os.path.join(folder_paths.models_dir, "geometry_estimation")


def _ensure_model(filename):
    target_dir = _geometry_estimation_dir()
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, filename)
    if os.path.exists(target):
        return target
    url = f"https://huggingface.co/Comfy-Org/Depth-Anything-3/resolve/main/geometry_estimation/{filename}"
    download_url_with_progress(
        url,
        target,
        label=filename,
        user_agent="CRT-DepthAnything3/1.0",
        console_prefix="CRT DepthAnything3",
    )
    return target


def _mp_to_resolution(megapixels, H, W):
    target_pixels = float(megapixels) * 1_000_000.0
    scale = math.sqrt(target_pixels / max(1.0, H * W))
    long_side = max(H, W) * scale
    res = int(round(long_side / 14.0)) * 14
    return min(max(res, 140), 2520)


def _unwrap_node_output(value):
    if hasattr(value, "args") and isinstance(value.args, tuple):
        if len(value.args) == 1:
            return value.args[0]
        return value.args
    return value


def _run_da3_mono(model, image, resolution, max_batch_size):
    B, H, W, _ = image.shape
    mm.load_model_gpu(model)
    diffusion = model.model.diffusion_model
    device = mm.get_torch_device()
    dtype = diffusion.dtype if diffusion.dtype is not None else torch.float32

    chunk = max_batch_size if max_batch_size and max_batch_size > 0 else B

    depths, confs, skies = [], [], []
    for start in range(0, B, chunk):
        batch = image[start:start + chunk].to(device)
        x = da3_preprocess.preprocess_image(batch, process_res=resolution, method="upper_bound_resize")
        x = x.to(dtype=dtype)
        with torch.no_grad():
            out = diffusion(x)

        depths.append(torch.nn.functional.interpolate(
            out["depth"].unsqueeze(1).float(), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(1).cpu())
        if "depth_conf" in out:
            confs.append(torch.nn.functional.interpolate(
                out["depth_conf"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu())
        if "sky" in out:
            skies.append(torch.nn.functional.interpolate(
                out["sky"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu())

    geometry = {
        "depth": torch.cat(depths, dim=0).contiguous(),
        "image": image[..., :3].cpu(),
        "mode": "mono",
    }
    if confs:
        geometry["confidence"] = torch.cat(confs, dim=0).contiguous()
    if skies:
        geometry["sky"] = torch.cat(skies, dim=0).contiguous()
    return geometry


def _run_da3_with_progress(model, image, resolution, mode_dict, max_batch_size=0):
    B = image.shape[0]
    pbar = comfy.utils.ProgressBar(B)
    if mode_dict["mode"] == "mono":
        geometry = _run_da3_mono(model, image, resolution, max_batch_size)
    else:
        geometry = DA3Inference.execute.__func__(
            DA3Inference, model, image, resolution, "upper_bound_resize", mode_dict,
        )
        geometry = _unwrap_node_output(geometry)
    pbar.update(B)
    return geometry


class CRT_DepthAnything3:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (MODEL_OPTIONS, {"default": MODEL_OPTIONS[0]}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "weight_dtype": (
                    ["default", "fp16", "bf16", "fp32"],
                    {"default": "default"},
                ),
                "image": ("IMAGE",),
                "megapixels": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.05, "max": 8.0, "step": 0.05},
                ),
                "mode": (["mono", "multiview"], {"default": "mono"}),
                "output": (
                    ["depth", "depth_colored", "sky_mask", "confidence"],
                    {"default": "depth"},
                ),
                "normalization": (
                    ["v2_style", "min_max", "raw"],
                    {"default": "v2_style"},
                ),
            },
            "optional": {
                "attention_method": (
                    _ATTENTION_METHODS,
                    {"default": "comfy_kitchen_int8", "tooltip": "Attention backend for the DINOv2 backbone. Falls back to the ComfyUI default if the selected backend is not installed."},
                ),
                "max_batch_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 256, "tooltip": "Max images per forward in mono mode (0 = all images in a single forward). Lower it to reduce peak VRAM."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "CRT/Depth"

    def execute(
        self,
        model_name,
        weight_dtype,
        image,
        megapixels,
        mode,
        output,
        normalization,
        keep_model_loaded,
        attention_method="comfy_kitchen_int8",
        max_batch_size=0,
    ):
        H, W = image.shape[1], image.shape[2]
        resolution = _mp_to_resolution(megapixels, H, W)

        model_path = _ensure_model(model_name)
        model_options = {}
        if weight_dtype == "fp16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16
        elif weight_dtype == "fp32":
            model_options["dtype"] = torch.float32

        da3_model = comfy.sd.load_diffusion_model(model_path, model_options=model_options)
        if da3_model is None:
            raise RuntimeError(f"[{TAG}] Failed to load DA3 model from: {model_path}")

        attention = _resolve_attention(attention_method)
        if attention is not None:
            da3_model.set_model_optimized_attention(attention)

        mode_dict = {"mode": mode}
        if mode == "multiview":
            mode_dict["ref_view_strategy"] = "saddle_balanced"
            mode_dict["pose_method"] = "cam_dec"

        geometry = _run_da3_with_progress(
            da3_model, image, resolution, mode_dict,
            max_batch_size=max_batch_size,
        )

        output_dict = {"output": output}
        if output in ("depth", "depth_colored"):
            output_dict["normalization"] = normalization
            output_dict["apply_sky_clip"] = False
        elif output in ("sky_mask", "confidence"):
            output_dict["colored"] = False

        result = DA3Render.execute.__func__(DA3Render, geometry, output_dict)
        result = _unwrap_node_output(result)

        if output == "depth" and normalization == "raw":
            result = 1.0 - result

        if not keep_model_loaded:
            mm.unload_model_and_clones(da3_model, unload_additional_models=True)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "CRT_DepthAnything3": CRT_DepthAnything3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_DepthAnything3": "Fast Depth Anything v3 (CRT)",
}
