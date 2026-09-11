import importlib
import json
import logging
import os
import sys
import comfy.model_management as model_management
import comfy.clip_vision
import comfy.sd
import comfy.utils
import folder_paths
import torch
from comfy.cli_args import args
from comfy.ldm.modules.attention import (
    attention_pytorch,
    wrap_attn,
    get_attention_function,
    COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE,
)

from .download_progress import download_url_with_progress

from .H3_Fun_ControlNet import load_h3_fun_control


TAG = "crt-autodl"
SAGE_ATTENTION_MODES = [
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
]

_ATTENTION_METHODS = ["disabled", "pytorch attention"] + SAGE_ATTENTION_MODES
if COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE:
    _ATTENTION_METHODS.insert(0, "comfy kitchen attention")
ATTENTION_METHODS = _ATTENTION_METHODS


MODELS = {
    "zimage_model": {
        "folder": "diffusion_models",
        "filename": "z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors",
        "url": "https://huggingface.co/Kijai/Z-Image_comfy_fp8_scaled/resolve/main/z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors",
    },
    "zimage_vae": {
        "folder": "vae",
        "filename": "ae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
    },
    "zimage_clip": {
        "folder": "text_encoders",
        "filename": "qwen_3_4b_fp8_mixed.safetensors",
        "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
    },
    "krea2_turbo_model": {
        "folder": "diffusion_models",
        "filename": "krea2_turbo_fp8_scaled.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors",
    },
    "krea2_raw_model": {
        "folder": "diffusion_models",
        "filename": "krea2_raw_fp8_scaled.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_raw_fp8_scaled.safetensors",
    },
    "krea2_clip": {
        "folder": "text_encoders",
        "filename": "qwen3vl_4b_fp8_scaled.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
    },
    "krea2_vae": {
        "folder": "vae",
        "filename": "qwen_image_vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors",
    },
    "fluxklein_vae": {
        "folder": "vae",
        "filename": "flux2-vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
    },
    "fluxklein_model": {
        "folder": "diffusion_models",
        "filename": "flux-2-klein-9b-fp8.safetensors",
        "url": "https://huggingface.co/PGCRYPT/OB_FK/resolve/main/OB_FK.safetensors",
        "alternate_filenames": ["OB_FK.safetensors"],
    },
    "fluxklein_clip": {
        "folder": "text_encoders",
        "filename": "qwen_3_8b_fp8mixed.safetensors",
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
    },
    "fluxklein_hdri_lora": {
        "folder": "loras",
        "filename": "Klein_9B - HDRI_360_panoramic.safetensors",
        "url": "https://huggingface.co/PGCRYPT/Flux2Klein_9B-HDRI/resolve/main/Klein_9B%20-%20HDRI_360_panoramic.safetensors",
    },
    "ernie_turbo_model": {
        "folder": "diffusion_models",
        "filename": "ernie-image-turbo-fp8.safetensors",
        "url": "https://huggingface.co/Bedovyy/ERNIE-Image-Quantized/resolve/main/ernie-image-turbo-fp8.safetensors",
    },
    "ernie_turbo_nvfp4_model": {
        "folder": "diffusion_models",
        "filename": "ernie-image-turbo-nvfp4.safetensors",
        "url": "https://huggingface.co/Bedovyy/ERNIE-Image-Quantized/resolve/main/ernie-image-turbo-nvfp4.safetensors",
    },
    "ernie_model": {
        "folder": "diffusion_models",
        "filename": "ernie-image-fp8.safetensors",
        "url": "https://huggingface.co/Bedovyy/ERNIE-Image-Quantized/resolve/main/ernie-image-fp8.safetensors",
    },
    "ernie_turbo_clip": {
        "folder": "text_encoders",
        "filename": "ministral-3-3b.safetensors",
        "url": "https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/82d237fcf02a10b75154717487d07a724a25dc5b/text_encoders/ministral-3-3b.safetensors",
    },
    "ernie_turbo_vae": {
        "folder": "vae",
        "filename": "flux2-vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
    },
    "chronoedit_model": {
        "folder": "diffusion_models",
        "filename": "ChronoEdit_fp8_e4m3fn_scaled_VAI.safetensors",
        "url": "https://huggingface.co/vantagewithai/ChronoEdit-fp8-scaled/resolve/main/ChronoEdit_fp8_e4m3fn_scaled_VAI.safetensors",
    },
    "chronoedit_distill_lora": {
        "folder": "loras",
        "filename": "chronoedit_distill_lora.safetensors",
        "url": "https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers/resolve/main/lora/chronoedit_distill_lora.safetensors",
    },
    "chronoedit_upscaler_lora": {
        "folder": "loras",
        "filename": "upsample_lora_diffusers.safetensors",
        "url": "https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers-Upscaler-Lora/resolve/main/upsample_lora_diffusers.safetensors",
    },
    "chronoedit_vae": {
        "folder": "vae",
        "filename": "wan_2.1_vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
    },
    "chronoedit_clip": {
        "folder": "text_encoders",
        "filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    },
    "chronoedit_clip_vision": {
        "folder": "clip_vision",
        "filename": "clip_vision_h.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors",
    },
    # MiniMax H3
    "minimax_h3_model_fl2va": {
        "folder": "diffusion_models",
        "filename": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    },
    "minimax_h3_model_ref2va": {
        "folder": "diffusion_models",
        "filename": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    },
    "minimax_h3_model_fl2va_w4a8": {
        "folder": "diffusion_models",
        "filename": "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
        "url": "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    },
    "minimax_h3_model_ref2va_w4a8": {
        "folder": "diffusion_models",
        "filename": "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors",
        "url": "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_ref2va_pruned_w4a8_mixed.safetensors",
    },
    "minimax_h3_clip_int8": {
        "folder": "text_encoders",
        "filename": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    },
    "minimax_h3_clip_nvfp4": {
        "folder": "text_encoders",
        "filename": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    },
    "minimax_h3_fl2va_turbo_lora": {
        "folder": "loras",
        "filename": "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
        "url": "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
        "alternate_filenames": [
            "MiniMaxH3/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors",
        ],
    },
    "minimax_h3_ref2va_turbo_lora": {
        "folder": "loras",
        "filename": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        "url": "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        "alternate_filenames": [
            "MiniMaxH3/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        ],
    },
    "minimax_h3_audio_vae": {
        "folder": "vae",
        "filename": "minimax_h3_audio_vae_fp32.safetensors",
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors",
    },
    "minimax_h3_video_vae": {
        "folder": "vae",
        "filename": "minimax_h3_video_vae_fp16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors",
    },
    "minimax_h3_fun_controlnet": {
        "folder": "controlnet",
        "filename": "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors",
        "url": "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/controlnet/minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors",
    },
    # LTX2.5
    "ltx25_model_32gb": {
        "folder": "diffusion_models",
        "filename": "LTX25-distilled-DiT-comfy-int8.safetensors",
        "url": "https://huggingface.co/joeygambino/LTX-2.5-Quantized/resolve/main/LTX25-distilled-DiT-comfy-int8.safetensors",
    },
    "ltx25_model_24gb": {
        "folder": "diffusion_models",
        "filename": "LTX25-distilled-DiT-comfy-mix4x8-17GB.safetensors",
        "url": "https://huggingface.co/joeygambino/LTX-2.5-Quantized/resolve/main/LTX25-distilled-DiT-comfy-mix4x8-17GB.safetensors",
    },
    "ltx25_model_16gb_4x8mix": {
        "folder": "diffusion_models",
        "filename": "LTX25-distilled-DiT-comfy-mix4x8-13.8GB.safetensors",
        "url": "https://huggingface.co/joeygambino/LTX-2.5-Quantized/resolve/main/LTX25-distilled-DiT-comfy-mix4x8-13.8GB.safetensors",
    },
    "ltx25_model_16gb_nvfp4": {
        "folder": "diffusion_models",
        "filename": "LTX25-distilled-DiT-comfy-nvfp4.safetensors",
        "url": "https://huggingface.co/joeygambino/LTX-2.5-Quantized/resolve/main/LTX25-distilled-DiT-comfy-nvfp4.safetensors",
    },
    "ltx25_spatial_upscaler": {
        "folder": "latent_upscale_models",
        "filename": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "url": "https://huggingface.co/UntMods/CRT_Nodes_25Gate/resolve/main/spatial_upscaler.safetensors",
    },
    "ltx25_temporal_upscaler": {
        "folder": "latent_upscale_models",
        "filename": "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
        "url": "https://huggingface.co/UntMods/CRT_Nodes_25Gate/resolve/main/temporal_upscaler.safetensors",
    },
    "ltx25_pixel_spatial_ic_lora": {
        "folder": "loras",
        "filename": "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
        "url": "https://huggingface.co/UntMods/CRT_Nodes_25Gate/resolve/main/pixel_spatial_upscaler.safetensors",
    },
    "ltx25_ic_cnet_lora": {
        "folder": "loras",
        "filename": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "url": "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    },
    "ltx25_outpaint_lora": {
        "folder": "loras",
        "filename": "ltx-2.3-22b-ic-lora-outpaint.safetensors",
        "url": "https://huggingface.co/oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint/resolve/main/ltx-2.3-22b-ic-lora-outpaint.safetensors",
    },
    "ltx25_upscale_ic_lora": {
        "folder": "loras",
        "filename": "ltx2.3_upscale_ic-lora_06250.safetensors",
        "url": "https://huggingface.co/Zlikwid/LTX_2.3_Upscale_IC_Lora/resolve/main/ltx2.3_upscale_ic-lora_06250.safetensors",
    },
    "ltx25_clip": {
        "folder": "text_encoders",
        "filename": "gemma4-12b-ltx25-comfy-w4a8.safetensors",
        "url": "https://huggingface.co/joeygambino/LTX-2.5-Quantized/resolve/main/gemma4-12b-ltx25-comfy-w4a8.safetensors",
    },
    "ltx25_audio_vae": {
        "folder": "vae",
        "filename": "ltx-2.5-audio-vae-bf16.safetensors",
        "url": "https://huggingface.co/UntMods/CRT_Nodes_25Gate/resolve/main/audio_vae.safetensors",
    },
    "ltx25_video_vae": {
        "folder": "vae",
        "filename": "ltx-2.5-video-vae-bf16.safetensors",
        "url": "https://huggingface.co/UntMods/CRT_Nodes_25Gate/resolve/main/video_vae.safetensors",
    },
    "ltx25_duration_head": {
        "folder": "model_patches",
        "filename": "ltx-2.5-duration-head-bf16.safetensors",
        "url": "https://huggingface.co/UntMods/CRT_Nodes_25Gate/resolve/main/duration_head.safetensors",
    },
    # Pixal3D
    "pixal3d_model_bf16": {
        "folder": "diffusion_models",
        "filename": "pixal3d_bf16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/diffusion_models/pixal3d_bf16.safetensors",
    },
    "pixal3d_model_int8": {
        "folder": "diffusion_models",
        "filename": "pixal3d_int8_convrot.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/diffusion_models/pixal3d_int8_convrot.safetensors",
    },
    "pixal3d_clip_vision": {
        "folder": "clip_vision",
        "filename": "dino_v3_L_naf_fp32.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/clip_vision/dino_v3_L_naf_fp32.safetensors",
    },
    "pixal3d_shape_vae": {
        "folder": "vae",
        "filename": "trellis_2_shape_vae_bf16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/vae/trellis_2_shape_vae_bf16.safetensors",
    },
    "pixal3d_texture_vae": {
        "folder": "vae",
        "filename": "trellis_2_texture_vae_bf16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/vae/trellis_2_texture_vae_bf16.safetensors",
    },
    # MoGe-2 (depth/FOV for Pixal3D and SAM3DBody pipelines)
    "moge_model": {
        "folder": "geometry_estimation",
        "filename": "moge_2_vitl_normal_fp16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_2_vitl_normal_fp16.safetensors",
    },
    # Background removal (Pixal3D input prep)
    "birefnet_model": {
        "folder": "background_removal",
        "filename": "birefnet.safetensors",
        "url": "https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors",
    },
    # SAM3DBody
    "sam3dbody_model_bf16": {
        "folder": "detection",
        "filename": "sam_3d_body_dinov3_bf16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/sam-3d-body/resolve/main/detection/sam_3d_body_dinov3_bf16.safetensors",
    },
    "sam3dbody_model_int8": {
        "folder": "detection",
        "filename": "sam_3d_body_dinov3_int8_convrot.safetensors",
        "url": "https://huggingface.co/Comfy-Org/sam-3d-body/resolve/main/detection/sam_3d_body_dinov3_int8_convrot.safetensors",
    },
    "sam3_checkpoint": {
        "folder": "checkpoints",
        "filename": "sam3.1_multiplex_fp16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors",
    },
    # RT-DETR v4 person detector (loads through the diffusion-model path)
    "rtdetr_model_fp16": {
        "folder": "diffusion_models",
        "filename": "rt_detr_v4-x-hgnet_fp16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/RT-DETR/resolve/main/diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors",
    },
}


from comfy_extras.nodes_model_patch import ModelPatchLoader


def _model_dir(folder):
    return os.path.join(folder_paths.models_dir, folder)


def _target_path(spec):
    return os.path.join(_model_dir(spec["folder"]), spec["filename"])


def _download(url, target):
    return download_url_with_progress(
        url,
        target,
        label=os.path.basename(target),
        user_agent="CRT-AutoDL/1.0",
        console_prefix="CRT AutoDL",
    )


def ensure_model(key):
    spec = MODELS[key]
    target = _target_path(spec)
    # No early return when the file already exists: download_url_with_progress
    # HEAD-validates it against the remote size and re-downloads if a previous
    # attempt left a truncated/corrupt file behind.
    for alternate in spec.get("alternate_filenames", []):
        alternate_path = os.path.join(_model_dir(spec["folder"]), alternate)
        if os.path.exists(alternate_path):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.replace(alternate_path, target)
            return target
    _download(spec["url"], target)
    return target


def _dtype_map(name):
    return {
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(name)


def _get_sage_attention(mode):
    if mode == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
    elif mode == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif mode == "sageattn_qk_int8_pv_fp8_cuda":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
    elif mode == "sageattn_qk_int8_pv_fp8_cuda++":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout)
    elif "sageattn3" in mode:
        from sageattn3 import sageattn3_blackwell

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD", **kwargs):
            q, k, v = [x.transpose(1, 2) if tensor_layout == "NHD" else x for x in (q, k, v)]
            out = sageattn3_blackwell(q, k, v, is_causal=is_causal, attn_mask=attn_mask, per_block_mean=(mode == "sageattn3_per_block_mean"))
            return out.transpose(1, 2) if tensor_layout == "NHD" else out
    else:
        raise RuntimeError(f"[{TAG}] Unsupported attention method: {mode}")

    sage_func = torch.compiler.disable()(sage_func)

    @wrap_attn
    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        if kwargs.get("low_precision_attention", True) is False:
            return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        if skip_reshape:
            b, _, _, dim_head = q.shape
            tensor_layout = "HND"
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(lambda t: t.view(b, -1, heads, dim_head), (q, k, v))
            tensor_layout = "NHD"
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        seq_dim = 2 if tensor_layout == "HND" else 1
        if any((t.shape[seq_dim] - 1) * t.stride(seq_dim) >= 2**31 for t in (q, k, v)):
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        elif skip_output_reshape:
            out = out.transpose(1, 2)
        else:
            out = out.reshape(b, -1, heads * dim_head)
        return out

    return attention_sage


def _apply_attention_method(model, attention_method):
    if attention_method == "disabled":
        return
    if attention_method == "comfy kitchen attention":
        attention_function = get_attention_function("comfy_kitchen_int8", None)
        if attention_function is None:
            logging.warning("[%s] Comfy Kitchen attention unavailable; falling back to PyTorch attention.", TAG)
            attention_function = get_attention_function("pytorch")
        model.set_model_optimized_attention(attention_function)
        return
    if attention_method == "pytorch attention":
        model.set_model_optimized_attention(get_attention_function("pytorch"))
        return
    model.set_model_optimized_attention(_get_sage_attention(attention_method))


def _load_diffusion_model(path, weight_dtype="bf16", compute_dtype="bf16"):
    model_options = {}
    dtype = _dtype_map(weight_dtype)
    if dtype is not None:
        model_options["dtype"] = dtype
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
    if model is None:
        model = comfy.sd.load_diffusion_model(path, model_options=model_options)
    if model is None:
        raise RuntimeError(f"[{TAG}] Failed to load diffusion model from: {path}")
    compute = _dtype_map(compute_dtype)
    if compute is not None:
        model.set_model_compute_dtype(compute)
        model.force_cast_weights = False
    return model


class _FixedDiffusionLoader:
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_model"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "patch_cublaslinear": ("BOOLEAN", {"default": False}),
                "attention_method": (ATTENTION_METHODS, {"default": ATTENTION_METHODS[0]}),
                "enable_fp16_accumulation": ("BOOLEAN", {"default": True}),
            }
        }

    def load_model(self, patch_cublaslinear, attention_method, enable_fp16_accumulation):
        if patch_cublaslinear:
            args.fast.add("cublas_ops")
        else:
            args.fast.discard("cublas_ops")
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
            torch.backends.cuda.matmul.allow_fp16_accumulation = enable_fp16_accumulation
        model = _load_diffusion_model(ensure_model(self.MODEL_KEY))
        _apply_attention_method(model, attention_method)
        return (model,)


class _FixedVAELoader:
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load_vae"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load_vae(self):
        path = ensure_model(self.MODEL_KEY)
        sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        is_audio_vae = (
            "vocoder.conv_post.weight" in sd
            or "vocoder.vocoder.conv_post.weight" in sd
            or "vocoder.resblocks.0.convs1.0.weight" in sd
            or "vocoder.vocoder.resblocks.0.convs1.0.weight" in sd
        )
        if is_audio_vae:
            sd = comfy.utils.state_dict_prefix_replace(dict(sd), {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}, filter_keys=True)
            vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        else:
            vae = comfy.sd.VAE(sd=sd, device=model_management.get_torch_device(), dtype=torch.bfloat16, metadata=metadata)
        vae.throw_exception_if_invalid()
        return (vae,)


class _CoreVAELoader:
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load_vae"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load_vae(self):
        sd, metadata = comfy.utils.load_torch_file(ensure_model(self.MODEL_KEY), return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return (vae,)


class _FixedCLIPLoader:
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("CLIP",)
    FUNCTION = "load_clip"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load_clip(self):
        clip_path = ensure_model(self.MODEL_KEY)
        clip_type = getattr(comfy.sd.CLIPType, self.CLIP_TYPE.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options={},
        )
        return (clip,)


class _FixedDiffusionSelector(_FixedDiffusionLoader):
    """Diffusion loader with a dropdown of model variants."""

    OPTIONS: dict[str, str] = {}

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        names = list(cls.OPTIONS.keys())
        new_required = {
            "model_name": (names, {"default": names[0]}),
        }
        new_required.update(inputs["required"])
        inputs["required"] = new_required
        return inputs

    def load_model(self, model_name, patch_cublaslinear, attention_method, enable_fp16_accumulation):
        self.MODEL_KEY = self.OPTIONS[model_name]
        return super().load_model(patch_cublaslinear, attention_method, enable_fp16_accumulation)


class _FixedCLIPSelector(_FixedCLIPLoader):
    """CLIP loader with a dropdown of variants."""

    OPTIONS: dict[str, str] = {}

    @classmethod
    def INPUT_TYPES(cls):
        names = list(cls.OPTIONS.keys())
        return {
            "required": {
                "model_name": (names, {"default": names[0]}),
            }
        }

    def load_clip(self, model_name):
        self.MODEL_KEY = self.OPTIONS[model_name]
        return super().load_clip()


class _FixedCLIPVisionLoader:
    RETURN_TYPES = ("CLIP_VISION",)
    RETURN_NAMES = ("CLIP_VISION",)
    FUNCTION = "load_clip_vision"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load_clip_vision(self):
        clip_vision = comfy.clip_vision.load(ensure_model(self.MODEL_KEY))
        if clip_vision is None:
            raise RuntimeError(
                f"[{TAG}] CLIP Vision file is invalid and does not contain a valid vision model: "
                f"{MODELS[self.MODEL_KEY]['filename']}"
            )
        return (clip_vision,)


class _FixedModelPatchLoader:
    RETURN_TYPES = ("MODEL_PATCH",)
    RETURN_NAMES = ("model_patch",)
    FUNCTION = "load_model_patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load_model_patch(self):
        model_path = ensure_model(self.MODEL_KEY)
        name = os.path.basename(model_path)
        return ModelPatchLoader().load_model_patch(name)


class _FixedLatentUpscaleModelLoader:
    RETURN_TYPES = ("LATENT_UPSCALE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load_model(self):
        from comfy_extras.nodes_hunyuan import HunyuanVideo15SRModel, LatentUpsampler

        sd, metadata = comfy.utils.load_torch_file(ensure_model(self.MODEL_KEY), safe_load=True, return_metadata=True)
        if "blocks.0.block.0.conv.weight" in sd:
            config = {
                "in_channels": sd["in_conv.conv.weight"].shape[1],
                "out_channels": sd["out_conv.conv.weight"].shape[0],
                "hidden_channels": sd["in_conv.conv.weight"].shape[0],
                "num_blocks": len([k for k in sd.keys() if k.startswith("blocks.") and k.endswith(".block.0.conv.weight")]),
                "global_residual": False,
            }
            model = HunyuanVideo15SRModel("720p", config)
            model.load_sd(sd)
        elif "up.0.block.0.conv1.conv.weight" in sd:
            sd = {key.replace("nin_shortcut", "nin_shortcut.conv", 1): value for key, value in sd.items()}
            config = {
                "z_channels": sd["conv_in.conv.weight"].shape[1],
                "out_channels": sd["conv_out.conv.weight"].shape[0],
                "block_out_channels": tuple(sd[f"up.{i}.block.0.conv1.conv.weight"].shape[0] for i in range(len([k for k in sd.keys() if k.startswith("up.") and k.endswith(".block.0.conv1.conv.weight")]))),
            }
            model = HunyuanVideo15SRModel("1080p", config)
            model.load_sd(sd)
        elif "post_upsample_res_blocks.0.conv2.bias" in sd:
            config = json.loads(metadata["config"])
            model = LatentUpsampler.from_config(config, operations=comfy.ops.disable_weight_init).to(dtype=model_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float32]))
            model_management.archive_model_dtypes(model)
            model_patcher = comfy.model_patcher.CoreModelPatcher(
                model,
                load_device=model_management.get_torch_device(),
                offload_device=model_management.unet_offload_device(),
            )
            model.load_state_dict(sd, assign=model_patcher.is_dynamic())
            return (model_patcher,)
        else:
            raise RuntimeError(f"[{TAG}] Unsupported latent upscale model format: {self.MODEL_KEY}")
        return (model,)


class _FixedLoRALoader:
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_lora"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    def load_lora(self, model, strength_model):
        lora = comfy.utils.load_torch_file(ensure_model(self.MODEL_KEY), safe_load=True)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return (model_lora,)


class _FixedControlNetLoader:
    RETURN_TYPES = ("H3_FUN_CONTROL",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "load_controlnet"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load_controlnet(self):
        path = ensure_model(self.MODEL_KEY)
        return (load_h3_fun_control(path),)


class _GGUFModelLoader:
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_gguf_model"

    @classmethod
    def _get_gguf_module(cls):
        for key, mod in sys.modules.items():
            if key.endswith("ComfyUI-GGUF") or key.endswith("comfyui-gguf"):
                if hasattr(mod, "ops") and hasattr(mod, "nodes") and hasattr(mod, "loader"):
                    return mod
        gguf_path = os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0], "ComfyUI-GGUF")
        for module_name in ["ComfyUI-GGUF", "custom_nodes.ComfyUI-GGUF", "comfyui-gguf", "custom_nodes.comfyui-gguf", gguf_path, gguf_path.lower()]:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "ops") and hasattr(module, "nodes") and hasattr(module, "loader"):
                    return module
            except ImportError:
                continue
        raise ImportError(
            "ComfyUI-GGUF is required for GGUF model loading. "
            "Please install it from: https://github.com/city96/ComfyUI-GGUF"
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "patch_cublaslinear": ("BOOLEAN", {"default": False}),
                "attention_method": (ATTENTION_METHODS, {"default": ATTENTION_METHODS[0]}),
                "enable_fp16_accumulation": ("BOOLEAN", {"default": True}),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            }
        }

    def load_gguf_model(self, patch_cublaslinear, attention_method, enable_fp16_accumulation, dequant_dtype, patch_dtype, patch_on_device):
        if patch_cublaslinear:
            args.fast.add("cublas_ops")
        else:
            args.fast.discard("cublas_ops")
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
            torch.backends.cuda.matmul.allow_fp16_accumulation = enable_fp16_accumulation

        gguf_nodes = self._get_gguf_module()
        ops = gguf_nodes.ops.GGMLOps()

        def set_linear_dtype(attr, value):
            if value == "default":
                setattr(ops.Linear, attr, None)
            elif value == "target":
                setattr(ops.Linear, attr, value)
            else:
                setattr(ops.Linear, attr, getattr(torch, value))

        set_linear_dtype("dequant_dtype", dequant_dtype)
        set_linear_dtype("patch_dtype", patch_dtype)

        model_path = ensure_model(self.MODEL_KEY)
        try:
            sd, extra = gguf_nodes.loader.gguf_sd_loader(model_path)
        except TypeError:
            sd = gguf_nodes.loader.gguf_sd_loader(model_path)
            extra = {}

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}, metadata=extra.get("metadata", {})
        )
        if model is None:
            raise RuntimeError(f"[{TAG}] Failed to load GGUF model from: {model_path}")

        model = gguf_nodes.nodes.GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device

        _apply_attention_method(model, attention_method)

        return (model,)


class ZImageTurboModel(_FixedDiffusionLoader):
    CATEGORY = "CRT/AutoDL/ZIMAGETURBO"
    MODEL_KEY = "zimage_model"


class ZImageTurboVAE(_CoreVAELoader):
    CATEGORY = "CRT/AutoDL/ZIMAGETURBO"
    MODEL_KEY = "zimage_vae"


class ZImageTurboCLIP(_FixedCLIPLoader):
    CATEGORY = "CRT/AutoDL/ZIMAGETURBO"
    MODEL_KEY = "zimage_clip"
    CLIP_TYPE = "lumina2"


class Krea2TurboModel(_FixedDiffusionLoader):
    CATEGORY = "CRT/AutoDL/KREA2"
    MODEL_KEY = "krea2_turbo_model"


class Krea2RawModel(_FixedDiffusionLoader):
    CATEGORY = "CRT/AutoDL/KREA2"
    MODEL_KEY = "krea2_raw_model"


class Krea2VAE(_CoreVAELoader):
    CATEGORY = "CRT/AutoDL/KREA2"
    MODEL_KEY = "krea2_vae"


class Krea2CLIP(_FixedCLIPLoader):
    CATEGORY = "CRT/AutoDL/KREA2"
    MODEL_KEY = "krea2_clip"
    CLIP_TYPE = "krea2"


class Flux2KleinModel(_FixedDiffusionLoader):
    CATEGORY = "CRT/AutoDL/FLUXKLEIN"
    MODEL_KEY = "fluxklein_model"


class Flux2KleinVAE(_CoreVAELoader):
    CATEGORY = "CRT/AutoDL/FLUXKLEIN"
    MODEL_KEY = "fluxklein_vae"


class Flux2KleinCLIP(_FixedCLIPLoader):
    CATEGORY = "CRT/AutoDL/FLUXKLEIN"
    MODEL_KEY = "fluxklein_clip"
    CLIP_TYPE = "flux2"


class Flux2KleinHDRILoRA(_FixedLoRALoader):
    CATEGORY = "CRT/AutoDL/FLUXKLEIN"
    MODEL_KEY = "fluxklein_hdri_lora"


class ErnieTurboModelSelector(_FixedDiffusionSelector):
    CATEGORY = "CRT/AutoDL/ERNIE"
    OPTIONS = {
        "Turbo": "ernie_turbo_model",
        "Turbo NVFP4": "ernie_turbo_nvfp4_model",
    }


class ErnieModel(_FixedDiffusionLoader):
    CATEGORY = "CRT/AutoDL/ERNIE"
    MODEL_KEY = "ernie_model"


class ErnieVAE(_CoreVAELoader):
    CATEGORY = "CRT/AutoDL/ERNIE"
    MODEL_KEY = "ernie_turbo_vae"


class ErnieCLIP(_FixedCLIPLoader):
    CATEGORY = "CRT/AutoDL/ERNIE"
    MODEL_KEY = "ernie_turbo_clip"
    CLIP_TYPE = "flux2"


class ChronoEditModel(_FixedDiffusionLoader):
    CATEGORY = "CRT/AutoDL/ChronoEdit"
    MODEL_KEY = "chronoedit_model"


class ChronoEditDistillLoRA(_FixedLoRALoader):
    CATEGORY = "CRT/AutoDL/ChronoEdit"
    MODEL_KEY = "chronoedit_distill_lora"


class ChronoEditUpscalerLoRA(_FixedLoRALoader):
    CATEGORY = "CRT/AutoDL/ChronoEdit"
    MODEL_KEY = "chronoedit_upscaler_lora"


class ChronoEditVAE(_CoreVAELoader):
    CATEGORY = "CRT/AutoDL/ChronoEdit"
    MODEL_KEY = "chronoedit_vae"


class ChronoEditCLIP(_FixedCLIPLoader):
    CATEGORY = "CRT/AutoDL/ChronoEdit"
    MODEL_KEY = "chronoedit_clip"
    CLIP_TYPE = "wan"


class ChronoEditCLIPVision(_FixedCLIPVisionLoader):
    CATEGORY = "CRT/AutoDL/ChronoEdit"
    MODEL_KEY = "chronoedit_clip_vision"


# MiniMax H3
class MiniMaxH3ModelSelector(_FixedDiffusionSelector):
    CATEGORY = "CRT/AutoDL/MINIMAXH3"
    OPTIONS = {
        "FL2VA": "minimax_h3_model_fl2va",
        "FL2VA Light W4A8": "minimax_h3_model_fl2va_w4a8",
        "REF2VA": "minimax_h3_model_ref2va",
        "REF2VA Light W4A8": "minimax_h3_model_ref2va_w4a8",
    }


class MiniMaxH3AudioVAE(_FixedVAELoader):
    CATEGORY = "CRT/AutoDL/MINIMAXH3"
    MODEL_KEY = "minimax_h3_audio_vae"


class MiniMaxH3VideoVAE(_FixedVAELoader):
    CATEGORY = "CRT/AutoDL/MINIMAXH3"
    MODEL_KEY = "minimax_h3_video_vae"


class MiniMaxH3CLIPSelector(_FixedCLIPSelector):
    CATEGORY = "CRT/AutoDL/MINIMAXH3"
    CLIP_TYPE = "minimax"
    OPTIONS = {
        "INT8": "minimax_h3_clip_int8",
        "NVFP4": "minimax_h3_clip_nvfp4",
    }


class MiniMaxH3FL2VATurboLoRA(_FixedLoRALoader):
    CATEGORY = "CRT/AutoDL/MINIMAXH3"
    MODEL_KEY = "minimax_h3_fl2va_turbo_lora"


class MiniMaxH3REF2VATurboLoRA(_FixedLoRALoader):
    CATEGORY = "CRT/AutoDL/MINIMAXH3"
    MODEL_KEY = "minimax_h3_ref2va_turbo_lora"


class MiniMaxH3FunControlNet(_FixedControlNetLoader):
    CATEGORY = "CRT/AutoDL/MINIMAXH3"
    MODEL_KEY = "minimax_h3_fun_controlnet"


# LTX2.5
class LTX25ModelSelector(_FixedDiffusionSelector):
    CATEGORY = "CRT/AutoDL/LTX2.5"
    OPTIONS = {
        "24gb": "ltx25_model_24gb",
        "32gb": "ltx25_model_32gb",
        "16gb 4x8mix": "ltx25_model_16gb_4x8mix",
        "16gb NVFP4": "ltx25_model_16gb_nvfp4",
    }


class LTX25AudioVAE(_FixedVAELoader):
    CATEGORY = "CRT/AutoDL/LTX2.5"
    MODEL_KEY = "ltx25_audio_vae"


class LTX25VideoVAE(_FixedVAELoader):
    CATEGORY = "CRT/AutoDL/LTX2.5"
    MODEL_KEY = "ltx25_video_vae"


class LTX25CLIP(_FixedCLIPLoader):
    CATEGORY = "CRT/AutoDL/LTX2.5"
    MODEL_KEY = "ltx25_clip"
    CLIP_TYPE = "ltxv"


class LTX25SpatialUpscaler(_FixedLatentUpscaleModelLoader):
    CATEGORY = "CRT/AutoDL/LTX2.5"
    MODEL_KEY = "ltx25_spatial_upscaler"


class LTX25TemporalUpscaler(_FixedLatentUpscaleModelLoader):
    CATEGORY = "CRT/AutoDL/LTX2.5"
    MODEL_KEY = "ltx25_temporal_upscaler"


class _FixedMetadataLoRALoader(_FixedLoRALoader):
    """LoRA loader that also reports reference_downscale_factor from file metadata."""

    RETURN_TYPES = ("MODEL", "FLOAT")
    RETURN_NAMES = ("MODEL", "latent_downscale_factor")

    def load_lora(self, model, strength_model):
        lora_path = ensure_model(self.MODEL_KEY)
        lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
        try:
            latent_downscale_factor = float(metadata["reference_downscale_factor"])
        except (KeyError, ValueError, TypeError):
            latent_downscale_factor = 1.0
            logging.warning("[%s] Failed to extract reference_downscale_factor for %s", TAG, lora_path)
        if strength_model == 0:
            return (model, latent_downscale_factor)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return (model_lora, latent_downscale_factor)


class LTX25ICPixelSpatialUpscaleLoRA(_FixedMetadataLoRALoader):
    CATEGORY = "CRT/AutoDL/LTX2.5"
    MODEL_KEY = "ltx25_pixel_spatial_ic_lora"


class LTX25ICCnetLoRA(_FixedMetadataLoRALoader):
    CATEGORY = "CRT/AutoDL/LTX2.5"
    MODEL_KEY = "ltx25_ic_cnet_lora"


class LTX25OutpaintLoRA(_FixedLoRALoader):
    CATEGORY = "CRT/AutoDL/LTX2.5"
    MODEL_KEY = "ltx25_outpaint_lora"


class LTX25UpscaleICLoRA(_FixedMetadataLoRALoader):
    CATEGORY = "CRT/AutoDL/LTX2.5"
    MODEL_KEY = "ltx25_upscale_ic_lora"


class LTX25DurationHead(_FixedModelPatchLoader):
    CATEGORY = "CRT/AutoDL/LTX2.5"
    MODEL_KEY = "ltx25_duration_head"


# Pixal3D
class Pixal3DModelSelector(_FixedDiffusionSelector):
    CATEGORY = "CRT/AutoDL/Pixal3D"
    OPTIONS = {
        "BF16": "pixal3d_model_bf16",
        "INT8 ConvRot": "pixal3d_model_int8",
    }


class Pixal3DCLIPVision(_FixedCLIPVisionLoader):
    CATEGORY = "CRT/AutoDL/Pixal3D"
    MODEL_KEY = "pixal3d_clip_vision"


class Pixal3DShapeVAE(_CoreVAELoader):
    CATEGORY = "CRT/AutoDL/Pixal3D"
    MODEL_KEY = "pixal3d_shape_vae"


class Pixal3DTextureVAE(_CoreVAELoader):
    CATEGORY = "CRT/AutoDL/Pixal3D"
    MODEL_KEY = "pixal3d_texture_vae"


def _call_core_node(core_cls, **kwargs):
    """Download-free delegate: run a native V3 node's execute() and unwrap NodeOutput."""
    result = core_cls.execute(**kwargs)
    output = result.result
    return tuple(output) if isinstance(output, (list, tuple)) else (output,)


class _CoreComboDelegate:
    """Base for nodes that download a model then delegate to its native loader node."""

    FUNCTION = "load_model"
    CORE_NODE = None
    WIDGET = "model_name"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load_model(self):
        ensure_model(self.MODEL_KEY)
        return _call_core_node(self.CORE_NODE, **{self.WIDGET: MODELS[self.MODEL_KEY]["filename"]})


class Pixal3DMoGeModel(_CoreComboDelegate):
    CATEGORY = "CRT/AutoDL/Pixal3D"
    MODEL_KEY = "moge_model"
    RETURN_TYPES = ("MOGE_MODEL",)
    RETURN_NAMES = ("moge_model",)

    def load_model(self):
        from comfy_extras.nodes_moge import LoadMoGeModel

        self.CORE_NODE = LoadMoGeModel
        self.WIDGET = "model_name"
        return super().load_model()


class SAM3BodyMoGeModel(Pixal3DMoGeModel):
    CATEGORY = "CRT/AutoDL/SAM3Body"


class Pixal3DBiRefNet(_CoreComboDelegate):
    CATEGORY = "CRT/AutoDL/Pixal3D"
    MODEL_KEY = "birefnet_model"
    RETURN_TYPES = ("BACKGROUND_REMOVAL",)
    RETURN_NAMES = ("bg_model",)

    def load_model(self):
        from comfy_extras.nodes_bg_removal import LoadBackgroundRemovalModel

        self.CORE_NODE = LoadBackgroundRemovalModel
        self.WIDGET = "bg_removal_name"
        return super().load_model()


# SAM3DBody
class SAM3BodyModelSelector(_CoreComboDelegate):
    CATEGORY = "CRT/AutoDL/SAM3Body"
    RETURN_TYPES = ("SAM3D_BODY_MODEL",)
    RETURN_NAMES = ("sam3d_body_model",)
    OPTIONS = {
        "BF16": "sam3dbody_model_bf16",
        "INT8 ConvRot": "sam3dbody_model_int8",
    }

    @classmethod
    def INPUT_TYPES(cls):
        names = list(cls.OPTIONS.keys())
        return {"required": {"model_name": (names, {"default": names[0]})}}

    def load_model(self, model_name):
        from comfy_extras.nodes_sam3d_body import SAM3DBody_Loader

        key = self.OPTIONS[model_name]
        ensure_model(key)
        return _call_core_node(SAM3DBody_Loader, model_file=MODELS[key]["filename"])


class SAM3BodyCheckpoint(_CoreComboDelegate):
    CATEGORY = "CRT/AutoDL/SAM3Body"
    MODEL_KEY = "sam3_checkpoint"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")

    def load_model(self):
        from nodes import CheckpointLoaderSimple

        ensure_model(self.MODEL_KEY)
        return CheckpointLoaderSimple().load_checkpoint(MODELS[self.MODEL_KEY]["filename"])


class SAM3BodyRTDETRDetector(_FixedDiffusionLoader):
    CATEGORY = "CRT/AutoDL/SAM3Body"
    MODEL_KEY = "rtdetr_model_fp16"


NODE_CLASS_MAPPINGS = {
    "CRTAutoDLZImageTurboModel": ZImageTurboModel,
    "CRTAutoDLZImageTurboVAE": ZImageTurboVAE,
    "CRTAutoDLZImageTurboCLIP": ZImageTurboCLIP,
    "CRTAutoDLKrea2TurboModel": Krea2TurboModel,
    "CRTAutoDLKrea2RawModel": Krea2RawModel,
    "CRTAutoDLKrea2VAE": Krea2VAE,
    "CRTAutoDLKrea2CLIP": Krea2CLIP,
    "CRTAutoDLFlux2KleinModel": Flux2KleinModel,
    "CRTAutoDLFlux2KleinVAE": Flux2KleinVAE,
    "CRTAutoDLFlux2KleinCLIP": Flux2KleinCLIP,
    "CRTAutoDLFlux2KleinHDRILoRA": Flux2KleinHDRILoRA,
    "CRTAutoDLErnieTurboModelSelector": ErnieTurboModelSelector,
    "CRTAutoDLErnieModel": ErnieModel,
    "CRTAutoDLErnieVAE": ErnieVAE,
    "CRTAutoDLErnieCLIP": ErnieCLIP,
    "CRTAutoDLChronoEditModel": ChronoEditModel,
    "CRTAutoDLChronoEditDistillLoRA": ChronoEditDistillLoRA,
    "CRTAutoDLChronoEditUpscalerLoRA": ChronoEditUpscalerLoRA,
    "CRTAutoDLChronoEditVAE": ChronoEditVAE,
    "CRTAutoDLChronoEditCLIP": ChronoEditCLIP,
    "CRTAutoDLChronoEditCLIPVision": ChronoEditCLIPVision,
    "CRTAutoDLMiniMaxH3ModelSelector": MiniMaxH3ModelSelector,
    "CRTAutoDLMiniMaxH3AudioVAE": MiniMaxH3AudioVAE,
    "CRTAutoDLMiniMaxH3VideoVAE": MiniMaxH3VideoVAE,
    "CRTAutoDLMiniMaxH3CLIPSelector": MiniMaxH3CLIPSelector,
    "CRTAutoDLMiniMaxH3FL2VATurboLoRA": MiniMaxH3FL2VATurboLoRA,
    "CRTAutoDLMiniMaxH3REF2VATurboLoRA": MiniMaxH3REF2VATurboLoRA,
    "CRTAutoDLMiniMaxH3FunControlNet": MiniMaxH3FunControlNet,
    "CRTAutoDLLTX25ModelSelector": LTX25ModelSelector,
    "CRTAutoDLLTX25AudioVAE": LTX25AudioVAE,
    "CRTAutoDLLTX25VideoVAE": LTX25VideoVAE,
    "CRTAutoDLLTX25CLIP": LTX25CLIP,
    "CRTAutoDLLTX25SpatialUpscaler": LTX25SpatialUpscaler,
    "CRTAutoDLLTX25TemporalUpscaler": LTX25TemporalUpscaler,
    "CRTAutoDLLTX25ICPixelSpatialUpscaleLoRA": LTX25ICPixelSpatialUpscaleLoRA,
    "CRTAutoDLLTX25ICCnetLoRA": LTX25ICCnetLoRA,
    "CRTAutoDLLTX25OutpaintLoRA": LTX25OutpaintLoRA,
    "CRTAutoDLLTX25UpscaleICLoRA": LTX25UpscaleICLoRA,
    "CRTAutoDLLTX25DurationHead": LTX25DurationHead,
    "CRTAutoDLPixal3DModel": Pixal3DModelSelector,
    "CRTAutoDLPixal3DCLIPVision": Pixal3DCLIPVision,
    "CRTAutoDLPixal3DShapeVAE": Pixal3DShapeVAE,
    "CRTAutoDLPixal3DTextureVAE": Pixal3DTextureVAE,
    "CRTAutoDLPixal3DMoGeModel": Pixal3DMoGeModel,
    "CRTAutoDLPixal3DBiRefNet": Pixal3DBiRefNet,
    "CRTAutoDLSAM3BodyModel": SAM3BodyModelSelector,
    "CRTAutoDLSAM3BodyCheckpoint": SAM3BodyCheckpoint,
    "CRTAutoDLSAM3BodyRTDETRDetector": SAM3BodyRTDETRDetector,
    "CRTAutoDLSAM3BodyMoGeModel": SAM3BodyMoGeModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTAutoDLZImageTurboModel": "Z-Image Turbo Model (CRT AutoDL)",
    "CRTAutoDLZImageTurboVAE": "Z-Image Turbo VAE (CRT AutoDL)",
    "CRTAutoDLZImageTurboCLIP": "Z-Image Turbo CLIP (CRT AutoDL)",
    "CRTAutoDLKrea2TurboModel": "Krea 2 Turbo Model (CRT AutoDL)",
    "CRTAutoDLKrea2RawModel": "Krea 2 Raw Model (CRT AutoDL)",
    "CRTAutoDLKrea2VAE": "Krea 2 VAE (CRT AutoDL)",
    "CRTAutoDLKrea2CLIP": "Krea 2 CLIP (CRT AutoDL)",
    "CRTAutoDLFlux2KleinModel": "Flux2Klein Model (CRT AutoDL)",
    "CRTAutoDLFlux2KleinVAE": "Flux2Klein VAE (CRT AutoDL)",
    "CRTAutoDLFlux2KleinCLIP": "Flux2Klein CLIP (CRT AutoDL)",
    "CRTAutoDLFlux2KleinHDRILoRA": "Flux2Klein HDRI LoRA (CRT AutoDL)",
    "CRTAutoDLErnieTurboModelSelector": "ERNIE Turbo Model (CRT AutoDL)",
    "CRTAutoDLErnieModel": "ERNIE Model (CRT AutoDL)",
    "CRTAutoDLErnieVAE": "ERNIE VAE (CRT AutoDL)",
    "CRTAutoDLErnieCLIP": "ERNIE CLIP (CRT AutoDL)",
    "CRTAutoDLChronoEditModel": "ChronoEdit Model (CRT AutoDL)",
    "CRTAutoDLChronoEditDistillLoRA": "ChronoEdit Distill LoRA (CRT AutoDL)",
    "CRTAutoDLChronoEditUpscalerLoRA": "ChronoEdit Upscaler LoRA (CRT AutoDL)",
    "CRTAutoDLChronoEditVAE": "ChronoEdit VAE (CRT AutoDL)",
    "CRTAutoDLChronoEditCLIP": "ChronoEdit CLIP - WAN (CRT AutoDL)",
    "CRTAutoDLChronoEditCLIPVision": "ChronoEdit CLIP Vision (CRT AutoDL)",
    "CRTAutoDLMiniMaxH3ModelSelector": "MiniMax H3 Model (CRT AutoDL)",
    "CRTAutoDLMiniMaxH3AudioVAE": "MiniMax H3 AUDIO VAE (CRT AutoDL)",
    "CRTAutoDLMiniMaxH3VideoVAE": "MiniMax H3 VIDEO VAE (CRT AutoDL)",
    "CRTAutoDLMiniMaxH3CLIPSelector": "MiniMax H3 CLIP (CRT AutoDL)",
    "CRTAutoDLMiniMaxH3FL2VATurboLoRA": "MiniMax H3 FL2VA Turbo LoRA (CRT AutoDL)",
    "CRTAutoDLMiniMaxH3REF2VATurboLoRA": "MiniMax H3 REF2VA Turbo LoRA (CRT AutoDL)",
    "CRTAutoDLMiniMaxH3FunControlNet": "MiniMax H3 Fun ControlNet (CRT AutoDL)",
    "CRTAutoDLLTX25ModelSelector": "LTX2.5 Model (CRT AutoDL)",
    "CRTAutoDLLTX25AudioVAE": "LTX2.5 AUDIO VAE (CRT AutoDL)",
    "CRTAutoDLLTX25VideoVAE": "LTX2.5 VIDEO VAE (CRT AutoDL)",
    "CRTAutoDLLTX25CLIP": "LTX2.5 CLIP w4a8 Light (CRT AutoDL)",
    "CRTAutoDLLTX25SpatialUpscaler": "LTX2.5 Spatial Upscaler (CRT AutoDL)",
    "CRTAutoDLLTX25TemporalUpscaler": "LTX2.5 Temporal Upscaler (CRT AutoDL)",
    "CRTAutoDLLTX25ICPixelSpatialUpscaleLoRA": "LTX2.5 IC Pixel Spatial Upscale LoRA (CRT AutoDL)",
    "CRTAutoDLLTX25ICCnetLoRA": "LTX2.5 IC Cnet LoRA (CRT AutoDL)",
    "CRTAutoDLLTX25OutpaintLoRA": "LTX2.5 IC Outpaint LoRA (CRT AutoDL)",
    "CRTAutoDLLTX25UpscaleICLoRA": "LTX2.5 IC Upscale LoRA (CRT AutoDL)",
    "CRTAutoDLLTX25DurationHead": "LTX2.5 Duration Head (CRT AutoDL)",
    "CRTAutoDLPixal3DModel": "Pixal3D Model (CRT AutoDL)",
    "CRTAutoDLPixal3DCLIPVision": "Pixal3D CLIP Vision DINOv3 (CRT AutoDL)",
    "CRTAutoDLPixal3DShapeVAE": "Pixal3D Shape VAE (CRT AutoDL)",
    "CRTAutoDLPixal3DTextureVAE": "Pixal3D Texture VAE (CRT AutoDL)",
    "CRTAutoDLPixal3DMoGeModel": "MoGe-2 Model (CRT AutoDL)",
    "CRTAutoDLPixal3DBiRefNet": "BiRefNet Background Removal (CRT AutoDL)",
    "CRTAutoDLSAM3BodyModel": "SAM 3D Body Model (CRT AutoDL)",
    "CRTAutoDLSAM3BodyCheckpoint": "SAM 3.1 Checkpoint (CRT AutoDL)",
    "CRTAutoDLSAM3BodyRTDETRDetector": "SAM3DBody RT-DETR Detector (CRT AutoDL)",
    "CRTAutoDLSAM3BodyMoGeModel": "MoGe-2 Model (CRT AutoDL)",
}
