"""MiniMax-H3 Fun ControlNet integration for the CRT pipeline.

Thin adapter over ComfyUI's native implementation:

  * ``comfy.ldm.minimax.controlnet.MiniMaxH3FunControl`` -- the 5-block tower.
  * ``comfy_extras.nodes_minimax_h3.MiniMaxH3FunControlPatch`` -- stream init,
    per-layer skip injection, sigma window, VAE encode and inpainting.

The native patch is a real model patch (registered through the patcher), so the
tower is moved, offloaded and cleaned up by ComfyUI's model management, and the
comfy compiler is handled (``pause_malloc_graph``). The CRT nodes only add the
AutoDL loader and the Unified Sampler's CNET path on top of it.
"""

import json
import logging
import os

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
from comfy.ldm.minimax.controlnet import MiniMaxH3FunControl, is_minimax_h3_fun_state_dict
from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlPatch

LOG = logging.getLogger(__name__)


def load_h3_fun_control(path):
    """Load a Fun ControlNet checkpoint into the native tower.

    Returns a ``CoreModelPatcher`` wrapping ``MiniMaxH3FunControl`` -- the same
    object the native ``ModelPatchLoader`` produces -- so the native
    ``MiniMaxH3FunControlPatch`` can consume it. The curve-form (pruned /
    ``adaln_basis``) vs full-width AdaLN is detected from the checkpoint itself,
    so a file without the metadata key still loads.
    """
    sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    if not is_minimax_h3_fun_state_dict(sd):
        raise RuntimeError(
            f"{os.path.basename(path)} is not a MiniMax H3 Fun ControlNet checkpoint.")

    load_device = comfy.model_management.get_torch_device()
    quant = comfy.utils.detect_layer_quantization(sd, "")
    if quant is not None:
        dtype = torch.bfloat16
        operations = comfy.ops.mixed_precision_ops(quant, dtype)
    else:
        dtype = comfy.model_management.unet_dtype(
            model_params=-1,
            supported_dtypes=[torch.bfloat16, torch.float32],
            weight_dtype=comfy.utils.weight_dtype(sd),
        )
        manual_cast_dtype = comfy.model_management.unet_manual_cast(
            dtype, load_device, supported_dtypes=[torch.bfloat16, torch.float32])
        operations = comfy.ops.pick_operations(dtype, manual_cast_dtype, load_device=load_device)

    num_blocks = 0
    while "control_blocks.{}.after_proj.weight".format(num_blocks) in sd:
        num_blocks += 1
    injection_layers = tuple(range(0, num_blocks * 10, 10))
    if metadata is not None and "control_blocks_places" in metadata:
        injection_layers = tuple(json.loads(metadata["control_blocks_places"]))
        if len(injection_layers) != num_blocks:
            raise ValueError("MiniMax H3 Fun control_blocks_places metadata does not match the checkpoint")

    qkv = sd["control_blocks.0.attn.qkv_proj.weight"]
    head_dim = sd["control_blocks.0.attn.q_norm.weight"].shape[0]
    # Curve-form (pruned / adaln_basis) vs full-width AdaLN. Prefer the
    # checkpoint metadata (as the native loader does); fall back to the adaln
    # basis width for files that carry no key, where a 4-bit quantized weight's
    # input dim is still the true basis width.
    declared_form = None if metadata is None else metadata.get("minimax_h3_fun_controlnet")
    if declared_form is not None:
        use_adaln_curves = declared_form == "adaln_basis"
    else:
        use_adaln_curves = sd["control_blocks.0.adaln_proj.linear.weight"].shape[-1] == 8
    time_embed_dim = 8 if use_adaln_curves else 2688

    model = MiniMaxH3FunControl(
        control_in_dim=49,
        injection_layers=injection_layers,
        hidden_size=sd["control_proj_in.weight"].shape[0],
        num_attention_heads=qkv.shape[0] // (3 * head_dim),
        attention_head_dim=head_dim,
        ffn_hidden_size=sd["control_blocks.0.mlp.fc1.weight"].shape[0] // 2,
        time_embed_dim=time_embed_dim,
        use_adaln_curves=use_adaln_curves,
        operations=operations,
        device=comfy.model_management.unet_offload_device(),
        dtype=dtype,
    )
    model.requires_grad_(False)
    patcher = comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=load_device,
        offload_device=comfy.model_management.unet_offload_device(),
    )
    model.load_state_dict(sd, assign=patcher.is_dynamic())
    return patcher


def _prune_dead_loaded_models():
    """Drop ``LoadedModel`` entries whose ``ModelPatcher`` was garbage-collected.

    ComfyUI logs these as memory leaks but leaves them in
    ``current_loaded_models``, so ``loaded_models(only_currently_used=True)``
    yields ``None`` and the native control VAE encode crashes inside
    ``load_models_gpu``. A dead entry's patcher is already gone, so there is
    nothing left to unload; removing it is safe.
    """
    loaded = getattr(comfy.model_management, "current_loaded_models", None)
    if not loaded:
        return
    for i in range(len(loaded) - 1, -1, -1):
        if loaded[i].model is None:
            loaded.pop(i)


def _install_dead_model_cleanup():
    """Prune stale ``LoadedModel`` entries after ComfyUI's leak check.

    ``cleanup_models_gc`` runs on every ``load_models_gpu``, logs the first dead
    entry and then runs a full ``gc.collect()``; because it never removes the
    entry, the same warning and GC repeat on every load. Pruning after its pass
    keeps ``current_loaded_models`` clean without changing what it reports.
    Additive and idempotent: the original still logs and collects first.
    """
    mgmt = comfy.model_management
    original = getattr(mgmt, "cleanup_models_gc", None)
    if original is None or getattr(original, "_crt_prunes_dead", False):
        return

    def cleanup_models_gc():
        original()
        _prune_dead_loaded_models()

    cleanup_models_gc._crt_prunes_dead = True
    mgmt.cleanup_models_gc = cleanup_models_gc


_install_dead_model_cleanup()


def _guard_control_latent(patch):
    """Prune stale loaded-model entries right before the native control encode.

    ``MiniMaxH3FunControlPatch.prepare_control_latent`` snapshots the currently
    loaded models and reloads them after the VAE encode; a dead entry in that
    snapshot makes ``load_models_gpu`` fail. Wrap the instance method so every
    encode (and the de-rope pass-2 re-encode) starts from a clean list.
    """
    original = patch.prepare_control_latent

    def prepare_control_latent(target_shape):
        _prune_dead_loaded_models()
        return original(target_shape)

    patch.prepare_control_latent = prepare_control_latent


def apply_fun_control(model, control_net, vae, control_video, strength,
                      start_percent, end_percent, mask=None, source_video=None,
                      log_fn=None):
    """Patch ``model`` with the native MiniMax H3 Fun ControlNet patch.

    ``control_net`` is the ``CoreModelPatcher`` produced by
    :func:`load_h3_fun_control` (or the native ``ModelPatchLoader``). The native
    patch fits the control video to the generation canvas and the ``17n+5``
    frame grid and encodes it, so callers pass the control video raw.

    ``mask`` (1 = regenerate) plus an optional ``source_video`` selects the
    inpainting path; ``control_video`` may be omitted when only inpainting.
    """
    if strength == 0 or (control_video is None and mask is None):
        return model

    if end_percent < start_percent:
        LOG.warning(
            "H3 Fun ControlNet: end_percent (%.3f) < start_percent (%.3f); the control "
            "window is empty and the branch stays off.", end_percent, start_percent)

    # Carry already-applied Fun ControlNet patches across the clone (clone()
    # does not copy custom attributes) so the de-rope second pass can re-time
    # every chained tower.
    registered = list(getattr(model, "_crt_h3_fun_control_patches", None) or ())
    patched = model.clone()
    model_sampling = model.get_model_object("model_sampling")
    patch = MiniMaxH3FunControlPatch(
        control_net,
        vae,
        control_video[..., :3].movedim(-1, 1) if control_video is not None else None,
        mask,
        source_video[..., :3].movedim(-1, 1) if mask is not None and source_video is not None else None,
        strength,
        float(model_sampling.percent_to_sigma(start_percent)),
        float(model_sampling.percent_to_sigma(end_percent)),
    )
    _guard_control_latent(patch)
    patch.register(patched)
    # Expose the patch so the Unified Sampler can re-time its inputs for the
    # de-rope second pass (see _derope_smear_control).
    registered.append(patch)
    patched._crt_h3_fun_control_patches = registered
    if log_fn is not None:
        mode = "inpaint" if mask is not None else "control"
        log_fn(f"H3 Fun ControlNet ({mode}) applied "
               f"(strength {strength}, schedule {start_percent:.0%}-{end_percent:.0%})",
               level="ok")
    return patched


class CRT_MiniMaxH3FunControlApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "control_net": ("H3_FUN_CONTROL",),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "strength": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0,
                                       "step": 0.01,
                                       "tooltip": "Scales every control skip. 0 is a true bypass."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,
                                            "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                          "step": 0.001}),
            },
            "optional": {
                "control_video": ("IMAGE", {"tooltip": "Control video (depth/canny/pose/HED/MLSD). Fitted to the generation canvas and the 17n+5 frame grid."}),
                "mask": ("MASK", {"tooltip": "1 marks the regions to regenerate."}),
                "source_video": ("IMAGE", {"tooltip": "Video behind the mask; only read when a mask is given."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "CRT/MiniMaxH3"
    DESCRIPTION = ("Condition MiniMax-H3 on a control video and/or an inpaint mask using "
                   "ComfyUI's native Fun ControlNet model patch. Wire between an AutoDL H3 "
                   "model loader and the MiniMax H3 US Models Pipe (CRT).")

    def apply(self, model, control_net, vae, strength, start_percent, end_percent,
              control_video=None, mask=None, source_video=None):
        return (apply_fun_control(model, control_net, vae, control_video, strength,
                                  start_percent, end_percent, mask, source_video),)


NODE_CLASS_MAPPINGS = {
    "CRT_MiniMaxH3FunControlApply": CRT_MiniMaxH3FunControlApply,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_MiniMaxH3FunControlApply": "MiniMax H3 Fun Control Apply (CRT)",
}
