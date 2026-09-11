import folder_paths
import json
import os
import math
import re

from typing import Union

from nodes import LoraLoader


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


class FlexibleOptionalInputType(dict):
    def __init__(self, type, data: Union[dict, None] = None):
        self.type = type
        self.data = data
        if self.data is not None:
            for k, v in self.data.items():
                self[k] = v

    def __getitem__(self, key):
        if self.data is not None and key in self.data:
            return self.data[key]
        return (self.type,)

    def __contains__(self, key):
        return True


any_type = AnyType("*")


def log_node_warn(node_name, message):
    print(f"[magic-lora-loader][{node_name}] {message}")


def _file_exists(path):
    return path is not None and os.path.isfile(path)


def _path_exists(path):
    return path is not None and os.path.exists(path)


def _abspath(file_path: str):
    if file_path and not _path_exists(file_path):
        maybe_path = os.path.abspath(os.path.realpath(os.path.expanduser(file_path)))
        file_path = maybe_path if _path_exists(maybe_path) else file_path
    return file_path


def _load_json_file(file: str, default=None):
    if _path_exists(file):
        with open(file, "r", encoding="UTF-8") as f:
            config = f.read()
            try:
                return json.loads(config)
            except json.decoder.JSONDecodeError:
                try:
                    config = re.sub(r"^\s*//\s.*", "", config, flags=re.MULTILINE)
                    return json.loads(config)
                except json.decoder.JSONDecodeError:
                    try:
                        config = re.sub(r"(?:^|\s)//.*", "", config, flags=re.MULTILINE)
                        return json.loads(config)
                    except json.decoder.JSONDecodeError:
                        pass
    return default


def get_lora_by_filename(file_path, lora_paths=None, log_node=None):
    lora_paths = (
        lora_paths if lora_paths is not None else folder_paths.get_filename_list("loras")
    )

    if file_path in lora_paths:
        return file_path

    no_ext = [os.path.splitext(x)[0] for x in lora_paths]
    if file_path in no_ext:
        return lora_paths[no_ext.index(file_path)]

    file_path_no_ext = os.path.splitext(file_path)[0]
    if file_path_no_ext in no_ext:
        return lora_paths[no_ext.index(file_path_no_ext)]

    names = [os.path.basename(x) for x in lora_paths]
    if file_path in names:
        return lora_paths[names.index(file_path)]

    file_name = os.path.basename(file_path)
    if file_name in names:
        return lora_paths[names.index(file_name)]

    names_no_ext = [os.path.splitext(os.path.basename(x))[0] for x in lora_paths]
    if file_path in names_no_ext:
        return lora_paths[names_no_ext.index(file_path)]

    file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    if file_name_no_ext in names_no_ext:
        return lora_paths[names_no_ext.index(file_name_no_ext)]

    for i, lora_path in enumerate(lora_paths):
        if file_path in lora_path:
            return lora_paths[i]

    if log_node is not None:
        log_node_warn(log_node, f'LoRA "{file_path}" not found, skipping.')
    return None


def _get_info_file(file_path: str, force=False):
    info_path = f"{file_path}.rgthree-info.json"
    return info_path if _file_exists(info_path) or force else None


def _get_folder_path(file: str, model_type) -> str | None:
    file_path = folder_paths.get_full_path(model_type, file)
    if not _file_exists(file_path):
        file_path = _abspath(file_path)
    if not _file_exists(file_path):
        file_path = None
    return file_path


def get_model_info_file_data(file: str, model_type, default=None):
    file_path = _get_folder_path(file, model_type)
    if file_path is None:
        return default
    return _load_json_file(_get_info_file(file_path), default=default)

NODE_NAME = "Magic LoRA Loader"
_PRESETS_FILE = None
_MODEL_TYPES = {
    "Flux2Klein",
    "LTX2.3",
    "ZImageTurbo",
    "WAN2.2",
    "ERNIEImage",
    "Ideogram4",
    "Krea2Turbo",
    "MiniMaxH3",
}
_DEFAULT_MT = "Flux2Klein"
_ROUTES_REGISTERED = globals().get("_ROUTES_REGISTERED", False)
_BLOCK_KEYS = ("double", "single", "transformer", "layers", "blocks")


def _is_all_ones(block_weights: dict) -> bool:
    if not block_weights:
        return True
    for key, vals in block_weights.items():
        if key == "non_block":
            if float(vals) != 1.0:
                return False
            continue
        for w in vals:
            if float(w) != 1.0:
                return False
    return True


def _copy_block_weights(raw) -> dict:
    """Return the supported block arrays as plain JSON-safe floats."""
    if not isinstance(raw, dict):
        return {}
    result = {}
    for block_key in _BLOCK_KEYS:
        values = raw.get(block_key)
        if values:
            result[block_key] = [float(value) for value in values]
    return result


def _json_number(value) -> float:
    """Keep report numbers readable without changing values used by the loader."""
    rounded = round(float(value), 10)
    return 0.0 if rounded == 0 else rounded


def _report_block_weights(raw) -> dict:
    return {
        block_key: [_json_number(value) for value in values]
        for block_key, values in _copy_block_weights(raw).items()
    }


def _combine_block_weights(per_lora: dict, global_blocks: dict) -> dict:
    """Combine per-LoRA weights with active global multipliers."""
    merged = {}
    for block_key in set(per_lora) | set(global_blocks):
        per_values = per_lora.get(block_key, [])
        global_values = global_blocks.get(block_key, [])
        if per_values and global_values:
            merged[block_key] = [
                value * (global_values[index] if index < len(global_values) else 1.0)
                for index, value in enumerate(per_values)
            ]
        elif per_values:
            merged[block_key] = list(per_values)
        elif global_values:
            merged[block_key] = list(global_values)
    return merged


def _key_block_weight(key: str, block_weights: dict) -> float:
    m = re.search(r"double_blocks\.(\d+)\.", key)
    if m:
        idx = int(m.group(1))
        weights = block_weights.get("double", [])
        return float(weights[idx]) if idx < len(weights) else 1.0

    m = re.search(r"single_blocks\.(\d+)\.", key)
    if m:
        idx = int(m.group(1))
        weights = block_weights.get("single", [])
        return float(weights[idx]) if idx < len(weights) else 1.0

    m = re.search(r"(?:diffusion_model\.)?transformer_blocks\.(\d+)\.", key)
    if m:
        idx = int(m.group(1))
        weights = block_weights.get("transformer", [])
        return float(weights[idx]) if idx < len(weights) else 1.0

    m = re.search(r"(?:^|\.|\b)layers[._](\d+)[._]", key)
    if m:
        idx = int(m.group(1))
        weights = block_weights.get("layers", [])
        return float(weights[idx]) if idx < len(weights) else 1.0

    m = re.search(r"(?:^|[._])(?<!double_)(?<!single_)(?<!refiner[._])blocks[._](\d+)[._]", key)
    if m:
        idx = int(m.group(1))
        weights = block_weights.get("blocks", [])
        return float(weights[idx]) if idx < len(weights) else 1.0

    # keys not covered by any block list (MiniMax token_refiner, embeddings,
    # final layers, text encoders, ...) get the global non-block weight
    return float(block_weights.get("non_block", 1.0))


def _compute_merged_lora(model, pre_patch_counts: dict) -> dict:
    """Collect the LoRA patches added since the pre_patch_counts snapshot.
    Returns {model_state_dict_key: [(strength, patch), ...]} validated against the
    stored weight shapes. The dense deltas are NOT materialized here: on large
    models they can total tens of GB of RAM, so the consumer builds them one key
    at a time via _materialize_delta."""
    # Collect logical weight shapes for validation (cheap - just metadata, no tensor copies).
    # NOTE: model_state_dict() reports packed storage sizes for quantized layers
    # (e.g. 4-bit w4a8 stores [N, K/2], half the logical numel), so override with
    # each module's _orig_shape where available.
    weight_numel = {}
    try:
        sd = model.model_state_dict()
        for k, v in sd.items():
            try:
                weight_numel[k] = v.numel()
            except Exception:
                continue
    except Exception:
        pass
    try:
        base = getattr(model, "model", None)
        mods = base.named_modules() if base is not None and hasattr(base, "named_modules") else []
        for name, mod in mods:
            orig = getattr(mod, "_orig_shape", None)
            if not orig:
                continue
            n = 1
            for d in orig:
                n *= int(d)
            for cand in (name + ".weight", "diffusion_model." + name + ".weight"):
                weight_numel[cand] = n
    except Exception:
        pass

    merged = {}
    all_patches = getattr(model, "patches", None) or {}
    for key, patch_list in all_patches.items():
        pre = pre_patch_counts.get(key, 0)
        new_patches = patch_list[pre:]
        if not new_patches:
            continue
        entries = []
        for p in new_patches:
            strength = float(p[0])
            v = p[1]
            try:
                if hasattr(v, "name") and hasattr(v, "weights"):
                    w = v.weights
                    if v.name != "lora":
                        log_node_warn(NODE_NAME, f"Unsupported adapter '{v.name}' on {key} - skipping (only standard LoRA supported).")
                        continue
                    if w[3] is not None:
                        log_node_warn(NODE_NAME, f"Tucker/LoCon mid weight on {key} - skipping (not supported in merge).")
                        continue
                    if w[4] is not None:
                        log_node_warn(NODE_NAME, f"DoRA scale on {key} - skipping (DoRA merge requires base weights, not supported).")
                        continue
                    # delta = up.flatten(1) @ down.flatten(1); get its size without building it
                    delta_numel = w[0].shape[0] * (w[1].numel() // w[1].shape[0])
                elif isinstance(v, tuple) and len(v) >= 2 and v[0] == "diff":
                    delta_numel = v[1][0].numel()
                else:
                    log_node_warn(NODE_NAME, f"Unknown patch format on {key} - skipping.")
                    continue
                # Skip patches whose delta is incompatible with the stored weight shape.
                # This occurs with FP8 quantized models that pack weights into a different layout.
                if key in weight_numel and delta_numel != weight_numel[key]:
                    log_node_warn(
                        NODE_NAME,
                        f"Skipping {key}: LoRA delta ({delta_numel} el) "
                        f"does not match weight ({weight_numel[key]} el) - "
                        f"LoRA was trained for a different shape and cannot be applied to this layer.",
                    )
                    continue
                entries.append((strength, v))
            except Exception as e:
                log_node_warn(NODE_NAME, f"merge skip {key}: {e}")
        if entries:
            merged[key] = entries
    return merged


def _materialize_delta(entries, device):
    """Sum one key's collected patches into a dense fp32 delta on device.
    Called per key so only a single dense delta exists at a time."""
    import torch

    delta = None
    for strength, v in entries:
        if hasattr(v, "name") and hasattr(v, "weights"):
            w = v.weights
            mat1 = w[0].to(device=device, dtype=torch.float32)
            mat2 = w[1].to(device=device, dtype=torch.float32)
            alpha_val = w[2]
            scale = float(alpha_val) / mat2.shape[0] if alpha_val is not None else 1.0
            d = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1))
            d *= strength * scale
            del mat1, mat2
        else:  # "diff" patch (validated at collect time)
            raw = v[1][0].to(device=device, dtype=torch.float32)
            d = strength * raw.reshape(raw.shape[0], -1)
        if delta is None:
            delta = d
        else:
            delta += d
            del d
    return delta


def _scale_new_patches(patcher, block_weights: dict, pre_counts: dict):
    if patcher is None:
        return
    patches = getattr(patcher, "patches", None)
    if not patches:
        return
    for key, entries in patches.items():
        weight = _key_block_weight(key, block_weights)
        if weight == 1.0:
            continue
        pre = pre_counts.get(key, 0)
        new_entries = entries[pre:]
        if not new_entries:
            continue
        patches[key] = entries[:pre] + [(e[0] * weight,) + e[1:] for e in new_entries]


def _extract_block_index(key: str) -> int | None:
    for pattern in (
        r"double_blocks\.(\d+)\.",
        r"single_blocks\.(\d+)\.",
        r"(?:diffusion_model\.)?transformer_blocks\.(\d+)\.",
        r"(?:^|\.|\b)layers[._](\d+)[._]",
        r"(?:^|[._])(?<!double_)(?<!single_)(?<!refiner[._])blocks[._](\d+)[._]",
    ):
        m = re.search(pattern, key)
        if m:
            return int(m.group(1))
    return None


def _parse_keep_blocks(value: str) -> set[int] | None:
    value = (value or "").strip()
    if not value:
        return None

    blocks = set()
    for token in re.split(r"[,\s]+", value):
        if not token:
            continue
        if re.fullmatch(r"\d+", token):
            blocks.add(int(token))
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            if start > end:
                start, end = end, start
            blocks.update(range(start, end + 1))
            continue
        raise ValueError(
            f"Invalid keep_blocks token '{token}'. Use comma-separated block numbers, e.g. 0,1,2,16-27."
        )
    return blocks


def _dynamic_rank(S, total_sq: float, max_r: int, method: str, param: float) -> int:
    """Pick the smallest rank meeting the retention target from the top-q
    singular values. total_sq is the exact squared Frobenius norm of the
    matrix, so sv_fro retention is exact even without the tail values."""
    import torch

    if S.numel() == 0 or float(S[0]) <= 1e-6:
        return 1
    if method == "sv_fro":
        cum = torch.cumsum(S.pow(2), dim=0) / total_sq
        idx = int(torch.searchsorted(cum, param**2)) + 1
    elif method == "sv_ratio":
        idx = int(torch.sum(S > S[0] / param).item())
    else:
        idx = max_r
    return max(1, min(idx, max_r))


def load_lora_with_blocks(
    model,
    clip,
    lora_name: str,
    strength_model: float,
    strength_clip: float,
    block_weights: dict | None,
):
    if block_weights is None or _is_all_ones(block_weights):
        return LoraLoader().load_lora(model, clip, lora_name, strength_model, strength_clip)

    model_pre = {k: len(v) for k, v in (getattr(model, "patches", None) or {}).items()}
    clip_pre = {k: len(v) for k, v in (getattr(clip, "patches", None) or {}).items()}

    new_model, new_clip = LoraLoader().load_lora(
        model, clip, lora_name, strength_model, strength_clip
    )

    _scale_new_patches(new_model, block_weights, model_pre)
    _scale_new_patches(new_clip, block_weights, clip_pre)

    return new_model, new_clip


class MagicLoraLoader:
    """Magic LoRA Loader with optional wet/cap/block controls."""

    NAME = NODE_NAME
    CATEGORY = "CRT/LoRA"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(
                type=any_type,
                data={
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                },
            ),
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "MERGED_LORA", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "merged_lora", "lora_stack_info")
    FUNCTION = "load_loras"

    @classmethod
    def IS_CHANGED(cls, unique_id=None, prompt=None, **kwargs):
        # When the merged_lora output is connected, return NaN so ComfyUI
        # never serves a stale cached {} from a previous disconnected run.
        if cls._merged_lora_output_is_connected(unique_id, prompt):
            return float("nan")
        return 0

    @staticmethod
    def _merged_lora_output_is_connected(unique_id, prompt) -> bool:
        """Return True if output slot 2 (merged_lora) is wired to at least one node."""
        try:
            uid = str(unique_id)
            for node_data in prompt.values():
                for val in node_data.get("inputs", {}).values():
                    if isinstance(val, list) and len(val) == 2 and str(val[0]) == uid and val[1] == 2:
                        return True
        except Exception:
            pass
        return False

    def load_loras(self, model=None, clip=None, unique_id=None, prompt=None, **kwargs):
        pre_patch_counts = {k: len(v) for k, v in (getattr(model, "patches", None) or {}).items()} if model is not None else {}
        model_connected = model is not None
        clip_connected = clip is not None
        wet = 1.0
        wet_raw = kwargs.pop("wet", None)
        if isinstance(wet_raw, dict) and wet_raw.get("__pgc_wet"):
            wet = max(0.0, min(2.0, float(wet_raw.get("value", 1.0))))
        elif isinstance(wet_raw, (int, float)):
            wet = max(0.0, min(2.0, float(wet_raw)))

        cap = 0.0
        cap_raw = kwargs.pop("pgc_cap", None)
        if isinstance(cap_raw, dict) and cap_raw.get("__pgc_cap") is True:
            cap = max(0.0, float(cap_raw.get("value", 0.0)))
        elif isinstance(cap_raw, (int, float)):
            cap = max(0.0, float(cap_raw))

        non_block = 1.0
        nb_raw = kwargs.pop("pgc_non_block", None)
        if isinstance(nb_raw, dict) and nb_raw.get("__pgc_non_block") is True:
            non_block = max(0.0, min(2.0, float(nb_raw.get("value", 1.0))))
        elif isinstance(nb_raw, (int, float)):
            non_block = max(0.0, min(2.0, float(nb_raw)))

        model_type = _DEFAULT_MT
        model_type_raw = kwargs.pop("pgc_model_type", None)
        if isinstance(model_type_raw, dict) and model_type_raw.get("__pgc_model_type") is True:
            candidate_model_type = model_type_raw.get("value")
            if candidate_model_type in _MODEL_TYPES:
                model_type = candidate_model_type
        elif model_type_raw in _MODEL_TYPES:
            model_type = model_type_raw

        global_blocks = None
        gb_raw = kwargs.pop("pgc_global_blocks", None)
        kwargs.pop("global_blocks", None)
        if isinstance(gb_raw, dict) and gb_raw.get("__pgc_global_blocks") is True:
            candidate = _copy_block_weights(gb_raw)
            if candidate and any(v != 1.0 for vals in candidate.values() for v in vals):
                global_blocks = candidate

        configured_global_blocks = _copy_block_weights(gb_raw)
        stack_info = {
            "schema": "crt.magic_lora_loader.stack_info.v1",
            "model_type": model_type,
            "inputs": {
                "model_connected": model_connected,
                "clip_connected": clip_connected,
            },
            "global_dry_wet": {
                "wet_multiplier": _json_number(wet),
                "wet_percent": _json_number(wet * 100.0),
            },
            "strength_cap": {
                "enabled": cap > 0.0,
                "value": _json_number(cap),
            },
            "global_block_weights": _report_block_weights(configured_global_blocks),
            "non_block_weight": _json_number(non_block),
            "used_lora_stacks": [],
            "skipped_loras": [],
        }
        configured_count = 0
        enabled_count = 0

        for input_name, value in kwargs.items():
            if not (
                input_name.upper().startswith("LORA_")
                and isinstance(value, dict)
                and "on" in value
                and "lora" in value
                and "strength" in value
            ):
                continue

            configured_count += 1
            stack_position = configured_count
            lora_name = value.get("lora")

            if not value["on"]:
                stack_info["skipped_loras"].append(
                    {
                        "stack_position": stack_position,
                        "input_name": input_name,
                        "lora": lora_name,
                        "reason": "disabled",
                    }
                )
                continue

            enabled_count += 1
            configured_model_weight = float(value["strength"])
            configured_clip_weight = value.get("strengthTwo")
            if configured_clip_weight is None:
                configured_clip_weight = configured_model_weight
            else:
                configured_clip_weight = float(configured_clip_weight)

            strength_model = configured_model_weight * wet
            strength_clip_raw = value.get("strengthTwo")

            if not clip_connected:
                if strength_clip_raw is not None and strength_clip_raw != 0:
                    log_node_warn(
                        NODE_NAME,
                        "Received clip strength even though no clip supplied!",
                    )
                strength_clip = 0
            else:
                strength_clip = (
                    float(strength_clip_raw) * wet
                    if strength_clip_raw is not None
                    else strength_model
                )

            if cap > 0.0:
                strength_model = max(-cap, min(cap, strength_model))
                if clip_connected:
                    strength_clip = max(-cap, min(cap, strength_clip))

            if strength_model == 0 and strength_clip == 0:
                stack_info["skipped_loras"].append(
                    {
                        "stack_position": stack_position,
                        "input_name": input_name,
                        "lora": lora_name,
                        "reason": "zero_effective_weight",
                    }
                )
                continue

            if not lora_name:
                stack_info["skipped_loras"].append(
                    {
                        "stack_position": stack_position,
                        "input_name": input_name,
                        "lora": lora_name,
                        "reason": "no_lora_selected",
                    }
                )
                continue

            lora = get_lora_by_filename(lora_name, log_node=self.NAME)
            if lora is None:
                stack_info["skipped_loras"].append(
                    {
                        "stack_position": stack_position,
                        "input_name": input_name,
                        "lora": lora_name,
                        "reason": "lora_not_found",
                    }
                )
                continue

            if not model_connected:
                stack_info["skipped_loras"].append(
                    {
                        "stack_position": stack_position,
                        "input_name": input_name,
                        "lora": lora_name,
                        "resolved_lora": lora,
                        "reason": "model_not_connected",
                    }
                )
                continue

            per_lora_blocks = _copy_block_weights(value.get("blocks"))
            if per_lora_blocks:
                blocks = _combine_block_weights(per_lora_blocks, global_blocks or {})
            else:
                blocks = global_blocks

            if per_lora_blocks and global_blocks:
                block_weights_mode = "per_lora_x_global"
            elif per_lora_blocks:
                block_weights_mode = "per_lora"
            elif global_blocks:
                block_weights_mode = "global"
            else:
                block_weights_mode = "default"

            if non_block != 1.0:
                blocks = {**(blocks or {}), "non_block": non_block}

            if blocks is not None:
                model, clip = load_lora_with_blocks(
                    model, clip, lora, strength_model, strength_clip, blocks
                )
            else:
                model, clip = LoraLoader().load_lora(
                    model, clip, lora, strength_model, strength_clip
                )

            effective_blocks = _report_block_weights(blocks)
            if not effective_blocks and configured_global_blocks:
                effective_blocks = _report_block_weights(configured_global_blocks)
            stack_info["used_lora_stacks"].append(
                {
                    "stack_position": stack_position,
                    "input_name": input_name,
                    "lora": lora_name,
                    "resolved_lora": lora,
                    "configured_weights": {
                        "model": _json_number(configured_model_weight),
                        "clip": _json_number(configured_clip_weight),
                    },
                    "effective_weights": {
                        "model": _json_number(strength_model),
                        "clip": _json_number(strength_clip) if clip_connected else None,
                    },
                    "automix_reduction": _json_number(value.get("reduction", 1.0)),
                    "block_weights_mode": block_weights_mode,
                    "per_lora_block_weights": _report_block_weights(per_lora_blocks) or None,
                    "effective_block_weights": effective_blocks,
                }
            )

        if model is not None and self._merged_lora_output_is_connected(unique_id, prompt):
            merged_lora = _compute_merged_lora(model, pre_patch_counts)
        else:
            merged_lora = {}
        stack_info["summary"] = {
            "configured_lora_count": configured_count,
            "enabled_lora_count": enabled_count,
            "used_lora_count": len(stack_info["used_lora_stacks"]),
            "skipped_lora_count": len(stack_info["skipped_loras"]),
        }
        lora_stack_info = json.dumps(stack_info, indent=2, ensure_ascii=False)
        return (model, clip, merged_lora, lora_stack_info)

    @classmethod
    def get_enabled_loras_from_prompt_node(
        cls, prompt_node: dict
    ) -> list[dict[str, Union[str, float]]]:
        result = []
        for name, lora in prompt_node["inputs"].items():
            if name.startswith("lora_") and lora["on"]:
                lora_file = get_lora_by_filename(lora["lora"], log_node=cls.NAME)
                if lora_file is not None:
                    lora_dict = {
                        "name": lora["lora"],
                        "strength": lora["strength"],
                        "path": folder_paths.get_full_path("loras", lora_file),
                    }
                    if "strengthTwo" in lora:
                        lora_dict["strength_clip"] = lora["strengthTwo"]
                    result.append(lora_dict)
        return result

    @classmethod
    def get_enabled_triggers_from_prompt_node(
        cls, prompt_node: dict, max_each: int = 1
    ):
        loras = [l["name"] for l in cls.get_enabled_loras_from_prompt_node(prompt_node)]
        trained_words = []
        for lora in loras:
            info = get_model_info_file_data(lora, "loras", default={})
            if not info or not info.keys():
                log_node_warn(
                    NODE_NAME,
                    f"No info found for lora {lora} when grabbing triggers. Have you generated an info file"
                    ' from the Magic LoRA Loader "Show Info" dialog?',
                )
                continue
            if "trainedWords" not in info or not info["trainedWords"]:
                log_node_warn(
                    NODE_NAME,
                    f"No trained words for lora {lora} when grabbing triggers. Have you fetched data from"
                    " civitai or manually added words?",
                )
                continue
            trained_words += [
                w
                for wi in info["trainedWords"][:max_each]
                if (wi and (w := wi["word"]))
            ]
        return trained_words


class SaveMergedLora:
    """Save merged LoRA deltas from Magic LoRA Loader as a single .safetensors file."""

    NAME = "Save Merged LoRA"
    CATEGORY = "CRT/LoRA"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merged_lora": ("MERGED_LORA",),
                "filename": ("STRING", {"default": "merged_lora"}),
                "keep_blocks": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Comma-separated block indexes to keep in the saved LoRA. "
                            "Empty = save all blocks. Supports ranges like 0-13,16,17,26,27."
                        ),
                    },
                ),
                "save_to": (["loras folder", "output folder"],),
                "rank": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 512,
                        "step": 8,
                        "tooltip": (
                            "SVD rank for compression. "
                            "0 = lossless diff format (larger file). "
                            ">0 = LoRA format with SVD at this rank. "
                            "Acts as max rank per layer when dynamic_method is enabled."
                        ),
                    },
                ),
                "dynamic_method": (
                    ["disabled", "sv_fro", "sv_ratio"],
                    {
                        "default": "disabled",
                        "tooltip": (
                            "Dynamic per-layer rank selection. "
                            "sv_fro: smallest rank keeping dynamic_param of the Frobenius norm. "
                            "sv_ratio: keep singular values above max(S)/dynamic_param."
                        ),
                    },
                ),
                "dynamic_param": (
                    "FLOAT",
                    {
                        "default": 0.99,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.001,
                        "tooltip": (
                            "Target for dynamic_method. sv_fro: kept Frobenius norm ratio "
                            "per layer, e.g. 0.99, 0.995, 0.998. sv_ratio: keep singular values above "
                            "max(S)/dynamic_param, e.g. 10-100. Ignored when "
                            "dynamic_method is disabled."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    OUTPUT_NODE = True
    FUNCTION = "save"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # This is a file-writing output node. Always re-run when queued instead
        # of letting ComfyUI reuse a cached filepath/empty result.
        return float("nan")

    def save(self, merged_lora, filename, keep_blocks, save_to, rank, dynamic_method, dynamic_param):
        import torch
        import safetensors.torch
        from comfy.utils import ProgressBar

        if not merged_lora:
            log_node_warn(NODE_NAME, "No merged LoRA data to save.")
            return ("",)

        keep = _parse_keep_blocks(keep_blocks)
        if keep is not None:
            before = len(merged_lora)
            merged_lora = {
                key: entries
                for key, entries in merged_lora.items()
                if (idx := _extract_block_index(key)) is not None and idx in keep
            }
            print(
                f"[crt-pll] Magic Save Merged LoRA: keeping blocks {sorted(keep)} "
                f"({len(merged_lora)}/{before} tensors)."
            )
            if not merged_lora:
                log_node_warn(NODE_NAME, "No merged LoRA tensors matched keep_blocks; nothing saved.")
                return ("",)

        if save_to == "loras folder":
            base_dir = folder_paths.get_folder_paths("loras")[0]
        else:
            base_dir = folder_paths.get_output_directory()

        save_path = os.path.join(base_dir, f"{filename}.safetensors")
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        import comfy.model_management as mm
        device = mm.get_torch_device()

        total = len(merged_lora)
        fmt = "diff" if rank == 0 else f"rank-{rank} LoRA"
        if rank > 0 and dynamic_method != "disabled":
            fmt = f"dynamic {dynamic_method} {dynamic_param} (max rank {rank}) LoRA"
        print(f"[crt-pll] Magic Save Merged LoRA: processing {total} keys ({fmt}) on {device}...")
        pbar = ProgressBar(total)

        # Keep GPU memory bounded. Each tensor is uploaded, decomposed, moved
        # back to CPU, then all GPU intermediates are released before the next key.
        keys = list(merged_lora.keys())
        lora_sd = {}
        skipped = 0
        rank_list = []

        if rank == 0:
            for i, model_key in enumerate(keys):
                base_key = model_key[: -len(".weight")] if model_key.endswith(".weight") else model_key
                print(f"[crt-pll]   [{i + 1}/{total}] {model_key}")
                delta = _materialize_delta(merged_lora[model_key], device)
                lora_sd[f"{base_key}.diff"] = delta.to(device="cpu", dtype=torch.float16)
                del delta
                pbar.update(1)
        else:
            print(f"[crt-pll]   Streaming SVD on {device} (CPU offload per tensor)...")
            for i, model_key in enumerate(keys):
                base_key = model_key[: -len(".weight")] if model_key.endswith(".weight") else model_key
                try:
                    d_gpu = _materialize_delta(merged_lora[model_key], device)
                    max_r = min(rank, min(d_gpu.shape))
                    # Randomized low-rank SVD: only the top-r subspace is needed, a full
                    # SVD of these large matrices is orders of magnitude slower.
                    if dynamic_method != "disabled":
                        # The exact squared Frobenius norm of the delta gives the total
                        # spectral energy, so dynamic rank selection needs no full SVD.
                        total_sq = float(d_gpu.pow(2).sum())
                        U, S, V = torch.svd_lowrank(d_gpu, q=max_r, niter=7)
                        r = _dynamic_rank(S, total_sq, max_r, dynamic_method, dynamic_param)
                    else:
                        # Slight oversampling improves accuracy of the truncated basis.
                        r = max_r
                        q = min(r + 16, min(d_gpu.shape))
                        U, S, V = torch.svd_lowrank(d_gpu, q=q, niter=7)
                    lora_sd[f"{base_key}.lora_up.weight"] = (
                        U[:, :r] * S[:r]
                    ).contiguous().to(device="cpu", dtype=torch.float16)
                    lora_sd[f"{base_key}.lora_down.weight"] = V[:, :r].t().contiguous().to(
                        device="cpu", dtype=torch.float16
                    )
                    lora_sd[f"{base_key}.alpha"] = torch.tensor(float(r))
                    rank_list.append(r)
                    print(f"[crt-pll]   [{i + 1}/{total}] {model_key} -> rank {r}")
                    del d_gpu, U, S, V
                except Exception as e:
                    log_node_warn(NODE_NAME, f"SVD failed for {model_key}: {e}")
                    skipped += 1
                finally:
                    if getattr(device, "type", str(device)) == "cuda":
                        torch.cuda.empty_cache()
                pbar.update(1)

        if not lora_sd:
            log_node_warn(NODE_NAME, "Merged LoRA is empty after processing, nothing saved.")
            return ("",)

        print(f"[crt-pll]   Writing {save_path} ...")
        safetensors.torch.save_file(lora_sd, save_path)
        keys_saved = total - skipped
        if rank_list:
            avg_rank = sum(rank_list) / len(rank_list)
            print(f"[crt-pll]   Ranks: min {min(rank_list)}, max {max(rank_list)}, avg {avg_rank:.1f}")
        print(f"[crt-pll] Done - saved {keys_saved}/{total} keys -> {save_path}")
        return (save_path,)


def _load_presets_raw():
    if _PRESETS_FILE and os.path.exists(_PRESETS_FILE):
        try:
            with open(_PRESETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_presets_raw(data: dict):
    os.makedirs(os.path.dirname(_PRESETS_FILE), exist_ok=True)
    with open(_PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _migrate_presets(data: dict) -> dict:
    if not data:
        return {mt: {} for mt in _MODEL_TYPES}
    if set(data.keys()) <= _MODEL_TYPES:
        result = {mt: {} for mt in _MODEL_TYPES}
        for mt in _MODEL_TYPES:
            result[mt] = data.get(mt, {})
        return result
    migrated = {mt: {} for mt in _MODEL_TYPES}
    migrated[_DEFAULT_MT] = data
    return migrated


def _load_presets(model_type=_DEFAULT_MT):
    data = _migrate_presets(_load_presets_raw())
    if model_type not in _MODEL_TYPES:
        model_type = _DEFAULT_MT
    return data.get(model_type, {})


def _save_presets(model_type: str, presets: dict):
    data = _migrate_presets(_load_presets_raw())
    if model_type not in _MODEL_TYPES:
        model_type = _DEFAULT_MT
    data[model_type] = presets
    _save_presets_raw(data)


_ARCH = {
    "Flux2Klein": {
        "block_types": [
            ("double", r"double_blocks[._](\d+)[._]", 8),
            ("single", r"single_blocks[._](\d+)[._]", 24),
        ],
    },
    "LTX2.3": {
        "block_types": [
            (
                "transformer",
                r"(?:diffusion_model[._])?transformer_blocks[._](\d+)[._]",
                48,
            ),
        ],
    },
    "ZImageTurbo": {
        "block_types": [
            ("layers", r"layers[._](\d+)[._]", 30),
        ],
    },
    "WAN2.2": {
        "block_types": [
            ("blocks", r"(?:^|[._])(?<!double_)(?<!single_)blocks[._](\d+)[._]", 40),
        ],
    },
    "ERNIEImage": {
        "block_types": [
            ("layers", r"layers[._](\d+)[._]", 36),
        ],
    },
    "Ideogram4": {
        "block_types": [
            ("layers", r"layers[._](\d+)[._]", 34),
        ],
    },
    "Krea2Turbo": {
        "block_types": [
            ("blocks", r"(?:^|[._])(?<!double_)(?<!single_)blocks[._](\d+)[._]", 28),
        ],
    },
    "MiniMaxH3": {
        "block_types": [
            (
                "blocks",
                r"(?:^|[._])(?<!double_)(?<!single_)(?<!refiner[._])blocks[._](\d+)[._]",
                50,
            ),
        ],
    },
}


def _flat_weights(model_type):
    cfg = _ARCH.get(model_type, _ARCH[_DEFAULT_MT])
    return {btype: [1.0] * count for btype, _, count in cfg["block_types"]}


def _compute_profile(lora_sd, model_type=_DEFAULT_MT):
    cfg = _ARCH.get(model_type, _ARCH[_DEFAULT_MT])
    accum = {btype: [0.0] * count for btype, _, count in cfg["block_types"]}
    for key, tensor in lora_sd.items():
        try:
            norm_sq = float(tensor.float().norm().item() ** 2)
        except Exception:
            continue
        for btype, pattern, count in cfg["block_types"]:
            m = re.search(pattern, key)
            if m:
                idx = int(m.group(1))
                if idx < count:
                    accum[btype][idx] += norm_sq
                break
    return {btype: [math.sqrt(v) for v in vals] for btype, vals in accum.items()}


def _normalise_profile(profile):
    all_vals = [v for vals in profile.values() for v in vals]
    peak = max(all_vals) if all_vals else 0.0
    if peak < 1e-9:
        peak = 1.0
    return {btype: [round(v / peak, 4) for v in vals] for btype, vals in profile.items()}


def _competitive_weights(profiles, model_type=_DEFAULT_MT):
    cfg = _ARCH.get(model_type, _ARCH[_DEFAULT_MT])
    block_types = [(btype, count) for btype, _, count in cfg["block_types"]]
    total_blocks = sum(count for _, count in block_types)
    n = len(profiles)
    weights = [_flat_weights(model_type) for _ in range(n)]

    frac = []
    for p in profiles:
        total = sum(v for vals in p.values() for v in vals)
        if total < 1e-9:
            uniform = 1.0 / total_blocks
            frac.append({btype: [uniform] * count for btype, count in block_types})
        else:
            frac.append({btype: [v / total for v in p[btype]] for btype, _ in block_types})

    avg = {
        btype: [sum(frac[i][btype][b] for i in range(n)) / n for b in range(count)]
        for btype, count in block_types
    }

    rel = []
    for i in range(n):
        ri = {}
        for btype, count in block_types:
            ri[btype] = [
                frac[i][btype][b] / avg[btype][b] if avg[btype][b] > 1e-12 else 1.0
                for b in range(count)
            ]
        rel.append(ri)

    for btype, count in block_types:
        for b in range(count):
            imps = [rel[i][btype][b] for i in range(n)]
            peak = max(imps)
            if peak < 1e-9:
                continue
            for i in range(n):
                weights[i][btype][b] = round(max(imps[i] / peak, 0.1), 3)
    return weights


def setup_routes(base_dir: str):
    global _PRESETS_FILE, _ROUTES_REGISTERED
    _PRESETS_FILE = os.path.join(base_dir, "magic_lora_loader_presets.json")

    if _ROUTES_REGISTERED:
        return

    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/crt-pll/api/loras")
    async def _get_loras(request):
        names = folder_paths.get_filename_list("loras")
        return web.json_response([{"file": n} for n in names])

    @PromptServer.instance.routes.get("/crt-pll/api/loras/info")
    async def _get_lora_info(request):
        file = request.rel_url.query.get("file", "")
        if not file:
            return web.json_response({}, status=200)
        info = get_model_info_file_data(file, "loras", default={})
        return web.json_response(info or {})

    @PromptServer.instance.routes.get("/crt-pll/api/presets")
    async def _get_presets(request):
        mt = request.rel_url.query.get("model_type", _DEFAULT_MT)
        return web.json_response({"status": 200, "data": _load_presets(mt)})

    @PromptServer.instance.routes.post("/crt-pll/api/presets")
    async def _save_preset(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"status": 400, "error": "Invalid JSON"}, status=400)
        name = body.get("name", "").strip()
        data = body.get("data")
        mt = body.get("model_type", _DEFAULT_MT)
        if not name or data is None:
            return web.json_response(
                {"status": 400, "error": "name and data required"}, status=400
            )
        presets = _load_presets(mt)
        presets[name] = data
        _save_presets(mt, presets)
        return web.json_response({"status": 200, "data": presets})

    @PromptServer.instance.routes.delete("/crt-pll/api/presets/{name}")
    async def _delete_preset(request):
        name = request.match_info.get("name", "")
        mt = request.rel_url.query.get("model_type", _DEFAULT_MT)
        presets = _load_presets(mt)
        if name in presets:
            del presets[name]
            _save_presets(mt, presets)
        return web.json_response({"status": 200, "data": presets})

    async def _automix_handler(request):
        try:
            import comfy.utils

            body = await request.json()
            loras = body.get("loras", [])
            model_type = body.get("model_type", _DEFAULT_MT)
            if model_type not in _ARCH:
                model_type = _DEFAULT_MT

            flat = _flat_weights(model_type)
            profiles = []
            errors = []
            for item in loras:
                name = item.get("name")
                if not name:
                    profiles.append(None)
                    continue
                path = folder_paths.get_full_path("loras", name)
                if not path:
                    errors.append(f"LoRA not found: {name}")
                    profiles.append(None)
                    continue
                try:
                    lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
                    profiles.append(_compute_profile(lora_sd, model_type=model_type))
                    del lora_sd
                except Exception as e:
                    errors.append(f"Error reading {name}: {e}")
                    profiles.append(None)

            valid = [p if p is not None else dict(flat) for p in profiles]
            weights = _competitive_weights(valid, model_type=model_type)

            return web.json_response(
                {
                    "profiles": [_normalise_profile(p) for p in valid],
                    "weights": weights,
                    "errors": errors,
                }
            )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _block_profile_handler(request):
        try:
            import comfy.utils

            body = await request.json()
            name = body.get("name")
            model_type = body.get("model_type", _DEFAULT_MT)
            if model_type not in _ARCH:
                model_type = _DEFAULT_MT

            if not name:
                return web.json_response({"error": "no name"}, status=400)

            path = folder_paths.get_full_path("loras", name)
            if not path:
                return web.json_response({"error": f"LoRA not found: {name}"}, status=404)

            lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
            profile = _compute_profile(lora_sd, model_type=model_type)
            del lora_sd
            return web.json_response({"profile": _normalise_profile(profile)})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    try:
        PromptServer.instance.app.router.add_post("/crt-pll/api/automix", _automix_handler)
        PromptServer.instance.app.router.add_post("/crt-pll/api/block_profile", _block_profile_handler)
        print("[crt-pll] presets + automix routes registered")
    except RuntimeError:
        pass

    _ROUTES_REGISTERED = True
