import os
import gc
import json
import random
import logging
import threading
import dataclasses
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

import folder_paths
import comfy.model_management
from comfy.utils import ProgressBar

if os.name == "nt":
    # Windows defaults to cp1252 for text I/O, which crashes on CJK lyrics when
    # yue2 writes JSON/artifacts. Force UTF-8 for this process and any children.
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_VERSIONS = {
    "YuE2-3B (Recommended)": "m-a-p/YuE2-3B",
}

VAE_VERSIONS = {
    "YuE2-Vae (Recommended)": "m-a-p/YuE2-Vae",
    "YuE2-Vae-legacy (Benchmark)": "m-a-p/YuE2-Vae-legacy",
}

DEFAULT_LYRICS = """[Verse]
Neon fades along the lane
Footsteps keep the time of rain
Fold the night and leave it here
Morning has a sky to clear

[Chorus]
Let the day come into view
Every road begins with you
Hold a little room for light
We will sing beyond the night"""

DEFAULT_STYLE = (
    "English, warm piano pop, expressive female voice, acoustic piano, "
    "rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM"
)

# One generated codec token is one acoustic latent frame; the VAE emits
# 1920 samples per frame at 48 kHz, i.e. 25 frames per second.
FRAMES_PER_SECOND = 25


def _resolve_save_dir(save_path):
    if save_path.startswith("./"):
        return os.path.join(os.getcwd(), save_path[2:])
    return save_path


def _resolve_lora_weights(lora_path):
    path = Path(lora_path).expanduser()
    if path.is_dir():
        return path / "adapter_model.safetensors", path / "adapter_config.json"
    return path, None


def load_lora_state(lora_path, scale=1.0):
    """Load a peft LoRA adapter (directory or .safetensors) into a mergeable dict.

    Returns ``{"scale", "scaling", "deltas": {module_name: (A, B)}}`` where
    ``module_name`` is relative to the base ``YuE2ForCausalLM`` (e.g.
    ``model.layers.0.nar_self_attn.q_proj``).
    """
    from safetensors.torch import load_file

    weights, config_path = _resolve_lora_weights(lora_path)
    if not weights.is_file():
        raise FileNotFoundError(f"YuE LoRA weights not found: {weights}")

    scaling = 1.0
    if config_path is not None and config_path.is_file():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        r = int(cfg.get("r") or 0)
        alpha = float(cfg.get("lora_alpha") or r)
        if r:
            scaling = alpha / r

    state = load_file(str(weights))
    pairs = {}
    for key, value in state.items():
        if key.endswith(".lora_A.weight"):
            pairs.setdefault(key[: -len(".lora_A.weight")], {})["A"] = value
        elif key.endswith(".lora_B.weight"):
            pairs.setdefault(key[: -len(".lora_B.weight")], {})["B"] = value

    deltas = {}
    for name, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        if name.startswith("base_model.model."):
            name = name[len("base_model.model."):]
        deltas[name] = (ab["A"], ab["B"])

    logger.info(f"YuE: loaded LoRA {weights.name} ({len(deltas)} modules, scaling {scaling:.3f})")
    return {"path": str(lora_path), "scale": float(scale), "scaling": scaling, "deltas": deltas}


def apply_lora(model, lora, sign):
    """Add (sign=+1) or remove (sign=-1) the LoRA delta on the base weights."""
    if not lora:
        return 0
    factor = sign * float(lora["scale"]) * float(lora["scaling"])
    applied = 0
    for name, (a, b) in lora["deltas"].items():
        try:
            module = model.get_submodule(name)
        except AttributeError:
            continue
        delta = (b.to(torch.float32) @ a.to(torch.float32)) * factor
        module.weight.data.add_(delta.to(module.weight))
        applied += 1
    return applied


class YuELoraLoader:
    """Load a trained YuE2 LoRA adapter for the Music Generator node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a LoRA adapter directory (adapter_model.safetensors) or a .safetensors file.",
                }),
                "scale": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("YUE_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "load_lora"
    CATEGORY = "CRT/Audio"

    def load_lora(self, lora_path, scale=1.0):
        if not lora_path or not lora_path.strip():
            raise ValueError("YuE LoRA: lora_path is empty")
        return (load_lora_state(lora_path.strip(), scale),)


@contextmanager
def _optimized_attention(selected):
    """Route YuE2's acoustic (NAR) attention through ComfyUI's optimized kernels.

    YuE2 already uses PyTorch fused SDPA (FlashAttention) for the AR stage and a
    CUDA graph with a native varlen flash kernel for decoding, so only the
    non-causal synthesis attention is redirected here. Causal or non-CUDA calls
    keep the native implementation.
    """
    if selected == "native":
        yield
        return

    try:
        import yue2.nar as nar
        from comfy.ldm.modules import attention as comfy_attention
    except Exception as e:
        logger.warning(f"YuE: could not load ComfyUI attention ({e}); using native SDPA")
        yield
        return

    if selected == "comfy":
        func = comfy_attention.optimized_attention
    else:
        func = comfy_attention.get_attention_function(selected, default=None)
    if func is None:
        logger.warning(f"YuE: '{selected}' attention is unavailable; using native SDPA")
        yield
        return

    original = nar.attention

    def patched(q, k, v, *, causal=False, backend="sdpa", query_chunk_size=None):
        if causal or q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
            return original(q, k, v, causal=causal, backend=backend, query_chunk_size=query_chunk_size)
        qh = q.transpose(0, 1).unsqueeze(0).contiguous()
        kh = k.transpose(0, 1).unsqueeze(0).contiguous()
        vh = v.transpose(0, 1).unsqueeze(0).contiguous()
        try:
            out = func(qh, kh, vh, q.shape[1], skip_reshape=True, skip_output_reshape=True,
                       enable_gqa=q.shape[1] != k.shape[1])
        except Exception as e:
            logger.warning(f"YuE: '{selected}' attention failed ({e}); using native SDPA")
            return original(q, k, v, causal=causal, backend=backend, query_chunk_size=query_chunk_size)
        return out[0].transpose(0, 1)

    logger.info(f"YuE: using ComfyUI '{selected}' attention for acoustic synthesis")
    nar.attention = patched
    try:
        yield
    finally:
        nar.attention = original


@contextmanager
def _model_residency(pipeline, mode):
    """Control where the model sits before the VAE decoder runs.

    'cpu' keeps it in system RAM (the pipeline default), 'disk' drops it so it is
    reloaded from the checkpoint on the next run, and 'none' leaves it on the GPU
    during decoding to skip the device round-trip.
    """
    if mode == "cpu":
        yield
        return

    original = pipeline.decode

    def decode(latents, **kwargs):
        saved = pipeline._model
        pipeline._model = None
        if mode == "disk":
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        try:
            return original(latents, **kwargs)
        finally:
            if mode == "none":
                pipeline._model = saved

    pipeline.decode = decode
    try:
        yield
    finally:
        pipeline.decode = original


def _patch_ar_attention():
    """Avoid the native flash kernel when the PyTorch build lacks it.

    YuE2's CUDA graph selects flash from the operator schema alone, but some
    CUDA builds ship the op without the compiled kernel. When the build reports
    no flash attention, default the graph to cuDNN (or public SDPA).
    """
    import yue2.cuda_graph as cuda_graph

    if getattr(cuda_graph.GraphAR, "_crt_flash_guard", False):
        return
    available = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    if available is None or available():
        return

    original = cuda_graph.GraphAR

    class GraphAR(original):
        def __init__(self, model, prefixes, max_tokens, *, attention_backend="auto", **kwargs):
            if attention_backend == "auto":
                attention_backend = "cudnn" if torch.backends.cudnn.is_available() else "sdpa"
            super().__init__(model, prefixes, max_tokens, attention_backend=attention_backend, **kwargs)

    GraphAR._crt_flash_guard = True
    cuda_graph.GraphAR = GraphAR
    logger.info("YuE: this PyTorch build has no flash kernel; AR graph will use cuDNN/SDPA")


class YuEModelManager:
    """Caches loaded YuE2 pipelines so repeated runs skip model loading."""

    def __init__(self):
        self.pipelines = {}
        self.lock = threading.Lock()

    def load(self, model, vae, backend, quantization, memory_budget_gib, offload_ar, auto_download):
        key = (model, vae, backend, quantization, float(memory_budget_gib), bool(offload_ar))
        with self.lock:
            cached = self.pipelines.get(key)
        if cached is not None:
            logger.info(f"YuE: reusing loaded pipeline {key}")
            return cached, key

        # A different configuration was requested: free the previous pipeline so
        # VRAM does not accumulate across model, decoder or backend changes.
        self.close_all()

        try:
            from yue2 import YuE2Pipeline
        except ImportError as e:
            raise ImportError(
                "YuE2 is not installed. Install it into the ComfyUI Python with: "
                "pip install --no-deps git+https://github.com/multimodal-art-projection/YuE.git"
            ) from e

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"YuE: loading {model} / {vae} on {device} (backend={backend}, quantization={quantization})")
        pipeline = YuE2Pipeline.from_pretrained(
            model,
            vae=vae,
            device=device,
            backend=backend,
            quantization=quantization,
            memory_budget_gib=float(memory_budget_gib),
            offload_ar=bool(offload_ar),
            local_files_only=not auto_download,
            progress=True,
        )
        with self.lock:
            self.pipelines[key] = pipeline
        return pipeline, key

    def close(self, key):
        with self.lock:
            pipeline = self.pipelines.pop(key, None)
        if pipeline is not None:
            logger.info(f"YuE: unloading pipeline {key}")
            try:
                pipeline.close()
            except Exception as e:
                logger.warning(f"YuE: pipeline close failed: {e}")
        self._empty_cache()

    def close_all(self):
        with self.lock:
            pipelines = list(self.pipelines.values())
            self.pipelines.clear()
        for pipeline in pipelines:
            try:
                pipeline.close()
            except Exception:
                pass
        self._empty_cache()

    @staticmethod
    def _empty_cache():
        gc.collect()
        if torch.cuda.is_available():
            # YuE2 caps the process CUDA memory while loaded; release that cap on unload.
            torch.cuda.set_per_process_memory_fraction(1.0)
            torch.cuda.empty_cache()


class YuEMusicGenerator:
    def __init__(self):
        self.manager = YuEModelManager()
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(MODEL_VERSIONS.keys()), {"default": "YuE2-3B (Recommended)"}),
                "vae": (list(VAE_VERSIONS.keys()), {"default": "YuE2-Vae (Recommended)"}),
                "style": ("STRING", {"multiline": True, "default": DEFAULT_STYLE}),
                "lyrics": ("STRING", {"multiline": True, "default": DEFAULT_LYRICS}),
                "cot": (["full", "melody", "off"], {"default": "full"}),
                "seed": ("INT", {"default": 831001, "min": 0, "max": 0x7fffffffffffffff}),
                "offload_after_run": ("BOOLEAN", {"default": False, "tooltip": "Offload this node's models from VRAM after the final step; they reload on the next run."}),
                "auto_download": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "style_override": ("STRING", {"forceInput": True}),
                "lyrics_override": ("STRING", {"forceInput": True}),
                "seed_override": ("INT", {"forceInput": True}),
                "abc": ("STRING", {"multiline": True, "default": ""}),
                "cfg_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.05}),
                "ode_steps": ("INT", {"default": 32, "min": 1, "max": 64, "step": 1}),
                "vae_core_frames": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 128}),
                "max_length_seconds": ("INT", {"default": 0, "min": 0, "max": 360, "step": 5}),
                "override_sampling": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 3.0, "step": 0.01}),
                "semantic_min_tokens": ("INT", {"default": 200, "min": 0, "max": 16000, "step": 50}),
                "semantic_max_tokens": ("INT", {"default": 9000, "min": 200, "max": 16000, "step": 100}),
                "abc_max_tokens": ("INT", {"default": 4096, "min": 32, "max": 8192, "step": 32}),
                "attention": (["comfy", "native", "sage", "flash"], {"default": "comfy"}),
                "backend": (["torch", "torch-eager", "vllm"], {"default": "torch"}),
                "quantization": (["none", "fp8"], {"default": "none"}),
                "memory_budget_gib": ("FLOAT", {"default": 24.0, "min": 3.0, "max": 1024.0, "step": 1.0}),
                "offload_ar": ("BOOLEAN", {"default": False}),
                "unload_model_before_vae": (["disk", "cpu", "none"], {"default": "cpu"}),
                "save_output": ("BOOLEAN", {"default": True}),
                "save_path": ("STRING", {"default": "./ComfyUI/output/Music_YuE2"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "score")
    OUTPUT_NODE = True
    FUNCTION = "generate_music"
    CATEGORY = "CRT/Audio"

    def generate_music(
        self, model, vae, style, lyrics, cot, seed, offload_after_run, auto_download,
        style_override=None, lyrics_override=None, seed_override=None, abc="",
        cfg_scale=-1.0, ode_steps=32, vae_core_frames=0, max_length_seconds=0,
        override_sampling=False, temperature=1.0, top_p=0.95, top_k=100, repetition_penalty=1.2,
        semantic_min_tokens=200, semantic_max_tokens=9000, abc_max_tokens=4096,
        attention="comfy", backend="torch", quantization="none", memory_budget_gib=24.0,
        offload_ar=False, unload_model_before_vae="cpu", save_output=False, save_path="",
        lora=None,
    ):
        active_style = style_override if style_override else style
        active_lyrics = lyrics_override if lyrics_override else lyrics
        active_seed = int(seed_override if seed_override is not None else seed)
        active_abc = abc.strip() or None
        active_cfg = None if cfg_scale < 0 else float(cfg_scale)

        if active_abc and cot == "off":
            raise ValueError("YuE: a supplied ABC score requires cot='full' or cot='melody'")

        model_id = MODEL_VERSIONS[model]
        vae_id = VAE_VERSIONS[vae]

        pipeline, cache_key = self.manager.load(
            model_id, vae_id, backend, quantization, memory_budget_gib, offload_ar, auto_download
        )

        _patch_ar_attention()

        config = pipeline.generation_config
        semantic = config.semantic
        if max_length_seconds > 0:
            # Cap the autoregressive codec tokens that define the song length.
            max_tokens = max(1, min(semantic.max_tokens, int(round(max_length_seconds * FRAMES_PER_SECOND))))
            semantic = dataclasses.replace(semantic, max_tokens=max_tokens,
                                           min_tokens=min(semantic.min_tokens, max_tokens))
        if semantic is not config.semantic or ode_steps != config.ode_steps:
            from yue2.protocol import GenerationConfig
            pipeline.generation_config = GenerationConfig(abc=config.abc, semantic=semantic,
                                                          ode_steps=int(ode_steps))
        if vae_core_frames > 0:
            pipeline.vae_core_frames = int(vae_core_frames)

        sampling_kwargs = {}
        if override_sampling:
            semantic_sampling = {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "repetition_penalty": float(repetition_penalty),
                "min_tokens": int(semantic_min_tokens),
                "max_tokens": int(semantic_max_tokens),
            }
            if max_length_seconds > 0:
                cap = max(1, int(round(max_length_seconds * FRAMES_PER_SECOND)))
                semantic_sampling["max_tokens"] = min(semantic_sampling["max_tokens"], cap)
                semantic_sampling["min_tokens"] = min(semantic_sampling["min_tokens"],
                                                       semantic_sampling["max_tokens"])
            sampling_kwargs["semantic_sampling"] = semantic_sampling
            sampling_kwargs["abc_sampling"] = {"max_tokens": int(abc_max_tokens)}

        abc_max = sampling_kwargs.get("abc_sampling", {}).get("max_tokens",
                                                             pipeline.generation_config.abc.max_tokens)
        semantic_max = sampling_kwargs.get("semantic_sampling", {}).get(
            "max_tokens", pipeline.generation_config.semantic.max_tokens)
        total_tokens = abc_max + semantic_max
        progress = ProgressBar(total_tokens)

        def on_token(phase, token):
            progress.update(1)

        request = {
            "style": active_style,
            "lyrics": active_lyrics,
            "cot": cot,
            "seed": active_seed,
            "abc": active_abc,
            "cfg_scale": active_cfg,
        }

        logger.info(f"YuE: generating (cot={cot}, seed={active_seed}, model={model_id}, vae={vae_id}, "
                    f"ode_steps={ode_steps}, attention={attention}, lora={bool(lora)})")
        lora_model = None
        if lora:
            lora_model = pipeline._load_model()
            applied = apply_lora(lora_model, lora, +1)
            logger.info(f"YuE: applying LoRA on {applied} modules (scale={lora['scale']}, scaling={lora['scaling']:.3f})")
        try:
            with _optimized_attention(attention):
                with _model_residency(pipeline, unload_model_before_vae):
                    song = pipeline(
                        **request,
                        **sampling_kwargs,
                        on_token=on_token,
                        cancelled=comfy.model_management.processing_interrupted,
                    )
        except InterruptedError:
            comfy.model_management.throw_exception_if_processing_interrupted()
            raise
        finally:
            if lora_model is not None:
                apply_lora(lora_model, lora, -1)
            if offload_after_run:
                self.manager.close(cache_key)

        audio_np = np.asarray(song.audio, dtype=np.float32)
        if audio_np.ndim == 1:
            audio_np = audio_np[:, None]
        waveform = torch.from_numpy(audio_np.T.copy()).unsqueeze(0).contiguous()
        sample_rate = int(song.sample_rate)

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            self.prefix_append, self.output_dir
        )
        preview_file = f"{filename}_{counter:05}_.wav"
        sf.write(os.path.join(full_output_folder, preview_file), audio_np, sample_rate)

        if save_output and save_path:
            saved_path = self._save_output(audio_np, sample_rate, save_path)
            if saved_path:
                self._save_preset(saved_path, {
                    "audio": os.path.basename(saved_path),
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "settings": {
                        "model": model,
                        "vae": vae,
                        "style": active_style,
                        "lyrics": active_lyrics,
                        "cot": cot,
                        "seed": active_seed,
                        "abc": abc,
                        "cfg_scale": cfg_scale,
                        "ode_steps": ode_steps,
                        "vae_core_frames": vae_core_frames,
                        "max_length_seconds": max_length_seconds,
                        "override_sampling": override_sampling,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                        "semantic_min_tokens": semantic_min_tokens,
                        "semantic_max_tokens": semantic_max_tokens,
                        "abc_max_tokens": abc_max_tokens,
                        "attention": attention,
                        "backend": backend,
                        "quantization": quantization,
                        "memory_budget_gib": memory_budget_gib,
                        "offload_ar": offload_ar,
                        "unload_model_before_vae": unload_model_before_vae,
                        "save_output": save_output,
                        "save_path": save_path,
                    },
                })

        mono = audio_np.mean(axis=1) if audio_np.ndim > 1 else audio_np
        peak = 20 * np.log10(np.max(np.abs(mono))) if np.max(np.abs(mono)) > 0 else -100
        rms = 20 * np.log10(np.sqrt(np.mean(mono ** 2))) if np.mean(mono ** 2) > 0 else -100

        truncated = song.truncated
        if any(truncated.values()):
            logger.warning(f"YuE: generation hit a token limit {truncated}")

        score = song.abc or ""
        return {
            "ui": {
                "audio": [{"filename": preview_file, "subfolder": subfolder, "type": self.type}],
                "metrics": [{"peak": f"{peak:.1f}", "rms": f"{rms:.1f}"}],
                "style": [active_style],
                "lyrics": [active_lyrics],
                "generation_info": [{
                    "model": model,
                    "vae": vae,
                    "cot": cot,
                    "seed": str(active_seed),
                    "truncated": bool(any(truncated.values())),
                }],
            },
            "result": ({"waveform": waveform, "sample_rate": sample_rate}, score),
        }

    @staticmethod
    def _save_output(audio_np, sample_rate, save_path):
        try:
            final_save_path = _resolve_save_dir(save_path)
            os.makedirs(final_save_path, exist_ok=True)

            index = 1
            while True:
                file_path = os.path.join(final_save_path, f"YuE_{index:04d}.wav")
                if not os.path.exists(file_path):
                    break
                index += 1
            sf.write(file_path, audio_np, sample_rate)
            logger.info(f"YuE: saved audio to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"YuE: failed to save output file: {e}")
            return None

    @staticmethod
    def _save_preset(audio_path, settings):
        try:
            preset_path = os.path.splitext(audio_path)[0] + ".json"
            with open(preset_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            logger.info(f"YuE: saved run preset to {preset_path}")
        except Exception as e:
            logger.error(f"YuE: failed to save run preset: {e}")


NODE_CLASS_MAPPINGS = {"YuEMusicGenerator": YuEMusicGenerator, "YuELoraLoader": YuELoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {
    "YuEMusicGenerator": "YuE Music Generator",
    "YuELoraLoader": "YuE LoRA Loader",
}

_ROUTES_REGISTERED = False


def _register_routes():
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return

    async def _list_runs(request):
        save_path = request.rel_url.query.get("path", "")
        folder = _resolve_save_dir(save_path) if save_path else None
        if not folder or not os.path.isdir(folder):
            return web.json_response({"runs": []})
        runs = [os.path.splitext(entry)[0] for entry in os.listdir(folder)
                if entry.lower().endswith(".json")]
        runs.sort(reverse=True)
        return web.json_response({"runs": runs})

    async def _get_run(request):
        save_path = request.rel_url.query.get("path", "")
        name = request.rel_url.query.get("name", "")
        folder = _resolve_save_dir(save_path) if save_path else None
        if not folder or not name or os.path.basename(name) != name:
            return web.json_response({"error": "invalid request"}, status=400)
        file_path = os.path.join(folder, name + ".json")
        if not os.path.isfile(file_path):
            return web.json_response({"error": "not found"}, status=404)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response(data)

    # A reloaded module can be re-imported after the aiohttp app has already
    # started and frozen its router, so routes added to PromptServer.routes are
    # never picked up. Register on the live router, briefly unfreezing it so a
    # re-imported module can still add its endpoints.
    router = server.app.router
    frozen = getattr(router, "_frozen", False)
    if frozen:
        router._frozen = False
    try:
        router.add_get("/crt/yue2/runs", _list_runs)
        router.add_get("/crt/yue2/run", _get_run)
    except RuntimeError:
        # Already registered by an earlier reload; the previous handler is equivalent.
        pass
    finally:
        if frozen:
            router.freeze()

    _ROUTES_REGISTERED = True


try:
    _register_routes()
except Exception as e:
    logger.warning(f"YuE: could not register run routes: {e}")
