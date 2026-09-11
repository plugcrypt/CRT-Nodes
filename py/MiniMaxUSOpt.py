"""MiniMaxUSOpt: embedded H3 speed optimizations (Sol / ChunkFF / Spectrum).

Standalone optimization stack for the MiniMax H3 Unified Sampler. Nothing here
invokes external custom nodes: the Sol-Attn Triton kernels and the Spectrum
forecast engine are vendored under ``py/usopt_sol_kernel`` and
``py/usopt_spectrum_h3`` (Sol-Attn MIT, Spectrum GPL-3.0+ — keep the notices).

apply_us_opt(model, ...) applies, in dependency order:
  1. Scheduled Sol attention  — sparsifies attention on high-noise steps
  2. Chunked feed-forward     — splits MLP activations to cut peak VRAM
  3. Spectrum forecast        — Chebyshev ridge forecasting to skip NFEs
"""

import logging

log = logging.getLogger(__name__)

SOL_ARCHES = {(8, 6), (8, 9), (9, 0), (12, 0), (12, 1)}

# Populated lazily by _ensure_sol_attn(); the per-block attention closures
# resolve this module global at call time.
sol_attn = None


def _ensure_sol_attn():
    global sol_attn
    if sol_attn is None:
        from .usopt_sol_kernel import sol_attn as _kernel
        sol_attn = _kernel
    return sol_attn


# --- Chunked feed-forward ----------------------------------------------------

class _ChunkLog:
    def __init__(self):
        self.active = False

    def hit(self, tokens, chunks):
        if not self.active:
            log.info("[MiniMax H3 FFN] active (%d tokens, %d chunks)", tokens, chunks)
            self.active = True


def _make_chunked_forward(original_forward, chunks, min_tokens, chunk_log):
    def forward(x):
        import torch

        if x.ndim != 2 or x.shape[0] < min_tokens or x.requires_grad:
            return original_forward(x)

        chunk_log.hit(x.shape[0], chunks)
        output = torch.empty_like(x)
        offset = 0
        for part in x.chunk(chunks, dim=0):
            end = offset + part.shape[0]
            output[offset:end].copy_(original_forward(part))
            offset = end
        return output

    forward._minimax_h3_ffn_fallback = original_forward
    return forward


def apply_chunk_feed_forward(model, chunks=2, min_tokens=8192, log_fn=None):
    import torch

    chunks = int(chunks)
    if chunks == 1:
        return model

    diffusion_model = model.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    token_refiner = getattr(diffusion_model, "token_refiner", None)
    refiner_blocks = getattr(token_refiner, "blocks", None)
    if diffusion_model.__class__.__name__ != "MiniMaxH3Model" or blocks is None or refiner_blocks is None:
        (log_fn or log.warning)("[MiniMaxUSOpt] FFN: expected a MiniMax H3 model; returning it unchanged")
        return model

    patched = model.clone()
    paths = [f"diffusion_model.blocks.{i}.mlp.forward" for i in range(len(blocks))]
    paths.extend(f"diffusion_model.token_refiner.blocks.{i}.mlp.forward" for i in range(len(refiner_blocks)))
    chunk_log = _ChunkLog()
    for path in paths:
        original_forward = patched.get_model_object(path)
        if hasattr(original_forward, "_minimax_h3_ffn_fallback"):
            original_forward = original_forward._minimax_h3_ffn_fallback
        patched.add_object_patch(
            path,
            _make_chunked_forward(original_forward, chunks, int(min_tokens), chunk_log),
        )

    (log_fn or log.info)(f"[MiniMaxUSOpt] Chunk FeedForward: patched {len(paths)} MLPs (chunks={chunks}, min_tokens={int(min_tokens)})")
    return patched


# --- Scheduled Sol attention ---------------------------------------------------

class _Unsupported(Exception):
    pass


class _SolLog:
    def __init__(self):
        self.active = False
        self.fallbacks = set()

    def hit(self, tokens):
        if not self.active:
            log.info("[MiniMax H3 Sol] active (%d tokens)", tokens)
            self.active = True

    def miss(self, reason):
        if reason not in self.fallbacks:
            self.fallbacks.add(reason)
            log.info("[MiniMax H3 Sol] dense fallback: %s", reason)


def _make_sol_attention_forward(attn, fallback_forward, tau, min_tokens, strict, sol_log,
                                thresh_type="diag", dense_percent=0.0, progress_fn=None,
                                int8_qk=False, int8_pv=False, sink_conditioning="exact_kv"):
    heads, head_dim = attn.heads, attn.head_dim
    inner = heads * head_dim

    def forward(x, rope_freqs=None, transformer_options={}):
        import torch

        handoff = isinstance(x, list) and len(x) == 1 and torch.is_tensor(x[0])
        tensor = x[0] if handoff else x
        handoff_released = False
        try:
            if not torch.is_tensor(tensor):
                raise _Unsupported("attention input is not a tensor")
            s = tensor.shape[0]
            if tensor.ndim != 2 or s < min_tokens or tensor.requires_grad:
                raise _Unsupported("below min_tokens or autograd requested")
            if tensor.dtype != torch.bfloat16 or tensor.device.type != "cuda":
                raise _Unsupported("requires bfloat16 on CUDA")
            if head_dim != 128:
                raise _Unsupported(f"head_dim {head_dim} != 128")
            arch = torch.cuda.get_device_capability(tensor.device)
            if arch not in SOL_ARCHES:
                raise _Unsupported(f"unsupported SM{arch[0]}{arch[1]}")

            sigmas = (transformer_options or {}).get("sigmas")
            sigma = float(sigmas.flatten()[0]) if torch.is_tensor(sigmas) and sigmas.numel() > 0 else None
            if dense_percent > 0.0 and progress_fn is not None and progress_fn(sigma) < dense_percent:
                raise _Unsupported(f"dense first {dense_percent:.0%} of sampling")

            sink_blocks = (0, 0)
            sink_q = (0, 0)
            if sink_conditioning != "off":
                span = (transformer_options or {}).get("sol_h3_video_span")
                if span is not None:
                    video_start, video_stop = span
                    if 0 < video_start and s >= video_stop:
                        sink_blocks = (0, (video_start + 63) // 64)
                        if sink_conditioning == "exact_kv_and_rows":
                            sink_q = sink_blocks

            if handoff:
                tensor = x.pop()
            device = tensor.device
            q, k, v = attn.qkv_proj(tensor).split(inner, dim=-1)
            if handoff:
                del tensor
                handoff_released = True
            q = q.view(1, s, heads, head_dim)
            k = k.view(1, s, heads, head_dim)
            v = v.view(1, s, heads, head_dim)
            if rope_freqs is not None:
                import comfy.model_management
                import comfy.quant_ops

                qw = comfy.model_management.cast_to(attn.q_norm.weight, device=device)
                kw = comfy.model_management.cast_to(attn.k_norm.weight, device=device)
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw,
                    epsilon=attn.q_norm.eps,
                    rot_dim=rope_freqs.shape[-3] * 2,
                )
            else:
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            out = _ensure_sol_attn()(q, k, v, tau=tau(sigma) if callable(tau) else tau,
                           thresh_type=thresh_type, int8_qk=int8_qk, int8_pv=int8_pv,
                           sink_blocks=sink_blocks, sink_q=sink_q)
            sol_log.hit(s)
            return attn.out_proj(out.view(s, inner))
        except _Unsupported as e:
            sol_log.miss(str(e))
        except Exception as e:
            if strict:
                raise
            if handoff and not x:
                if handoff_released:
                    raise
                return fallback_forward(
                    tensor,
                    rope_freqs=rope_freqs,
                    transformer_options=transformer_options,
                )
            sol_log.miss(f"{type(e).__name__}: {e}")
        return fallback_forward(x, rope_freqs=rope_freqs, transformer_options=transformer_options)

    forward._minimax_h3_sol_fallback = fallback_forward
    return forward


class _TauSchedule:
    def __init__(self, tau_start, tau_end, curve, sigma_hi, sigma_lo):
        import math

        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.curve = curve
        self.sigma_hi = float(sigma_hi)
        self.span = max(float(sigma_hi) - float(sigma_lo), 1e-8)

    def weight(self, f):
        import math

        if self.curve == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * f)
        if self.curve == "sqrt":
            return math.sqrt(f)
        if self.curve == "smoothstep":
            return f * f * (3.0 - 2.0 * f)
        if self.curve == "exponential":
            return math.expm1(3.0 * f) / math.expm1(3.0)
        if self.curve == "step":
            return 1.0 if f >= 0.5 else 0.0
        return f

    def progress(self, sigma):
        if sigma is None:
            return 1.0
        return min(max((self.sigma_hi - sigma) / self.span, 0.0), 1.0)

    def tau(self, sigma):
        return self.tau_end + (self.tau_start - self.tau_end) * self.weight(1.0 - self.progress(sigma))


def _make_span_injector(original_forward):
    def forward(x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        try:
            from comfy.ldm.minimax.model import PackedLayout
        except Exception:
            PackedLayout = None
        if isinstance(transformer_options, dict) and PackedLayout is not None:
            payload = minimax_payload or {}
            video_x, audio_x = x[0], x[1]
            signature = (
                context.shape[1],
                video_x.shape[2],
                -(-video_x.shape[3] // 2) * 2,
                -(-video_x.shape[4] // 2) * 2,
                audio_x.shape[-1],
            )
            layout = payload.get("layout")
            try:
                if layout is None or layout.signature != signature:
                    layout = PackedLayout(
                        *signature,
                        keyframes=payload.get("keyframes"),
                        refs=payload.get("refs"),
                        frame_count=payload.get("frame_count"),
                    )
                span = next(((a, b) for a, b, kind in layout.segments if kind == "video"), None)
            except Exception:
                span = None
            if span is not None:
                transformer_options["sol_h3_video_span"] = span
        return original_forward(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)

    forward._minimax_h3_span_fallback = original_forward
    return forward


def _install_span_injector(patched):
    model_forward = patched.get_model_object("diffusion_model._forward")
    if hasattr(model_forward, "_minimax_h3_span_fallback"):
        model_forward = model_forward._minimax_h3_span_fallback
    patched.add_object_patch("diffusion_model._forward", _make_span_injector(model_forward))


def _parse_dense_blocks(spec, count):
    """Parse "0-3,47,-1" into absolute block indices; negatives count from the end."""
    import re

    out = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part)
        if match is None:
            raise ValueError(f"cannot parse dense_blocks entry {part!r}; use indices and ranges like '0-3,47,-1'")
        first = int(match.group(1))
        last = first if match.group(2) is None else int(match.group(2))
        first = first if first >= 0 else count + first
        last = last if last >= 0 else count + last
        if first > last:
            first, last = last, first
        out.update(range(max(first, 0), min(last, count - 1) + 1))
    return out


def apply_scheduled_sol_attention(model, tau_start=1.15, tau_end=0.8, curve="smoothstep",
                                  min_tokens=4096, strict=False, dense_percent=0.0,
                                  thresh_type="diag", int8_qk=False, int8_pv=False,
                                  sink_conditioning="exact_kv", dense_blocks="",
                                  log_fn=None):
    try:
        _ensure_sol_attn()
    except Exception as exc:
        raise RuntimeError(f"Sol Triton kernel import failed ({type(exc).__name__}: {exc}); Triton is required for Sol Attention.") from exc

    diffusion_model = model.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    if diffusion_model.__class__.__name__ != "MiniMaxH3Model" or blocks is None:
        (log_fn or log.warning)("[MiniMaxUSOpt] Sol: expected a MiniMax H3 model; returning it unchanged")
        return model

    patched = model.clone()
    model_sampling = patched.get_model_object("model_sampling")
    schedule = _TauSchedule(
        tau_start,
        tau_end,
        curve,
        float(model_sampling.percent_to_sigma(0.0)),
        float(model_sampling.percent_to_sigma(1.0)),
    )
    _install_span_injector(patched)

    sol_log = _SolLog()
    dense = _parse_dense_blocks(dense_blocks, len(blocks))
    patched_count = 0
    for i in range(len(blocks)):
        if i in dense:
            continue
        attn = patched.get_model_object(f"diffusion_model.blocks.{i}.attn")
        prior = getattr(patched, "object_patches", {}).get(f"diffusion_model.blocks.{i}.attn.forward")
        fallback_forward = prior if prior is not None else attn.forward
        if hasattr(fallback_forward, "_minimax_h3_sol_fallback"):
            fallback_forward = fallback_forward._minimax_h3_sol_fallback
        patched.add_object_patch(
            f"diffusion_model.blocks.{i}.attn.forward",
            _make_sol_attention_forward(
                attn, fallback_forward, schedule.tau, int(min_tokens), bool(strict), sol_log,
                thresh_type, float(dense_percent), schedule.progress, bool(int8_qk),
                bool(int8_pv), sink_conditioning,
            ),
        )
        patched_count += 1

    (log_fn or log.info)(f"[MiniMaxUSOpt] Sol Attention: scheduled tau {float(tau_start):.2f} -> {float(tau_end):.2f} ({curve}) on {patched_count} of {len(blocks)} blocks (min_tokens={int(min_tokens)}, dense={100 * float(dense_percent):.0f}%, thresh={thresh_type})")
    return patched


# --- Spectrum forecast (vendored engine) ---------------------------------------

def apply_spectrum_forecast(model, params, log_fn=None):
    from .usopt_spectrum_h3.nodes import SpectrumApplyMiniMaxH3

    outputs = SpectrumApplyMiniMaxH3().apply(model=model, enabled=True, **params)
    if isinstance(outputs, tuple):
        patched = outputs[0]
    elif isinstance(outputs, list):
        patched = outputs[0]
    else:
        patched = outputs
    if patched is not None:
        (log_fn or log.info)("[MiniMaxUSOpt] Spectrum forecast active.")
    return patched


# --- Public entry ---------------------------------------------------------------

def apply_us_opt(model, enable_sol=False, sol_params=None, enable_chunk_ff=False,
                 chunk_params=None, enable_spectrum=False, spectrum_params=None, log_fn=None):
    """Apply the embedded optimization chain in dependency order."""
    import traceback

    if enable_sol:
        try:
            model = apply_scheduled_sol_attention(model, log_fn=log_fn, **(sol_params or {}))
        except Exception as exc:
            traceback.print_exc()
            (log_fn or log.warning)(f"[MiniMaxUSOpt] Sol Attention failed to apply ({type(exc).__name__}: {exc}); continuing without it.")
    if enable_chunk_ff:
        try:
            model = apply_chunk_feed_forward(model, log_fn=log_fn, **(chunk_params or {}))
        except Exception as exc:
            traceback.print_exc()
            (log_fn or log.warning)(f"[MiniMaxUSOpt] Chunk FeedForward failed to apply ({type(exc).__name__}: {exc}); continuing without it.")
    if enable_spectrum:
        try:
            model = apply_spectrum_forecast(model, spectrum_params or {}, log_fn=log_fn)
        except Exception as exc:
            traceback.print_exc()
            (log_fn or log.warning)(f"[MiniMaxUSOpt] Spectrum forecast failed to apply ({type(exc).__name__}: {exc}); continuing without it.")
    return model
