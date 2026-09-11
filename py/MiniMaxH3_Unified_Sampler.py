import gc
import math
import os
import weakref

import numpy as np
import torch

import folder_paths
import comfy.model_management as mm
import comfy.model_sampling
import comfy.nested_tensor
import comfy.samplers
import comfy.sd
import comfy.utils
import latent_preview
import nodes

from comfy_extras.nodes_custom_sampler import (
    BasicGuider,
    BasicScheduler,
    KSamplerSelect,
    RandomNoise,
    SamplerCustomAdvanced,
)
from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3ImageToVideo,
    MiniMaxH3ReferenceToVideo,
    align_frame_count,
    temporal_shape,
)

from comfy_extras.nodes_audio import vae_decode_audio

from ._cache_fingerprint import stable_fingerprint
from ._minimaxh3_preview import _PreviewFixGuider, _PREVIEW_STATE, apply_h3_preview_override, kickoff_taeh3_download, wipe_all_caches
from .H3_Fun_ControlNet import apply_fun_control
from . import MiniMaxUSOpt

try:
    from . import _minimaxh3_negpip as _NegPiP
except Exception as _negpip_err:                 # optional capability probe
    _NegPiP = None
    print(f"[CRT MiniMaxH3] negpip helper unavailable: {type(_negpip_err).__name__}: {_negpip_err}")

MODE_T2V = "T2V"
MODE_FL2VA = "I2V"
MODE_REF2VA = "R2V"
WORKFLOW_MODES = (MODE_T2V, MODE_FL2VA, MODE_REF2VA)

FL_ASPECT_MODES = ("Preserve First", "Preserve Last", "Optimal", "ControlNet")

FPS = 24.0
SAMPLER_NAME = "res_multistep"
SCHEDULER_NAME = "simple"
STEPS_FULL_DEFAULT = 20
STEPS_TURBO_DEFAULT = 4

SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0

# Official Ref2VA limits: <=9 images, <=3 videos, <=3 standalone audios,
# <=12 files mixed. The native MiniMaxH3ReferenceToVideo node matches these.
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3

ASPECT_RATIOS = [
    "1:1 (Square)",
    "2:3 (Portrait)",
    "3:4 (Portrait)",
    "4:5 (Portrait)",
    "5:7 (Portrait)",
    "5:8 (Portrait)",
    "7:9 (Portrait)",
    "9:16 (Portrait)",
    "9:19 (Portrait)",
    "9:21 (Portrait)",
    "3:2 (Landscape)",
    "4:3 (Landscape)",
    "5:3 (Landscape)",
    "5:4 (Landscape)",
    "7:5 (Landscape)",
    "8:5 (Landscape)",
    "9:7 (Landscape)",
    "16:9 (Landscape)",
    "19:9 (Landscape)",
    "21:9 (Landscape)",
    "Ref Image 1",
    "Ref Video 1",
]


def _active_family(workflow_mode):
    """T2V and FL2VA share the FL2VA checkpoint family."""
    return "ref2va" if str(workflow_mode) == MODE_REF2VA else "fl2va"


# --- Motion Smearing Fix (De-ROPE) ------------------------------------------
# The second pass re-renders the temporally dilated spans of a clip at a slowed
# rate, then recovers the original frame count. H3's latent token clock is 5
# tokens per 17 pixel frames covering (1, 4, 4, 4, 4) frames, and legal clip
# lengths are 17k+5; everything here follows that grid exactly.

DEROPE_MODES = ("adaptive (jerk oracle)", "uniform x4")
DEROPE_PROFILE_MODES = (
    "value |d3| (default)",
    "value |d1| (energy baseline)",
    "trajectory centroid |d3|",
    "value |d3| camera-compensated",
)
DEROPE_PROFILE_DEFAULT = DEROPE_PROFILE_MODES[0]
DEROPE_AUDIO_MODES = (
    "keep pass-1 audio (safe default)",
    "pass-2 foley (seeded)",
    "pass-2 invents audio",
)
# The de-rope UI is a single scenario dropdown. Each preset bundles the whole
# tuning set; the individual knobs (mode / inject / profile / abstain /
# protect_tail / audio) apply only when "Custom" is selected, and are hidden in
# the UI otherwise. "Custom" carries no bundle.
DEROPE_PRESETS = {
    "Balanced (recommended)": {
        "q": 0.75, "d_max": 4, "mode": "adaptive (jerk oracle)",
        "inject": 0.70, "profile": "value |d3| (default)",
        "abstain": 1.5, "protect_tail": 0,
        "audio": "keep pass-1 audio (safe default)",
    },
    "Fast action (sharper)": {
        "q": 0.70, "d_max": 4, "mode": "adaptive (jerk oracle)",
        "inject": 0.50, "profile": "value |d3| (default)",
        "abstain": 1.3, "protect_tail": 0,
        "audio": "keep pass-1 audio (safe default)",
    },
    "Camera motion": {
        "q": 0.75, "d_max": 4, "mode": "adaptive (jerk oracle)",
        "inject": 0.70, "profile": "value |d3| camera-compensated",
        "abstain": 1.5, "protect_tail": 0,
        "audio": "keep pass-1 audio (safe default)",
    },
    "End burst": {
        "q": 0.75, "d_max": 4, "mode": "adaptive (jerk oracle)",
        "inject": 0.70, "profile": "value |d3| (default)",
        "abstain": 1.5, "protect_tail": 17,
        "audio": "keep pass-1 audio (safe default)",
    },
    "Uniform x4 (reference)": {
        "q": 0.75, "d_max": 4, "mode": "uniform x4",
        "inject": 0.70, "profile": "value |d3| (default)",
        "abstain": 0.0, "protect_tail": 0,
        "audio": "keep pass-1 audio (safe default)",
    },
    "Custom": None,
}
DEROPE_PRESET_DEFAULT = "Balanced (recommended)"
# Old preset names (q / d_max only) kept so workflows saved before the scenario
# dropdown still resolve to their original tuning. Not offered in the UI; they
# use the individual knobs for everything else, exactly as they used to.
DEROPE_LEGACY_PRESETS = {
    "balanced (default)": (0.75, 4),
    "max quality (wide plateau)": (0.70, 4),
    "economy (tight spans)": (0.85, 3),
}
DEROPE_UNIFORM_DILATION = 4
DEROPE_INJECT_DEFAULT = 0.70
DEROPE_COST_EXP = 1.7          # per-step time scales as tokens**1.7
# The oracle's q is a quantile, so it always dilates the top (1-q) of
# token-times even on a clip that needs nothing. Below this peak-to-mean
# contrast the de-rope abstains and returns the pass-1 output.
DEROPE_ABSTAIN_DEFAULT = 1.5

# MATLOWAI's rank-16 LoRA trained on the de-rope holdout-infill task. The
# temporal-expansion warm100 checkpoint supersedes the pilot: same recipe but
# scored against real intermediate frames, and the playback pick of the run.
# Applied to the pass-2 model only; see _apply_derope_adapter.
DEROPE_ADAPTER_REPO = "MATLOWAI/MiniMax-H3-Motion-Adapter"
DEROPE_ADAPTER_FILENAME = "minimax_h3_temporal_expansion_warm100_r16.safetensors"
DEROPE_ADAPTER_URL = (f"https://huggingface.co/{DEROPE_ADAPTER_REPO}/resolve/main/"
                      f"temporal_expansion/{DEROPE_ADAPTER_FILENAME}")


def _derope_tok_start_frame(t):
    c, i = divmod(t, 5)
    return c * 17 + (0 if i == 0 else 4 * (i - 1) + 1)


def _derope_frame_token(f, t_lat):
    for t in range(t_lat - 1, -1, -1):
        if _derope_tok_start_frame(t) <= f:
            return t
    return 0


def _derope_legal_ceil(n):
    k = max(2, -(-(n - 5) // 17))
    return 17 * k + 5


def _derope_token_count(frames):
    return (_derope_legal_ceil(frames) - 5) // 17 * 5 + 2


def _derope_video_component(samples):
    z = samples["samples"]
    if hasattr(z, "is_nested") and z.is_nested:
        z = z.tensors[0]
    return z  # (1, 24, t_lat, h, w)


def _derope_value_profile(v, order):
    """Per-token |Δ^order| of the latent VALUES, averaged over c/h/w."""
    j = np.abs(np.diff(v, n=order, axis=2)).mean(axis=(0, 1, 3, 4))
    lead = order // 2
    return np.pad(j, (lead, v.shape[2] - len(j) - lead), mode="edge")


def _derope_trajectory_profile(v, order=3):
    """Per-token |Δ^order| of the energy CENTROID's path.

    The value-domain score is contaminated by motion energy: a textured object
    passing a location makes the value there pulse, and a pulse has large
    differences of every order even at constant velocity. Differentiating the
    centroid TRAJECTORY measures how abruptly the motion changes instead, which
    is what the method actually cares about, and separates jerk from velocity
    where the value domain does not."""
    e = np.abs(v).mean(axis=(0, 1))                # (T, h, w) energy per token
    e = e - e.min(axis=(1, 2), keepdims=True)
    tot = e.sum(axis=(1, 2)) + 1e-8
    ys = np.arange(e.shape[1], dtype=np.float64)[None, :, None]
    xs = np.arange(e.shape[2], dtype=np.float64)[None, None, :]
    cy = (e * ys).sum(axis=(1, 2)) / tot
    cx = (e * xs).sum(axis=(1, 2)) / tot
    path = np.stack([cy, cx], axis=1)              # (T, 2)
    j = np.linalg.norm(np.diff(path, n=order, axis=0), axis=1)
    lead = order // 2
    return np.pad(j, (lead, path.shape[0] - len(j) - lead), mode="edge")


def _derope_camera_compensate(v, max_shift=3):
    """Align each latent frame to its predecessor by the integer (dy, dx) shift
    that minimises their mean absolute difference, accumulated along the clip,
    so a steady pan or scroll reads as stillness and only motion AGAINST the
    camera survives into the differences. Edges wrap (np.roll)."""
    T = v.shape[2]
    out = np.empty_like(v)
    out[:, :, 0] = v[:, :, 0]
    dy = dx = 0
    for t in range(1, T):
        prev = out[:, :, t - 1]
        cur = v[:, :, t]
        best, bs = None, (dy, dx)
        for sy in range(dy - max_shift, dy + max_shift + 1):
            for sx in range(dx - max_shift, dx + max_shift + 1):
                cand = np.roll(cur, (sy, sx), axis=(-2, -1))
                err = float(np.abs(cand - prev).mean())
                if best is None or err < best:
                    best, bs = err, (sy, sx)
        dy, dx = bs
        out[:, :, t] = np.roll(cur, (dy, dx), axis=(-2, -1))
    return out


def _derope_jerk_profile(z, mode=DEROPE_PROFILE_DEFAULT):
    """Per-token motion-overload profile, phase-normalised on the (1,4,4,4,4)
    grid. The default reproduces the original |Δ³| exactly; the other modes
    exist so the detector can be ablated rather than assumed."""
    v = z.detach().float().cpu().numpy()          # (1, 24, T, h, w)
    if mode == "trajectory centroid |d3|":
        prof = _derope_trajectory_profile(v, 3)
    elif mode == "value |d1| (energy baseline)":
        prof = _derope_value_profile(v, 1)
    elif mode == "value |d3| camera-compensated":
        prof = _derope_value_profile(_derope_camera_compensate(v), 3)
    else:
        prof = _derope_value_profile(v, 3)
    for ph in range(5):
        m = prof[ph::5].mean()
        if m > 0:
            prof[ph::5] /= m
    return prof  # (t_lat,)


def _derope_adapter_path():
    """Local path for the Motion Adapter LoRA, downloaded on first use.

    The file is fetched from Hugging Face only when the adapter checkbox is
    enabled; once present it is reused without any network access.
    """
    target = os.path.join(folder_paths.models_dir, "loras", "minimax_h3",
                          DEROPE_ADAPTER_FILENAME)
    if os.path.isfile(target):
        return target
    from .download_progress import download_url_with_progress
    download_url_with_progress(
        DEROPE_ADAPTER_URL, target,
        label=DEROPE_ADAPTER_FILENAME,
        user_agent="CRT-Nodes",
        console_prefix="CRT MiniMaxH3",
    )
    return target


def _apply_derope_adapter(model, log_fn=None):
    """Merge the Motion Adapter LoRA (strength 1.0) into a clone of ``model``
    for pass 2.

    The clone inherits the model's registered patches (including any Fun
    ControlNet tower), so the adapter rides on top of the existing graph.
    """
    lora = comfy.utils.load_torch_file(_derope_adapter_path(), safe_load=True)
    patched, _ = comfy.sd.load_lora_for_models(model, None, lora, 1.0, 0.0)
    if patched is None:
        raise RuntimeError("Motion Adapter LoRA produced no patched model")
    if log_fn is not None:
        log_fn("Derope: Motion Adapter applied to pass 2 (strength 1.0).", level="ok")
    return patched


def _derope_compile_holds(prof, length, q, d_max, ramp=True, bridge=8):
    """Per-token jerk profile -> per-world-frame integer hold map (1 = real
    time, 2..d_max = hold the frame that many times), with C1 ramp shoulders
    and inter-peak valley bridging."""
    prof = np.asarray(prof, dtype=np.float64)
    t_lat = len(prof)
    thr = np.quantile(prof, q)
    tok_d = np.where(prof >= thr, d_max, 1).astype(int)
    if bridge:
        hot = np.where(tok_d == d_max)[0]
        for a, b in zip(hot[:-1], hot[1:]):
            if 1 < b - a <= bridge:
                tok_d[a:b + 1] = d_max
    if ramp:
        for _ in range(d_max - 1):
            left = np.concatenate([[1], tok_d[:-1]])
            right = np.concatenate([tok_d[1:], [1]])
            tok_d = np.maximum(tok_d, np.maximum(left, right) - 1)
    return [int(tok_d[_derope_frame_token(f, t_lat)]) for f in range(length)]


def _derope_expand_hold_map_to_end(holds):
    """Run a trailing expansion span through the last frame when the map ends
    in a short (<= 17-frame) rate-1 tail behind a higher-rate span; kills the
    little jump back to real time at the clip end."""
    holds = [int(h) for h in holds]
    n = len(holds)
    tail = 0
    while tail < n and holds[n - 1 - tail] == 1:
        tail += 1
    if tail == 0 or tail == n or tail > 17:
        return holds
    start = n - tail - 1
    r = holds[start]
    while start > 0 and holds[start - 1] == r:
        start -= 1
    out = holds[:start] + [r] * (n - start)
    m = n - start
    q, rem = divmod(_derope_legal_ceil(sum(out)) - sum(out), m)
    for i in range(start, n):
        out[i] += q
    for i in range(n - rem, n):
        out[i] += 1
    return out


def _derope_smear(images, holds):
    """Duplicate frames by the hold map onto the dilated clock, tail-pad
    snapped to the 17k+5 grid. Returns (smeared_frames, holds_used)."""
    n = images.shape[0]
    holds = [int(h) for h in holds]
    if len(holds) != n:
        raise ValueError(f"hold map covers {len(holds)} frames, batch has {n}")
    holds = list(holds)
    holds[-1] += _derope_legal_ceil(sum(holds)) - sum(holds)
    idx = torch.tensor([i for i, h in enumerate(holds) for _ in range(h)])
    return images[idx].detach().cpu(), holds


def _derope_smear_control(model, length, holds):
    """Re-time active Fun ControlNet inputs onto the dilated clock for pass 2.

    The de-rope second pass re-renders the smeared (slowed) frames, so the
    control video, inpaint mask and source video must be smeared with the same
    hold map. Keeping the original timing would pull the re-render back toward
    the un-dilated clock and fight the repair. Inputs are fitted to the pass-1
    frame count first (repeat last / truncate), matching the native patch's own
    fit. Returns the number of control patches re-timed.
    """
    patches = getattr(model, "_crt_h3_fun_control_patches", None)
    if not patches:
        return 0
    for patch in patches:
        updates = {}
        for attr in ("control_video", "source_video"):
            tensor = getattr(patch, attr, None)
            if tensor is not None and tensor.ndim >= 4:
                idx = torch.arange(length).clamp(max=int(tensor.shape[0]) - 1)
                updates[attr] = _derope_smear(tensor[idx], list(holds))[0]
        mask = getattr(patch, "mask", None)
        if mask is not None and mask.ndim == 3:
            idx = torch.arange(length).clamp(max=int(mask.shape[0]) - 1)
            updates["mask"] = _derope_smear(mask[idx], list(holds))[0]
        for attr, value in updates.items():
            setattr(patch, attr, value)
        patch.control_latent = None
        patch.control_latent_shape = None
    return len(patches)


def _derope_recover(images, holds):
    """Invert the smear by keeping the first frame of every hold group."""
    starts, cur = [], 0
    for h in holds:
        starts.append(cur)
        cur += h
    if cur != images.shape[0]:
        raise ValueError(f"hold map totals {cur} frames, batch has {images.shape[0]}")
    return images[torch.tensor(starts)].cpu()


def _derope_inject_sigmas(model, scheduler, total_steps, inject):
    """Truncated sigma schedule for partial-denoise v2v injection."""
    full = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, total_steps)
    run = max(1, int(round(total_steps * inject)))
    return full[total_steps - run:]


def _derope_torchaudio():
    try:
        import torchaudio
        return torchaudio
    except Exception:
        return None


def _derope_vocoder_rate_for(spec, rate, hop, need):
    """Clamp the phase-vocoder rate so ISTFT can cover `need` samples without
    reaching into the last window's Hann tail."""
    n_in = spec.shape[-1]
    if n_in <= 0 or need <= 0:
        return rate
    return min(float(rate), (n_in * hop) / float(need))


def _derope_audio_runs(holds):
    runs = []
    for h in holds:
        if runs and runs[-1][0] == h:
            runs[-1][1] += 1
        else:
            runs.append([int(h), 1])
    return [(r, c) for r, c in runs]


def _derope_audio_smear(audio, holds, fps=24):
    """Stretch a world-clock track onto the dilated clock (pitch preserved)."""
    torchaudio = _derope_torchaudio()
    if torchaudio is None:
        raise RuntimeError("torchaudio is required to smear audio for derope")
    wav = audio["waveform"].detach().float().cpu()
    sr = int(audio["sample_rate"])
    b, c, n = wav.shape
    x = wav.reshape(b * c, n)
    runs = _derope_audio_runs(holds)
    n_fft, hop = 2048, 512
    window = torch.hann_window(n_fft)
    phase_adv = torch.linspace(0, math.pi * hop, n_fft // 2 + 1)[..., None]
    spf = sr / float(fps)
    xfade = max(1, int(round(0.005 * sr)))
    segs, joins, cursor = [], [], 0.0
    prev_tgt = 0
    for h, count in runs:
        src = count * spf
        tgt = int(round(h * count * spf))
        s0, s1 = int(round(cursor)), int(round(cursor + src))
        cursor += src
        f = 0 if not segs else min(xfade, tgt, prev_tgt, s0 * max(h, 1))
        prev_tgt = tgt
        seg = x[:, s0 - max(1, f // max(h, 1)):min(s1, n)] if f else x[:, s0:min(s1, n)]
        if seg.shape[1] == 0:
            segs.append(torch.zeros(x.shape[0], tgt))
            joins.append(0)
            continue
        if h > 1:
            spec = torch.stft(seg, n_fft, hop, window=window, return_complex=True)
            spec = torchaudio.functional.phase_vocoder(
                spec, _derope_vocoder_rate_for(spec, 1.0 / float(h), hop, f + tgt),
                phase_adv)
            seg = torch.istft(spec, n_fft, hop, window=window, length=f + tgt)
        if seg.shape[1] < f + tgt:
            seg = torch.nn.functional.pad(seg, (0, f + tgt - seg.shape[1]))
        segs.append(seg[:, :f + tgt])
        joins.append(f)
    parts = []
    for seg, f in zip(segs, joins):
        if f:
            t = torch.arange(1, f + 1, dtype=seg.dtype) / f
            prev = parts[-1].clone()
            prev[:, -f:] = (prev[:, -f:] * torch.cos(t * (math.pi / 2))
                            + seg[:, :f] * torch.sin(t * (math.pi / 2)))
            parts[-1] = prev
        parts.append(seg[:, f:])
    y = torch.cat(parts, dim=1)
    return {"waveform": y.reshape(b, c, -1).contiguous(), "sample_rate": sr}


def _derope_audio_recover(audio, holds, fps=24, reference=None, reference_mix=1.0):
    """Compress a dilated-clock track back onto the world clock (pitch kept),
    optionally blended with a real-time reference track."""
    torchaudio = _derope_torchaudio()
    if torchaudio is None:
        raise RuntimeError("torchaudio is required to recover audio for derope")
    wav = audio["waveform"].detach().float().cpu()
    sr = int(audio["sample_rate"])
    b, c, n = wav.shape
    x = wav.reshape(b * c, n)
    runs = _derope_audio_runs(holds)
    n_fft, hop = 2048, 512
    window = torch.hann_window(n_fft)
    phase_adv = torch.linspace(0, math.pi * hop, n_fft // 2 + 1)[..., None]
    spf = sr / float(fps)
    xfade = max(1, int(round(0.005 * sr)))
    segs, joins, cursor = [], [], 0.0
    prev_tgt = 0
    for h, count in runs:
        src = h * count * spf
        tgt = int(round(count * spf))
        s0, s1 = int(round(cursor)), int(round(cursor + src))
        cursor += src
        f = 0 if not segs else min(xfade, tgt, prev_tgt, s0 // max(h, 1))
        prev_tgt = tgt
        seg = x[:, s0 - f * h:min(s1, n)]
        if seg.shape[1] == 0:
            segs.append(torch.zeros(x.shape[0], tgt))
            joins.append(0)
            continue
        if h > 1:
            spec = torch.stft(seg, n_fft, hop, window=window, return_complex=True)
            spec = torchaudio.functional.phase_vocoder(
                spec, _derope_vocoder_rate_for(spec, float(h), hop, f + tgt),
                phase_adv)
            seg = torch.istft(spec, n_fft, hop, window=window, length=f + tgt)
        if seg.shape[1] < f + tgt:
            seg = torch.nn.functional.pad(seg, (0, f + tgt - seg.shape[1]))
        segs.append(seg[:, :f + tgt])
        joins.append(f)
    parts = []
    for seg, f in zip(segs, joins):
        if f:
            t = torch.arange(1, f + 1, dtype=seg.dtype) / f
            prev = parts[-1].clone()
            prev[:, -f:] = (prev[:, -f:] * torch.cos(t * (math.pi / 2))
                            + seg[:, :f] * torch.sin(t * (math.pi / 2)))
            parts[-1] = prev
        parts.append(seg[:, f:])
    y = torch.cat(parts, dim=1)
    if reference is not None and reference_mix > 0:
        ref = reference["waveform"].detach().float().cpu()
        ref = ref.reshape(-1, ref.shape[-1])
        if reference["sample_rate"] != sr:
            ref = torchaudio.functional.resample(ref, reference["sample_rate"], sr)
        n_out = min(y.shape[1], ref.shape[1])
        if ref.shape[0] != y.shape[0]:
            ref = ref[:1].expand(y.shape[0], -1)
        y = (1 - reference_mix) * y[:, :n_out] + reference_mix * ref[:, :n_out]
    return {"waveform": y.reshape(b, c, -1).contiguous(), "sample_rate": sr}


def _derope_audio_vae_encode(audio_vae, audio):
    """Encode an AUDIO dict into the audio-latent half (VAEEncodeAudio path)."""
    torchaudio = _derope_torchaudio()
    waveform = audio["waveform"].detach().float().cpu()
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if vae_sr != sr:
        if torchaudio is None:
            raise RuntimeError("torchaudio is required to resample audio for derope")
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform.movedim(1, -1))
    return {"samples": z}


def _derope_nested_av_latent(video_latent, audio_latent, length, audio_strength=1.0):
    """Wrap VAE-encoded video/audio latents as the nested AV latent the H3
    sampler expects, with an optional audio noise mask so pass 2 keeps part of
    the seeded performance (0.5 = re-render detail, keep bulk timing)."""
    _, t_lat, audio_t = temporal_shape(length)
    if video_latent.shape[2] != t_lat:
        raise ValueError(f"video latent has {video_latent.shape[2]} tokens, "
                         f"length {length} needs {t_lat}")
    audio = torch.zeros(video_latent.shape[0], 32, 2, audio_t,
                        device=video_latent.device, dtype=video_latent.dtype)
    if audio_latent is not None:
        a = audio_latent["samples"] if isinstance(audio_latent, dict) else audio_latent
        a = a.to(device=video_latent.device, dtype=video_latent.dtype)
        if a.dim() == 3:
            a = a[None]
        if a.shape[-1] > audio_t:
            a = a[..., :audio_t]
        elif a.shape[-1] < audio_t:
            a = torch.nn.functional.pad(a, (0, audio_t - a.shape[-1]))
        if a.shape[0] != audio.shape[0]:
            a = a[:1].expand(audio.shape[0], -1, -1, -1)
        audio = a.contiguous()
    out = {"samples": comfy.nested_tensor.NestedTensor((video_latent, audio))}
    if audio_latent is not None and audio_strength < 1.0:
        vid_mask = torch.ones(1, 1, t_lat, video_latent.shape[3],
                              video_latent.shape[4]).to(video_latent.device)
        aud_mask = torch.full((1, 32, 2, audio_t), float(audio_strength))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((vid_mask, aud_mask))
    return out


class CRT_MiniMaxH3USModelsPipe:
    SAMPLER_CLASS = "CRT_MiniMaxH3UnifiedSampler"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE. Decodes generated audio; REF2VA also uses it to encode reference soundtracks."}),
                "clip": ("CLIP", {"tooltip": "Qwen3-VL MiniMax text/image encoder."}),
            },
            "optional": {
                "fl2va_model": ("MODEL", {"lazy": True, "tooltip": "FL2VA diffusion model. Used by T2V and FL2VA modes; loaded only when those modes run."}),
                "fl2va_turbo_model": ("MODEL", {"lazy": True, "tooltip": "FL2VA base merged with the FL2VA Turbo LoRA. Loaded only when Turbo is enabled in an FL2VA-family mode."}),
                "ref2va_model": ("MODEL", {"lazy": True, "tooltip": "REF2VA diffusion model. Used by the REF2VA mode; loaded only when REF2VA runs."}),
                "ref2va_turbo_model": ("MODEL", {"lazy": True, "tooltip": "REF2VA base merged with the REF2VA Turbo LoRA. Loaded only when Turbo is enabled in REF2VA."}),
            },
            "hidden": {
                "minimax_h3_us_prompt": "DYNPROMPT",
                "minimax_h3_us_unique": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("MINIMAXH3_US_MODELS_PIPE",)
    RETURN_NAMES = ("models_pipe",)
    FUNCTION = "build_pipe"
    CATEGORY = "CRT/MiniMaxH3"
    DESCRIPTION = "Bundles the MiniMax H3 model variants, video/audio VAEs, and CLIP. Reads workflow mode and Turbo state from the connected Unified Sampler(s) and lazy-loads - or downloads - only the variant the current run actually uses."

    @classmethod
    def _iter_prompt_nodes(cls, prompt):
        """Yield (node_id, node) over plain dicts and DynamicPrompts alike.

        DYNPROMPT is required so subgraph-expanded nodes ("24:30"-style ids)
        are visible; the original prompt only contains subgraph placeholders.
        """
        if prompt is None:
            return
        get_node = getattr(prompt, "get_node", None)
        all_ids = getattr(prompt, "all_node_ids", None)
        if callable(get_node) and callable(all_ids):
            for node_id in all_ids():
                try:
                    node = get_node(node_id)
                except Exception:
                    continue
                if isinstance(node, dict):
                    yield str(node_id), node
            return
        if isinstance(prompt, dict):
            for node_id, node in prompt.items():
                yield str(node_id), node

    @classmethod
    def _sampler_needs(cls, inputs, unique_id, exact_only=False):
        link = inputs.get("models_pipe")
        if not (isinstance(link, (list, tuple)) and len(link) >= 1):
            return None
        if exact_only and str(link[0]) != str(unique_id):
            return None
        mode = inputs.get("workflow_mode", MODE_FL2VA)
        turbo = bool(inputs.get("turbo", False))
        family = _active_family(mode)
        return f"{family}_turbo_model" if turbo else f"{family}_model"

    @classmethod
    def _required_model_keys(cls, prompt, unique_id):
        """Find Unified Samplers consuming THIS pipe and their needed variants.

        Primary match: sampler links pointing at this pipe's unique id. If the
        graph aliases ids (subgraph expansion variants), fall back to the union
        of every sampler's needs so the run still gets the right weights.
        """
        needs = set()
        fallback_needs = set()
        found_exact = False
        for node_id, node in cls._iter_prompt_nodes(prompt):
            if not isinstance(node, dict):
                continue
            if node.get("class_type") != cls.SAMPLER_CLASS:
                continue
            needed = cls._sampler_needs(node.get("inputs", {}), unique_id, exact_only=True)
            if needed is not None:
                found_exact = True
                needs.add(needed)
                continue
            needed = cls._sampler_needs(node.get("inputs", {}), unique_id, exact_only=False)
            if needed is not None:
                fallback_needs.add(needed)
        if not found_exact:
            if fallback_needs:
                print(
                    "[CRT MiniMaxH3] Models Pipe id not matched exactly; "
                    f"using union of all Unified Samplers in this run: {sorted(fallback_needs)}"
                )
            return fallback_needs
        return needs

    _MODEL_INPUT_KEYS = ("fl2va_model", "fl2va_turbo_model", "ref2va_model", "ref2va_turbo_model")
    # Class-type suffixes that identify a models-pipe node in the prompt graph.
    # Subclasses (the CNET pipe) extend this with their own type string.
    _PIPE_TYPE_KEYS = ("CRT_MiniMaxH3USModelsPipe",)
    # Bump when pipe-building semantics change: older cached outputs (which may
    # have baked None into unevaluated sockets) must invalidate exactly once.
    _PIPE_CACHE_VERSION = "v3-all-sockets"

    @classmethod
    def _connected_model_keys(cls, prompt, unique_id):
        """Which of the four model sockets have links on THIS pipe node.

        Reads the prompt graph: connected optional inputs appear as
        [node_id, slot] link lists; unconnected ones are absent/None.
        Matching is defensive because subgraph expansion mangles ids: exact
        uid, then uid-suffix, then the single pipe, else the union across
        every pipe node (over-requesting a connected loader is harmless).
        """
        connected = set()
        pipe_nodes = []
        for node_id, node in cls._iter_prompt_nodes(prompt):
            if not isinstance(node, dict):
                continue
            ctype = str(node.get("class_type", ""))
            if not ctype.endswith(cls._PIPE_TYPE_KEYS):
                continue
            inputs = node.get("inputs", {})
            pipe_nodes.append((str(node_id), inputs))
            for k in cls._MODEL_INPUT_KEYS:
                link = inputs.get(k)
                if isinstance(link, (list, tuple)) and len(link) >= 1:
                    connected.add(k)
        if not pipe_nodes:
            return set()
        uid = str(unique_id)
        for node_id, inputs in pipe_nodes:
            if node_id == uid or node_id.endswith(":" + uid) or uid.endswith(":" + node_id):
                connected = {
                    k
                    for k in cls._MODEL_INPUT_KEYS
                    if isinstance(inputs.get(k), (list, tuple)) and len(inputs.get(k)) >= 1
                }
                break
        return connected

    @classmethod
    def check_lazy_status(
        cls,
        minimax_h3_us_prompt=None,
        minimax_h3_us_unique=None,
        fl2va_model=None,
        fl2va_turbo_model=None,
        ref2va_model=None,
        ref2va_turbo_model=None,
        **_other_inputs,
    ):
        # Connected-but-unevaluated lazy inputs arrive as None, so requesting a
        # key forces the engine to evaluate that loader branch; unconnected ones
        # surface as a clear missing-input error naming the exact socket.
        available = {
            "fl2va_model": fl2va_model,
            "fl2va_turbo_model": fl2va_turbo_model,
            "ref2va_model": ref2va_model,
            "ref2va_turbo_model": ref2va_turbo_model,
        }
        needed = cls._required_model_keys(minimax_h3_us_prompt, minimax_h3_us_unique)
        # Request EVERY connected loader, not just the active variant: the pipe
        # must carry all four models so mode/turbo switches never see a stale
        # None baked into an unevaluated socket.
        connected = cls._connected_model_keys(minimax_h3_us_prompt, minimax_h3_us_unique)
        needed = needed | connected
        sampler_ids = [
            nid
            for nid, node in cls._iter_prompt_nodes(minimax_h3_us_prompt)
            if isinstance(node, dict) and node.get("class_type") == cls.SAMPLER_CLASS
        ]
        if not needed:
            # No sampler matched this pipe's uid (common when the pipe lives
            # inside a subgraph and the sampler is outside, or vice versa).
            # Fall back to evaluating every connected loader so the sampler
            # can at least fall back to an available family instead of
            # crashing with "no model connected".
            print(
                "[CRT MiniMaxH3] Models Pipe: no exact Unified Sampler match; "
                f"evaluating all connected loaders for fallback."
            )
            needed = {k for k, v in available.items() if v is None}
            # If nothing is connected yet, still request all 4 to surface a
            # clear "connect a model" error from build_pipe.
            if not needed:
                needed = set(available.keys())
        return sorted(key for key in needed if available[key] is None)

    @classmethod
    def IS_CHANGED(cls, minimax_h3_us_prompt=None, minimax_h3_us_unique=None, **kwargs):
        # Bust the cache when the sampler's required variant changes OR when
        # the evaluation state of any of the four model inputs changes (a
        # newly-connected loader must never keep serving a stale pipe that
        # baked None into that socket).
        try:
            needed = cls._required_model_keys(minimax_h3_us_prompt, minimax_h3_us_unique)
            presence = tuple(
                kwargs.get(k) is not None
                for k in ("fl2va_model", "fl2va_turbo_model", "ref2va_model", "ref2va_turbo_model")
            )
            return (
                hash(frozenset(needed))
                ^ hash(presence)
                ^ hash(str(minimax_h3_us_unique))
                ^ hash(cls._PIPE_CACHE_VERSION)
            )
        except Exception:
            return float("nan")

    def build_pipe(
        self,
        vae,
        audio_vae,
        clip,
        fl2va_model=None,
        fl2va_turbo_model=None,
        ref2va_model=None,
        ref2va_turbo_model=None,
        **_hidden,
    ):
        # Fail loudly instead of emitting a None-filled pipe: a returned output
        # here would be cached and served forever without re-running the lazy
        # resolution, permanently starving downstream samplers.
        available = {
            "fl2va_model": fl2va_model,
            "fl2va_turbo_model": fl2va_turbo_model,
            "ref2va_model": ref2va_model,
            "ref2va_turbo_model": ref2va_turbo_model,
        }
        needed = self._required_model_keys(
            _hidden.get("minimax_h3_us_prompt"), _hidden.get("minimax_h3_us_unique")
        )
        if not needed:
            # Never return a None-filled pipe from here: it would be cached and
            # served forever without re-running lazy resolution. Fail loudly so
            # this is visible instead of starving downstream samplers silently.
            raise ValueError(
                "MiniMax H3 US Models Pipe: could not find a Unified Sampler for "
                f"this run (pipe uid={_hidden.get('minimax_h3_us_unique')!r}). "
                "Check that this pipe output feeds a MiniMax H3 Unified Sampler (CRT)."
            )
        missing = sorted(key for key in needed if available[key] is None)
        if missing:
            raise ValueError(
                "MiniMax H3 US Models Pipe: "
                + ", ".join(f"'{k}'" for k in missing)
                + " is required by the connected Unified Sampler but nothing is "
                "connected to that socket."
            )
        pipe = {
            "vae": vae,
            "audio_vae": audio_vae,
            "clip": clip,
            "fl2va_model": fl2va_model,
            "fl2va_turbo_model": fl2va_turbo_model,
            "ref2va_model": ref2va_model,
            "ref2va_turbo_model": ref2va_turbo_model,
        }
        return (pipe,)


class CRT_MiniMaxH3USModelsPipeCNET(CRT_MiniMaxH3USModelsPipe):
    """MiniMax H3 US Models Pipe with an integrated Fun-ControlNet.

    Same lazy model loading as the base pipe, plus control_net / control_video
    inputs, an apply_to dropdown (which variants get the control tower),
    strength / start / end, and its own megapixels target.

    The control video is NOT encoded here: its aspect, quantized frame count and
    megapixels target ride along in the pipe dict, and the Unified Sampler
    resizes + encodes the control once it knows the final canvas.
    """

    SAMPLER_CLASS = "CRT_MiniMaxH3UnifiedSampler"
    _PIPE_TYPE_KEYS = ("CRT_MiniMaxH3USModelsPipe", "CRT_MiniMaxH3USModelsPipeCNET")
    _PIPE_CACHE_VERSION = "v4-cnet-native"

    APPLY_TO_OPTIONS = ("All", "FL2VA", "FL2VA Turbo", "REF2VA", "REF2VA Turbo")
    APPLY_TO_KEYS = {
        "All": {"fl2va_model", "fl2va_turbo_model", "ref2va_model", "ref2va_turbo_model"},
        "FL2VA": {"fl2va_model"},
        "FL2VA Turbo": {"fl2va_turbo_model"},
        "REF2VA": {"ref2va_model"},
        "REF2VA Turbo": {"ref2va_turbo_model"},
    }

    RETURN_TYPES = ("MINIMAXH3_US_MODELS_PIPE",)
    RETURN_NAMES = ("models_pipe",)
    FUNCTION = "build_pipe"
    CATEGORY = "CRT/MiniMaxH3"
    DESCRIPTION = ("Alt Models Pipe with an integrated MiniMax-H3 Fun-ControlNet. The control "
                   "video's aspect and quantized 17n+5 frame count override the sampler's "
                   "aspect/duration, and its megapixels target overrides the sampler's unless "
                   "US Config's MegaPixels override is wired. An optional mask (1 = regenerate) "
                   "plus source_video selects the native inpainting path, with or without a "
                   "control video.")

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "apply_to": (cls.APPLY_TO_OPTIONS, {
                "default": "All",
                "tooltip": "Which model variant(s) the control tower is applied to. 'All' covers "
                           "whatever the run actually picks. If the active variant is not covered, "
                           "the run continues WITHOUT control (warned).",
            }),
            "strength": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.05,
                                   "tooltip": "Scales every control skip. 0 is a true bypass."}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Releasing control early (~0.5) lets late steps "
                                                 "put in detail the control pass cannot describe."}),
            "megapixels": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 2.0, "step": 0.02,
                                     "tooltip": "Canvas area target; the control video's aspect "
                                                "defines the ratio (rounded to the nearest 32). "
                                                "Sampler default is 0.5."}),
        })
        inputs["optional"].update({
            "control_net": ("H3_FUN_CONTROL", {
                "tooltip": "Fun-ControlNet from the AutoDL loader. Disconnect to run as a plain "
                           "models pipe.",
            }),
            "control_video": ("IMAGE", {
                "tooltip": "Control video as an IMAGE batch (depth/canny/pose/HED/MLSD). Its "
                           "aspect overrides the sampler aspect; frames are quantized down to "
                           "the 17n+5 grid. Optional when only inpainting.",
            }),
            "mask": ("MASK", {
                "tooltip": "Inpaint mask (1 = regenerate). Uses the control video as structure "
                           "and/or source_video as the content behind the visible regions.",
            }),
            "source_video": ("IMAGE", {
                "tooltip": "Video behind the mask; only read when a mask is given.",
            }),
        })
        return inputs

    @classmethod
    def IS_CHANGED(cls, minimax_h3_us_prompt=None, minimax_h3_us_unique=None, **kwargs):
        try:
            base = super().IS_CHANGED(
                minimax_h3_us_prompt=minimax_h3_us_prompt,
                minimax_h3_us_unique=minimax_h3_us_unique, **kwargs)
            extra = stable_fingerprint(
                kwargs.get("control_video"), kwargs.get("control_net"),
                kwargs.get("mask"), kwargs.get("source_video"),
                kwargs.get("apply_to"), kwargs.get("strength"),
                kwargs.get("start_percent"), kwargs.get("end_percent"),
                kwargs.get("megapixels"))
            return f"{base}|{extra}"
        except Exception:
            return float("nan")

    def build_pipe(self, vae, audio_vae, clip, apply_to="All", strength=1.0,
                   start_percent=0.0, end_percent=1.0, megapixels=0.5,
                   fl2va_model=None, fl2va_turbo_model=None, ref2va_model=None,
                   ref2va_turbo_model=None, control_net=None, control_video=None,
                   mask=None, source_video=None, **_hidden):
        pipe = super().build_pipe(
            vae, audio_vae, clip,
            fl2va_model=fl2va_model, fl2va_turbo_model=fl2va_turbo_model,
            ref2va_model=ref2va_model, ref2va_turbo_model=ref2va_turbo_model,
            **_hidden)[0]

        mp = float(megapixels)
        has_control = control_video is not None and hasattr(control_video, "shape")
        has_mask = mask is not None and hasattr(mask, "shape")
        if control_net is None or (not has_control and not has_mask):
            CRT_MiniMaxH3UnifiedSampler._log(
                "ControlNet pipe: control_net plus control_video/mask missing; built a PLAIN "
                "models pipe (controlnet not applied).", level="warn")
            pipe["cnet"] = None
            return (pipe,)

        # The canvas aspect and the generation length follow the control video;
        # an inpaint-only run (no control video) follows the source video when
        # one is given, else the sampler's own aspect/length settings.
        ref = control_video if has_control else source_video
        if ref is not None and hasattr(ref, "shape"):
            n = int(ref.shape[0])
            nq = max(5, ((max(5, n) - 5) // 17) * 17 + 5)
            if n < 5:
                CRT_MiniMaxH3UnifiedSampler._log(
                    f"ControlNet pipe: reference has only {n} frame(s); the model needs at "
                    f"least 5. Clamped to 5.", level="warn")
            h, w = int(ref.shape[1]), int(ref.shape[2])
            aspect = (w, h)
        else:
            n, nq, aspect = 0, 0, None

        CRT_MiniMaxH3UnifiedSampler._log(
            f"ControlNet pipe: active (apply_to={apply_to}, control {has_control}, "
            f"mask {has_mask}, frames {n}->{nq}, aspect {aspect}, megapixels target {mp})",
            level="ok")

        pipe["cnet"] = {
            "control_net": control_net,
            "control_video": control_video if has_control else None,
            "mask": mask if has_mask else None,
            "source_video": source_video if (has_mask and source_video is not None
                                             and hasattr(source_video, "shape")) else None,
            "apply_to": str(apply_to),
            "strength": float(strength),
            "start_percent": float(start_percent),
            "end_percent": float(end_percent),
            "megapixels": mp,
            "aspect": aspect,
            "frames": nq,
        }
        return (pipe,)


class CRT_MiniMaxH3USConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Official prompt structure: 'integrated_multimodal_description:' (shots/motion), 'overall_soundscape:' (ambient/dialogue/SFX) and, optionally, 'non_diegetic_music:'. In R2V address references as <Picture i> / <Video k> / <Audio j>.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
            },
            "optional": {
                "First Frame (I2V)": (
                    "IMAGE",
                    {"tooltip": "FL2VA starting keyframe. The first frame is a geometry anchor stretched to the canvas."},
                ),
                "Last Frame (I2V)": (
                    "IMAGE",
                    {"tooltip": "FL2VA ending keyframe. Aspect-preserving cover-crop; motion is generated between both frames."},
                ),
                **{
                    f"Ref Image {i} (REF2VA)": (
                        "IMAGE",
                        {"tooltip": f"REF2VA reference image {i}, addressed as <Picture {i}> in the prompt."},
                    )
                    for i in range(1, MAX_REF_IMAGES + 1)
                },
                **{
                    f"Ref Video {i} (REF2VA)": (
                        "IMAGE",
                        {"tooltip": f"REF2VA reference video {i} as an IMAGE batch at 24 fps (2-15s at 24 fps, 48+ frames recommended), addressed as <Video {i}> in the prompt."},
                    )
                    for i in range(1, MAX_REF_VIDEOS + 1)
                },
                **{
                    f"Ref Video Audio {i} (REF2VA)": (
                        "AUDIO",
                        {"tooltip": f"Soundtrack paired with Ref Video {i}; addressed as its own <Audio> tag before <Video {i}>."},
                    )
                    for i in range(1, MAX_REF_VIDEOS + 1)
                },
                **{
                    f"Ref Audio {i} (REF2VA)": (
                        "AUDIO",
                        {"tooltip": f"Standalone REF2VA reference audio {i}, addressed as <Audio j> in the prompt."},
                    )
                    for i in range(1, MAX_REF_AUDIOS + 1)
                },
                "Frames (override)": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1, "forceInput": True, "tooltip": "Values above 0 override the sampler frame count; snapped up to the 17n+5 grid. 0 keeps the sampler setting."},
                ),
                "MegaPixels (override)": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.02, "forceInput": True, "tooltip": "Values above 0 override the sampler megapixels_target; 0 keeps the sampler setting."},
                ),
            },
        }

    RETURN_TYPES = ("MINIMAXH3_US_CONFIG_PIPE",)
    RETURN_NAMES = ("config_pipe",)
    FUNCTION = "build_pipe"
    CATEGORY = "CRT/MiniMaxH3"
    DESCRIPTION = "Collects the prompt, seed, optional keyframes and REF2VA reference media, plus per-workflow overrides for the unified sampler."

    def build_pipe(self, prompt, seed, **kwargs):
        first_frame = kwargs.get("First Frame (I2V)", None)
        last_frame = kwargs.get("Last Frame (I2V)", None)
        override_frames = kwargs.get("Frames (override)", None)
        override_megapixels = kwargs.get("MegaPixels (override)", None)

        ref_images = {}
        for i in range(1, MAX_REF_IMAGES + 1):
            img = kwargs.get(f"Ref Image {i} (REF2VA)", None)
            if img is not None:
                ref_images[f"ref_image_{i - 1}"] = img
        ref_videos = {}
        ref_video_audios = {}
        for i in range(1, MAX_REF_VIDEOS + 1):
            vid = kwargs.get(f"Ref Video {i} (REF2VA)", None)
            if vid is not None:
                ref_videos[f"ref_video_{i - 1}"] = vid
                aud = kwargs.get(f"Ref Video Audio {i} (REF2VA)", None)
                if aud is not None:
                    ref_video_audios[f"ref_video_audio_{i - 1}"] = aud
        ref_audios = {}
        for i in range(1, MAX_REF_AUDIOS + 1):
            aud = kwargs.get(f"Ref Audio {i} (REF2VA)", None)
            if aud is not None:
                ref_audios[f"ref_audio_{i - 1}"] = aud

        def _optional_int(v):
            if v is None:
                return None
            try:
                iv = int(v)
            except Exception:
                return None
            return iv if iv > 0 else None

        def _optional_float(v):
            if v is None:
                return None
            try:
                fv = float(v)
            except Exception:
                return None
            return fv if fv > 0 else None

        pipe = {
            "prompt": str(prompt),
            "seed": int(seed),
            "first_frame": first_frame if first_frame is None else first_frame[:1],
            "last_frame": last_frame if last_frame is None else last_frame[:1],
            "ref_images": ref_images,
            "ref_videos": ref_videos,
            "ref_video_audios": ref_video_audios,
            "ref_audios": ref_audios,
            "override_frames": _optional_int(override_frames),
            "override_megapixels": _optional_float(override_megapixels),
        }
        return (pipe,)


def _apply_sigma_shift(model, shift_video, shift_audio):
    """Mirror of the native MiniMaxH3SigmaShift operation."""
    m = model.clone()

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    original = m.get_model_object("model_sampling")
    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=shift_video, audio_shift=shift_audio)
    if hasattr(original, "noise_scale"):
        model_sampling.set_noise_scale(original.noise_scale)
    m.add_object_patch("model_sampling", model_sampling)

    to = m.model_options["transformer_options"] = m.model_options.get("transformer_options", {}).copy()
    to["minimax_h3_sigma_shift_video"] = shift_video
    to["minimax_h3_sigma_shift_audio"] = shift_audio
    return m


# --- Per-token prompt weighting (Krea2PromptWeight port) --------------------
# H3's Qwen3-VL presentation is NOT chat-templated (raw prompt tokens), and the
# tokenizer runs with disable_weights=True, so ComfyUI's native (word:weight)
# does nothing. Same trick as KJNodes Krea2PromptWeight: scale the weighted
# tokens' attention VALUE (de-emphasis / removal at weight<1) and bias their
# attention LOGIT (emphasis at weight>1), patched into every DiT block's attn.

_PROMPT_WEIGHT_PATTERN = None


def _h3_prompt_weight_pattern():
    global _PROMPT_WEIGHT_PATTERN
    if _PROMPT_WEIGHT_PATTERN is None:
        import re
        _PROMPT_WEIGHT_PATTERN = re.compile(r"\(([^():]+):(-?\d*\.?\d+)\)")
    return _PROMPT_WEIGHT_PATTERN


def _h3_token_ids(clip, text):
    tok = clip.tokenize(text)
    key = next(iter(tok))
    return [t[0] for t in tok[key][0]]


def _h3_find_subsequence(seq, sub):
    n = len(sub)
    out = []
    if n == 0:
        return out
    for i in range(len(seq) - n + 1):
        if seq[i:i + n] == sub:
            out.append(i)
    return out


def _h3_parse_prompt_weights(clip, prompt, log):
    """Return (weight_pairs, cleaned_prompt). pairs = (pos, v_factor, k_bias)."""
    pattern = _h3_prompt_weight_pattern()
    terms = [(m.group(1).strip(), float(m.group(2))) for m in pattern.finditer(prompt)]
    if not terms:
        return [], prompt
    clean = pattern.sub(lambda m: m.group(1), prompt)
    ids = _h3_token_ids(clip, clean)
    pairs = []
    for phrase, w in terms:
        if w > 1.0:
            v_factor, k_bias = 1.0, (w - 1.0) * 2.0   # emphasis via attention boost
        else:
            v_factor, k_bias = 1.0 + (w - 1.0), 0.0   # de-emphasis / removal via value scaling
        positions = []
        for variant in (" " + phrase, phrase):  # words usually carry a leading-space token
            sub = _h3_token_ids(clip, variant)
            matches = _h3_find_subsequence(ids, sub)
            if matches:
                for mi in matches:
                    positions.extend(mi + off for off in range(len(sub)))
                break
        if not positions:
            log(f"Prompt weight: phrase '{phrase}' not found in prompt; skipped.", level="warn")
            continue
        for cp in positions:
            pairs.append((cp, v_factor, k_bias))
    return pairs, clean


class _H3WeightPatch:
    """Descriptor binding the weighting attention forward onto H3's Attention."""

    def __get__(self, obj, objtype=None):
        import types
        return types.MethodType(_h3_attn_forward_weight, obj)


def _h3_attn_forward_weight(self, x, rope_freqs=None, transformer_options={}):
    import comfy.model_management
    from comfy.ldm.modules.attention import (
        AttentionTensorContainer,
        attention_pytorch,
        optimized_attention,
    )

    s = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        # fused per-head RMSNorm + partial split-half rope, in place on the qkv buffer
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
    v = v.clone()
    weights = transformer_options.get("minimax_h3_token_weights")
    if weights:
        for pos, v_factor, _ in weights:
            if v_factor != 1.0 and pos < s:
                v[pos] = v[pos] * v_factor
    bias = None
    if weights and any(kb != 0.0 for _, _, kb in weights):
        bias = q.new_zeros(1, s)
        for pos, _, kb in weights:
            if kb != 0.0 and pos < s:
                bias[:, pos] = kb
    q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    if bias is not None:
        # per-key logit bias needs the raw sdpa path; the optimized dispatcher
        # only forwards its own mask conventions
        out = attention_pytorch(q, k, v, self.heads, mask=bias, skip_reshape=True)
    else:
        out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
    return self.out_proj(out.squeeze(0))


class CRT_MiniMaxH3UnifiedSampler:
    COLOR_INFO = "\033[38;5;117m"
    COLOR_WARN = "\033[38;5;208m"
    COLOR_OK = "\033[38;5;120m"
    COLOR_RESET = "\033[0m"

    # Speed-optimization patch defaults follow the tuned reference chain.
    SOL_DEFAULTS = dict(
        tau_start=1.15,
        tau_end=0.8,
        curve="smoothstep",
        min_tokens=4096,
        strict=False,
        dense_percent=0.0,
        thresh_type="diag",
        int8_qk=False,
        int8_pv=False,
        sink_conditioning="exact_kv",
        dense_blocks="",
    )
    CHUNK_FF_DEFAULTS = dict(chunks=2, min_tokens=8192)
    SPECTRUM_DEFAULTS = dict(
        blend_weight=0.5,
        degree=1,
        ridge_lambda=0.1,
        window_size=2,
        flex_window=0.75,
        warmup_steps=1,
        tail_actual_steps=1,
        max_history=8,
        debug=False,
        history_storage="system_ram",
        bootstrap_first_forecast=True,
        anchor_residual_feedback=False,
        selective_rollback_correction=False,
        offline_smoothing_replay=True,
        audio_blend_weight=0.0,
        offline_archive_storage="system_ram",
        model_aware_mode="off",
        model_aware_risk_threshold=0.65,
        model_aware_trust_shrinkage=False,
        model_aware_replay_generic_correction=False,
        generic_correction_mode="coordinate_rls",
        generic_correction_limiter="hard_clip",
        generic_correction_limit=0.4,
        generic_correction_attenuation="no_attenuation",
    )

    # Single-entry prompt-only conditioning cache (the expensive Qwen encode).
    # Guarded by weakrefs so a freed/reallocated model can never satisfy a hit,
    # and only used when NO media is attached - media paths always rebuild.
    _TEXT_COND_CACHE = {
        "key": None,
        "clip": None,
        "vae": None,
        "audio_vae": None,
        "value": None,
    }

    # One patched CLIP per source CLIP: patching clones the CLIP, so reusing the
    # clone across runs keeps the text-conditioning cache's weakref checks valid.
    _NEGPIP_CLIP_CACHE = {"clip": None, "patched": None}

    @classmethod
    def _log(cls, message, level="info"):
        color = cls.COLOR_INFO
        if level == "warn":
            color = cls.COLOR_WARN
        elif level == "ok":
            color = cls.COLOR_OK
        print(f"{color}[CRT MiniMaxH3]{cls.COLOR_RESET} {message}")

    @classmethod
    def _progress(cls, step, total, label):
        width = 18
        filled = max(0, min(width, int(round((step / float(max(1, total))) * width))))
        bar = "#" * filled + "-" * (width - filled)
        cls._log(f"[{step}/{total}] {bar} {label}")

    @staticmethod
    def _result_tuple(result):
        if result is None:
            return tuple()
        if isinstance(result, tuple):
            return result
        if isinstance(result, list):
            return tuple(result)
        if hasattr(result, "result"):
            node_result = getattr(result, "result")
            if node_result is None:
                return tuple()
            if isinstance(node_result, tuple):
                return node_result
            if isinstance(node_result, list):
                return tuple(node_result)
            return (node_result,)
        try:
            return tuple(result)
        except Exception:
            pass
        return (result,)

    @classmethod
    def VALIDATE_INPUTS(cls, derope_preset=None, **kwargs):
        # Accept any de-rope preset string so workflows saved with the old
        # preset names still validate; unknown names fall back to Custom.
        return True

    @classmethod
    def IS_CHANGED(
        cls,
        models_pipe,
        config_pipe,
        workflow_mode,
        steps,
        steps_turbo,
        turbo,
        enable_sol_attn,
        enable_chunk_ff,
        enable_spectrum,
        live_preview,
        vae_decode_tiled,
        unload_before_decode,
        low_vram,
        megapixels_target,
        aspect_ratio,
        fl_aspect_mode,
        length_frames,
        audio_frames_override,
        video_frames_override,
        generated_audio_gain_db,
        enable_derope,
        derope_preset,
        derope_inject,
        derope_audio,
        derope_mode,
        derope_profile,
        derope_abstain,
        derope_protect_tail,
        enable_derope_adapter,
        enable_negpip,
        shift_video,
        shift_audio,
        **_other_inputs,
    ):
        return stable_fingerprint(
            models_pipe,
            config_pipe,
            workflow_mode,
            int(steps),
            int(steps_turbo),
            bool(turbo),
            bool(enable_sol_attn),
            bool(enable_chunk_ff),
            bool(enable_spectrum),
            bool(live_preview),
            bool(vae_decode_tiled),
            bool(unload_before_decode),
            bool(low_vram),
            float(megapixels_target),
            str(aspect_ratio),
            str(fl_aspect_mode),
            int(length_frames),
            bool(audio_frames_override),
            bool(video_frames_override),
            float(generated_audio_gain_db),
            bool(enable_derope),
            str(derope_preset),
            float(derope_inject),
            str(derope_audio),
            str(derope_mode),
            str(derope_profile),
            float(derope_abstain),
            int(derope_protect_tail),
            bool(enable_derope_adapter),
            bool(enable_negpip),
            float(shift_video),
            float(shift_audio),
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models_pipe": ("MINIMAXH3_US_MODELS_PIPE",),
                "config_pipe": ("MINIMAXH3_US_CONFIG_PIPE",),
                "workflow_mode": (
                    WORKFLOW_MODES,
                    {"default": MODE_FL2VA, "tooltip": "T2V: pure text (FL2VA checkpoint). FL2VA (I2V): first/last-frame keyframes. REF2VA (R2V): reference images/videos/audio. Selects which model variant the Models Pipe loads."},
                ),
                "steps": (
                    "INT",
                    {"default": STEPS_FULL_DEFAULT, "min": 1, "max": 60, "step": 1, "tooltip": "Steps used when Turbo is OFF."},
                ),
                "steps_turbo": (
                    "INT",
                    {"default": STEPS_TURBO_DEFAULT, "min": 1, "max": 60, "step": 1, "tooltip": "Steps used when Turbo is ON (the official Turbo LoRAs are trained for 4)."},
                ),
                "turbo": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Use the Turbo-LoRA variant of the active mode's family from the Models Pipe, together with the Steps Turbo count."},
                ),
                "enable_sol_attn": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Scheduled Sol attention: sparsifies self-attention on high-noise steps (tau ramp), denser near the end. Applied after the sigma shift so its schedule matches."},
                ),
                "enable_chunk_ff": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Chunked feed-forward: splits each MLP pass to reduce peak VRAM. Composable with Sol attention and Spectrum."},
                ),
                "enable_spectrum": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Spectrum forecast: predicts solver steps from past denoised anchors to cut sampling NFEs. Applied last so it wraps the final patched model."},
                ),
                "live_preview": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Decode intermediate video previews during sampling via the auto-downloaded taeh3 approximation (RGB-factor fallback when offline). Adds per-step decode overhead - disabled by default."},
                ),
                "vae_decode_tiled": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Decode video latents in tiles to reduce peak VRAM at the cost of additional processing time."},
                ),
                "unload_before_decode": (
                    "BOOLEAN",
                    {"default": True, "advanced": True, "tooltip": "Unload the diffusion model after sampling and before VAE decode to reduce decode-time VRAM."},
                ),
                "low_vram": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Unload CLIP after conditioning and the VAEs before sampling, then reload the VAEs for decode."},
                ),
                "megapixels_target": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.05,
                        "max": 2.0,
                        "step": 0.02,
                        "tooltip": "Target canvas area in megapixels, ceiled to the model's 32px grid (0.98 at 16:9 is the official 1344x768 768p canvas).",
                    },
                ),
                "aspect_ratio": (
                    ASPECT_RATIOS,
                    {"default": "16:9 (Landscape)", "tooltip": "Canvas aspect. Ignored in I2V when a First/Last frame is connected - the frame sets the canvas. In R2V, 'Ref Image 1' / 'Ref Video 1' take the aspect from the connected reference (scaled to megapixels, rounded to the nearest 32)."},
                ),
                "fl_aspect_mode": (
                    FL_ASPECT_MODES,
                    {
                        "default": "Preserve First",
                        "tooltip": "I2V with BOTH frames connected: how to reconcile different aspect ratios. Preserve First/Last: that frame defines the canvas and the other is cover-cropped to it. Optimal: a middle-ground canvas (geometric mean of both ratios) at the megapixel target; both frames are cover-cropped to it. ControlNet: with a CNET models pipe, keep the I2V keyframe canvas (Optimal) and let the native patch resize the control video to it instead of forcing the control video's aspect.",
                    },
                ),
                "length_frames": (
                    "INT",
                    {
                        "default": 124,
                        "min": 5,
                        "max": 362,
                        "step": 1,
                        "tooltip": "Clip length in frames at 24 fps, snapped to the model's 17n+5 grid (124 frames = ~5s; trained range is 124-362).",
                    },
                ),
                "audio_frames_override": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "REF2VA only. OFF (default): Duration (frames) is a hard cap - the output length equals it and longer references are trimmed, never stretched. ON: the output length derives from the longest Ref Audio instead (no cap - long inputs risk OOM). Output is always snapped to the 17n+5 grid at 24 fps."},
                ),
                "video_frames_override": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "REF2VA only. ON (default): the output length derives from the longest Ref Video, snapped to the 17n+5 grid (no cap - long inputs risk OOM). OFF: Duration (frames) is a hard cap and longer references are trimmed. A CNET models pipe always wins with the control_video quantized frame count."},
                ),
                "generated_audio_gain_db": (
                    "FLOAT",
                    {"default": 0.0, "min": -60.0, "max": 24.0, "step": 0.1, "tooltip": "Gain applied to the generated audio after decode, in decibels."},
                ),
                "enable_derope": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "De-ROPE (motion smearing fix): after the main pass, detect jerk-fast motion from the latent, time-smear those spans onto a slower dilated clock, re-render them with a partial-denoise second pass, then recover the original frame count. Costs ~2-3x the pass-1 compute on hot clips."},
                ),
                "derope_preset": (
                    list(DEROPE_PRESETS),
                    {"default": DEROPE_PRESET_DEFAULT,
                     "tooltip": "One scenario dropdown for the whole de-rope. Balanced = measured default; Fast action = sharper, keeps more of the source; Camera motion = camera-compensated oracle (stops pans scoring as jerk); End burst = holds the tail at real time; Uniform x4 = slow the whole clip 4x (zero-artifact reference, most expensive). Pick Custom to reveal the individual knobs."},
                ),
                "derope_inject": (
                    "FLOAT",
                    {"default": DEROPE_INJECT_DEFAULT, "min": 0.05, "max": 1.0, "step": 0.05, "advanced": True,
                     "tooltip": "Custom preset only. How much of the denoise trajectory runs on top of the smeared init. 0.50 is the metric best (sharpest, closest choreography), 0.70 the safe playback default, 0.80 looser / more creative."},
                ),
                "derope_audio": (
                    list(DEROPE_AUDIO_MODES),
                    {"default": "keep pass-1 audio (safe default)", "advanced": True,
                     "tooltip": "Custom preset only. Safe default: pass-2's audio rows are seeded with the slowed performance (keeps mouths/foley on the dilated clock) but the OUTPUT keeps pass-1's track. 'pass-2 foley (seeded)' returns the regenerated performance instead (leaner, quality varies). 'pass-2 invents audio' does not seed the rows."},
                ),
                "derope_mode": (
                    list(DEROPE_MODES),
                    {"default": "adaptive (jerk oracle)", "advanced": True,
                     "tooltip": "Custom preset only. adaptive = dilate only the jerk-hot spans (cheaper, preserves the clip's pacing). uniform x4 = hold every frame 4x (zero-artifact reference, highest cost)."},
                ),
                "derope_profile": (
                    list(DEROPE_PROFILE_MODES),
                    {"default": DEROPE_PROFILE_DEFAULT, "advanced": True,
                     "tooltip": "Custom preset only. Which signal the adaptive oracle thresholds. 'value |d3|' is the shipped detector but tracks motion ENERGY, so a steady pan scores as jerk. 'trajectory centroid |d3|' differentiates the energy centroid's path (closer to true jerk); 'camera-compensated' aligns each frame to the last so pans read as stillness."},
                ),
                "derope_abstain": (
                    "FLOAT",
                    {"default": DEROPE_ABSTAIN_DEFAULT, "min": 0.0, "max": 10.0, "step": 0.1, "advanced": True,
                     "tooltip": "Custom preset only. Absolute gate on the jerk profile's peak-to-mean contrast. q is a quantile, so the oracle always dilates the top (1-q) of token-times even on a clip with nothing to fix; below this contrast the de-rope abstains and returns the pass-1 output. 0 disables the gate."},
                ),
                "derope_protect_tail": (
                    "INT",
                    {"default": 0, "min": 0, "max": 96, "advanced": True,
                     "tooltip": "Custom preset only. Hold the LAST n frames at real time whatever the oracle says. A burst that runs into the end of the clip has no 'after' for the model to slow into, so it can play back fast after recovery; 17 (one token group) is the measured fix. 0 = off."},
                ),
                "enable_derope_adapter": (
                    "BOOLEAN",
                    {"default": False, "advanced": True,
                     "tooltip": "Apply the MATLOWAI MiniMax-H3 Motion Adapter LoRA (temporal-expansion warm100, strength 1.0) to the de-rope pass only (never pass 1). Downloads ~63 MB from Hugging Face on first use into models/loras/minimax_h3/. Trained on the de-rope task: smooths fast motion, but mutes strong colour and over-corrects calm content."},
                ),
                "enable_negpip": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "NegPiP: (word:-1.0) in the prompt SUBTRACTS the concept by flipping that token's attention value vector, and (word:-1.0@2.5-4.0) limits the subtraction to that window (@v = video only, @a = audio only). Plain (word:1.3) emphasis is scaled as well. Runs at exact-flip strength (1.0); turn OFF to fall back to the weaker in-place weighting."},
                ),
                "shift_video": (
                    "FLOAT",
                    {"default": SHIFT_VIDEO, "min": 0.01, "max": 100.0, "step": 0.01,
                     "tooltip": "Video sigma shift. Turbo LoRAs distill a fixed schedule: 12.0 for the 544p/v0.1 family, 6.0 for the 768p v1.0 family - match this to the connected Turbo LoRA."},
                ),
                "shift_audio": (
                    "FLOAT",
                    {"default": SHIFT_AUDIO, "min": 0.01, "max": 100.0, "step": 0.01,
                     "tooltip": "Audio sigma shift. Turbo LoRAs distill a fixed schedule: 3.0 for all current families."},
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "sample"
    CATEGORY = "CRT/MiniMaxH3"
    DESCRIPTION = "Unified MiniMax H3 sampler: orchestrates conditioning, sigma shift, speed patches, scheduling, and AV decode for T2V / FL2VA / REF2VA."

    @staticmethod
    def _require_pipe_dict(pipe, pipe_name):
        if not isinstance(pipe, dict):
            raise ValueError(f"{pipe_name} is not a valid pipe object.")
        return pipe

    @classmethod
    def _unpack_models_pipe(cls, models_pipe, workflow_mode):
        pipe = cls._require_pipe_dict(models_pipe, "models_pipe")
        for key in ("vae", "audio_vae", "clip"):
            if pipe.get(key, None) is None:
                raise ValueError(f"models_pipe is missing {key}.")

        family = _active_family(workflow_mode)
        base = pipe.get(f"{family}_model", None)
        turbo_model = pipe.get(f"{family}_turbo_model", None)
        return {
            "vae": pipe["vae"],
            "audio_vae": pipe["audio_vae"],
            "clip": pipe["clip"],
            "family": family,
            "base_model": base,
            "turbo_model": turbo_model,
        }

    @staticmethod
    def _unpack_config_pipe(config_pipe):
        pipe = CRT_MiniMaxH3UnifiedSampler._require_pipe_dict(config_pipe, "config_pipe")
        return {
            "prompt": str(pipe.get("prompt", "")),
            "seed": int(pipe.get("seed", 0)),
            "first_frame": pipe.get("first_frame", None),
            "last_frame": pipe.get("last_frame", None),
            "ref_images": pipe.get("ref_images", {}) or {},
            "ref_videos": pipe.get("ref_videos", {}) or {},
            "ref_video_audios": pipe.get("ref_video_audios", {}) or {},
            "ref_audios": pipe.get("ref_audios", {}) or {},
            "override_frames": pipe.get("override_frames", None),
            "override_megapixels": pipe.get("override_megapixels", None),
        }

    @staticmethod
    def _offload_clip(clip):
        try:
            if clip is None:
                return
            target = mm.unet_offload_device()
            for obj in (
                getattr(clip, "cond_stage_model", None),
                getattr(getattr(clip, "patcher", None), "model", None),
            ):
                if obj is not None and hasattr(obj, "to"):
                    try:
                        obj.to(target)
                    except Exception:
                        pass
            gc.collect()
            mm.soft_empty_cache()
        except Exception as e:
            CRT_MiniMaxH3UnifiedSampler._log(f"CLIP offload failed: {e}", level="warn")

    @staticmethod
    def _offload_vae(vae):
        try:
            if vae is None:
                return
            target = mm.unet_offload_device()
            obj = getattr(vae, "first_stage_model", None) or getattr(vae, "model", None)
            if obj is not None and hasattr(obj, "to"):
                obj.to(target)
            gc.collect()
            mm.soft_empty_cache()
        except Exception as e:
            CRT_MiniMaxH3UnifiedSampler._log(f"VAE offload failed: {e}", level="warn")

    @staticmethod
    def _reload_vae(vae):
        try:
            if vae is None:
                return
            device = mm.get_torch_device()
            obj = getattr(vae, "first_stage_model", None) or getattr(vae, "model", None)
            if obj is not None and hasattr(obj, "to"):
                obj.to(device)
        except Exception as e:
            CRT_MiniMaxH3UnifiedSampler._log(f"VAE reload failed: {e}", level="warn")

    @staticmethod
    def _unload_sampling_model(model):
        errors = []
        try:
            inner = getattr(model, "model", None)
            if inner is not None and hasattr(inner, "to"):
                inner.to(mm.unet_offload_device())
        except Exception as e:
            errors.append(f"model.to(offload) failed: {e}")
        try:
            mm.unload_all_models()
        except Exception as e:
            errors.append(f"unload_all_models failed: {e}")
        try:
            mm.cleanup_models_gc()
        except Exception as e:
            errors.append(f"cleanup_models_gc failed: {e}")
        try:
            gc.collect()
            mm.soft_empty_cache()
        except Exception as e:
            errors.append(f"soft_empty_cache failed: {e}")
        return errors

    @classmethod
    def _get_negpip_clip(cls, clip):
        """NegPiP-patched CLIP for this run; memoized per source CLIP so the
        text-conditioning cache's weakref checks keep working across runs."""
        cached = cls._NEGPIP_CLIP_CACHE
        ref = cached["clip"]
        if ref is not None and ref() is clip and cached["patched"] is not None:
            return cached["patched"]
        patched = _NegPiP.patch_clip(clip)
        try:
            cached["clip"] = weakref.ref(clip)
            cached["patched"] = patched
        except TypeError:
            cached["clip"] = None
            cached["patched"] = None
        return patched

    @classmethod
    def _apply_speed_patches(cls, model, enable_sol_attn, enable_chunk_ff, enable_spectrum):
        """Apply the optimization chain in dependency order.

        Sigma shift must already be on the model: the Sol patch reads
        model_sampling.percent_to_sigma at patch time. Sol and ChunkFF install
        object patches that adopt earlier patches as fallbacks, and Spectrum's
        sampler wrappers go on last so they run the fully patched model.
        Everything is embedded (MiniMaxUSOpt) — no external custom nodes.
        """
        return MiniMaxUSOpt.apply_us_opt(
            model,
            enable_sol=bool(enable_sol_attn),
            sol_params=dict(cls.SOL_DEFAULTS),
            enable_chunk_ff=bool(enable_chunk_ff),
            chunk_params=dict(cls.CHUNK_FF_DEFAULTS),
            enable_spectrum=bool(enable_spectrum),
            spectrum_params=dict(cls.SPECTRUM_DEFAULTS),
            log_fn=cls._log,
        )

    @staticmethod
    def _dims_from_megapixels_aspect(megapixels, aspect_ratio):
        """Official ResolutionSelector recipe: aspect * megapixels, ceiled to 32."""
        ratio_str = str(aspect_ratio).split(" ")[0]
        try:
            width_ratio, height_ratio = map(int, ratio_str.split(":"))
        except Exception:
            width_ratio, height_ratio = 16, 9

        ratio = float(width_ratio) / float(max(height_ratio, 1))
        total_pixels = max(1, int(float(megapixels) * 1_000_000))
        width = math.sqrt(total_pixels * ratio)
        height = math.sqrt(total_pixels / max(ratio, 1e-8))

        multiple = 32
        width = max(multiple, int(math.ceil(width / multiple)) * multiple)
        height = max(multiple, int(math.ceil(height / multiple)) * multiple)
        return int(width), int(height)

    @staticmethod
    def _dims_from_aspect_round32(megapixels, w, h):
        """Canvas from a raw aspect (w, h) and megapixel target, rounded to the
        NEAREST multiple of 32 (min 32). Used when the ControlNet pipe's control
        video defines the aspect instead of the tab widget."""
        ratio = float(w) / float(max(1, h))
        total_pixels = max(1, int(float(megapixels) * 1_000_000))
        width = math.sqrt(total_pixels * ratio)
        height = math.sqrt(total_pixels / max(ratio, 1e-8))
        width = max(32, int(round(width / 32)) * 32)
        height = max(32, int(round(height / 32)) * 32)
        return int(width), int(height)

    @staticmethod
    def _resolve_ref_aspect(aspect_ratio, config):
        """Canvas aspect (w, h) from a connected reference when the tab widget
        selects 'Ref Image 1' / 'Ref Video 1'; else None.

        The reference content is only used for inference in R2V, but its aspect
        is a valid canvas source in any mode, so the selection is honored in
        T2V and I2V too."""
        ratio_str = str(aspect_ratio)
        if ratio_str == "Ref Image 1":
            img = (config.get("ref_images") or {}).get("ref_image_0")
            if img is None:
                raise ValueError(
                    "Aspect 'Ref Image 1' needs 'Ref Image 1 (REF2VA)' connected in the US Config."
                )
            return int(img.shape[2]), int(img.shape[1])
        if ratio_str == "Ref Video 1":
            vid = (config.get("ref_videos") or {}).get("ref_video_0")
            if vid is None:
                raise ValueError(
                    "Aspect 'Ref Video 1' needs 'Ref Video 1 (REF2VA)' connected in the US Config."
                )
            return int(vid.shape[2]), int(vid.shape[1])
        return None

    @staticmethod
    def _unpack_cnet(models_pipe):
        """Extract the ControlNet payload a CNET models pipe carries, if any."""
        cnet = models_pipe.get("cnet") if isinstance(models_pipe, dict) else None
        return cnet if isinstance(cnet, dict) else None

    def _apply_cnet_control(self, model, cnet, vae, mode, use_turbo):
        """Patch the active model with the native Fun ControlNet model patch.

        The native patch fits the control video to the generation canvas and the
        17n+5 frame grid and encodes it, so it is handed the raw video plus the
        optional inpaint mask/source. apply_to decides whether the ACTIVE model
        is covered; when it isn't, the run continues WITHOUT control (warned)
        instead of erroring.
        """
        apply_to = str(cnet.get("apply_to", "All"))
        family = "ref2va" if mode == MODE_REF2VA else "fl2va"
        active_key = f"{family}_{'turbo_' if use_turbo else ''}model"
        covered = active_key in CRT_MiniMaxH3USModelsPipeCNET.APPLY_TO_KEYS.get(apply_to, set())
        if not covered:
            self._log(
                f"ControlNet apply_to='{apply_to}' does not cover the active model "
                f"({active_key}); running WITHOUT controlnet.", level="warn")
            return model

        start_p = float(cnet.get("start_percent", 0.0))
        end_p = float(cnet.get("end_percent", 1.0))
        strength = float(cnet.get("strength", 1.0))

        patched = apply_fun_control(
            model, cnet["control_net"], vae, cnet.get("control_video"),
            strength, start_p, end_p,
            mask=cnet.get("mask"), source_video=cnet.get("source_video"),
            log_fn=self._log)
        if patched is model:
            self._log(
                f"ControlNet: nothing to apply for {active_key} (strength {strength}, "
                f"control_video {'set' if cnet.get('control_video') is not None else 'unset'}, "
                f"mask {'set' if cnet.get('mask') is not None else 'unset'}).", level="warn")
            return model
        self._log(
            f"ControlNet applied to {active_key} (strength {strength}, "
            f"schedule {start_p:.0%}-{end_p:.0%})", level="ok")
        return patched

    @staticmethod
    def _resize_crop_cover(image, width, height):
        """Aspect-preserving cover resize, then center-crop to exactly
        width x height - never stretches."""
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            return image

        _, src_h, src_w, _ = image.shape
        if src_w == width and src_h == height:
            return image

        scale = max(width / float(max(1, src_w)), height / float(max(1, src_h)))
        scaled_w = max(width, int(math.ceil(src_w * scale)))
        scaled_h = max(height, int(math.ceil(src_h * scale)))

        upscaled = comfy.utils.common_upscale(
            image.movedim(-1, 1), scaled_w, scaled_h, "lanczos", "disabled"
        ).movedim(1, -1)

        x0 = max(0, (scaled_w - width) // 2)
        y0 = max(0, (scaled_h - height) // 2)
        return (
            upscaled[:, y0:y0 + height, x0:x0 + width, :].contiguous(),
        )[0]

    @staticmethod
    def _frame_canvas(image, megapixels_target):
        """Aspect-preserving resize to the closest resolution above the
        megapixel target, then center-crop the few pixels down to a 32-multiple.
        Returns (cropped_image, width, height)."""
        _, src_h, src_w, _ = image.shape
        scale = math.sqrt(
            max(1.0, float(megapixels_target) * 1_000_000)
            / float(max(1, src_w * src_h))
        )
        rw = max(32, int(math.ceil(src_w * scale)))
        rh = max(32, int(math.ceil(src_h * scale)))
        resized = comfy.utils.common_upscale(
            image.movedim(-1, 1), rw, rh, "lanczos", "disabled"
        ).movedim(1, -1)
        width = max(32, (rw // 32) * 32)
        height = max(32, (rh // 32) * 32)
        x0 = (rw - width) // 2
        y0 = (rh - height) // 2
        cropped = resized[:, y0:y0 + height, x0:x0 + width, :].contiguous()
        return cropped, width, height

    @classmethod
    def _resolve_i2v_canvas(cls, first, last, megapixels_target, fl_aspect_mode):
        """I2V canvas: the connected keyframes override the aspect widget.

        Exactly one frame -> that frame's aspect defines the canvas. Both
        frames -> fl_aspect_mode picks which one is preserved; Optimal builds a
        middle-ground canvas from both ratios. The non-preserved frame is
        cover-cropped (never stretched) onto the canvas."""
        if last is None:
            first, width, height = cls._frame_canvas(first, megapixels_target)
            cls._log(f"I2V canvas from first frame: {width}x{height}", level="ok")
            return width, height, first, None
        if first is None:
            last, width, height = cls._frame_canvas(last, megapixels_target)
            cls._log(f"I2V canvas from last frame: {width}x{height}", level="ok")
            return width, height, None, last

        _, fh, fw, _ = first.shape
        _, lh, lw, _ = last.shape

        if str(fl_aspect_mode) in ("Optimal", "ControlNet"):
            ratio = math.sqrt((fw / float(fh)) * (lw / float(lh)))
            target_area = max(1.0, float(megapixels_target) * 1_000_000)
            rw = max(32, int(math.ceil(math.sqrt(target_area * ratio))))
            rh = max(32, int(math.ceil(math.sqrt(target_area / max(ratio, 1e-8)))))
            width = max(32, (rw // 32) * 32)
            height = max(32, (rh // 32) * 32)
            first = cls._resize_crop_cover(first, width, height)
            last = cls._resize_crop_cover(last, width, height)
            cls._log(f"I2V optimal F/L canvas: {width}x{height}", level="ok")
            return width, height, first, last

        if str(fl_aspect_mode) == "Preserve Last":
            base_img, width, height = cls._frame_canvas(last, megapixels_target)
            first = cls._resize_crop_cover(first, width, height)
            cls._log(f"I2V canvas from preserved last frame: {width}x{height}", level="ok")
            return width, height, first, base_img

        base_img, width, height = cls._frame_canvas(first, megapixels_target)
        last = cls._resize_crop_cover(last, width, height)
        cls._log(f"I2V canvas from preserved first frame: {width}x{height}", level="ok")
        return width, height, base_img, last

    @staticmethod
    def _length_from_frames(frames):
        return align_frame_count(max(5, int(frames)))

    @staticmethod
    def _weak(obj):
        try:
            return weakref.ref(obj)
        except TypeError:
            return None

    @classmethod
    def _get_text_conditioning_cache(cls, cache_key, clip, vae, audio_vae):
        cached = cls._TEXT_COND_CACHE
        if cached["key"] != cache_key:
            return None
        for field, obj in (("clip", clip), ("vae", vae), ("audio_vae", audio_vae)):
            ref = cached[field]
            if ref is None or ref() is not obj:
                return None
        cls._log("Conditioning cache HIT - skipping CLIP/VAE encode", level="ok")
        return cached["value"]

    @classmethod
    def _store_text_conditioning_cache(cls, cache_key, clip, vae, audio_vae, value):
        try:
            refs = {
                "clip": cls._weak(clip),
                "vae": cls._weak(vae),
                "audio_vae": cls._weak(audio_vae),
            }
            if any(r is None for r in refs.values()):
                return
            cached = cls._TEXT_COND_CACHE
            cached["key"] = cache_key
            cached.update(refs)
            cached["value"] = value
        except Exception:
            pass

    @classmethod
    def _build_conditioning_and_latent(cls, mode, clip, vae, audio_vae, prompt, config, width, height, length):
        # Tensors must not be evaluated with `or` / `bool()` (ambiguous for
        # multi-value tensors). Check `is not None` for frames, len for dicts.
        has_media = bool(
            (config.get("first_frame") is not None)
            or (config.get("last_frame") is not None)
            or bool(config.get("ref_images"))
            or bool(config.get("ref_videos"))
            or bool(config.get("ref_video_audios"))
            or bool(config.get("ref_audios"))
        )

        cache_key = None
        if not has_media:
            cache_key = (
                "minimaxh3-cond-text-v2",
                str(prompt),
                mode,
                int(width),
                int(height),
                int(length),
            )
            cached = cls._get_text_conditioning_cache(cache_key, clip, vae, audio_vae)
            if cached is not None:
                return cached

        if mode == MODE_REF2VA:
            total_refs = (
                len(config["ref_images"])
                + len(config["ref_videos"])
                + len(config["ref_video_audios"])
                + len(config["ref_audios"])
            )
            if total_refs > 12:
                cls._log(
                    f"{total_refs} reference files connected; the official Ref2VA limit is 12 mixed files - quality may degrade.",
                    level="warn",
                )
            has_refs = bool(
                config["ref_images"] or config["ref_videos"] or config["ref_audios"]
            )
            # AIToolkit ref-video treatment (always on): snap the ref's own
            # duration DOWN to 17n+5 and trim the paired audio to the same
            # real-time window. Matches ai-toolkit's minimax_h3_ref2va recipe.
            def _snap_down(n):
                return ((max(5, int(n)) - 5) // 17) * 17 + 5

            for k in list((config.get("ref_videos") or {}).keys()):
                vid = config["ref_videos"][k]
                if vid is not None and hasattr(vid, "shape"):
                    try:
                        total = int(vid.shape[0])
                        n = _snap_down(total)
                        if n < total:
                            config["ref_videos"][k] = vid[:n]
                            audio_key = k.replace("ref_video_", "ref_video_audio_")
                            if audio_key in (config.get("ref_video_audios") or {}):
                                aud = config["ref_video_audios"][audio_key]
                                if isinstance(aud, dict) and aud.get("waveform") is not None:
                                    sr = int(aud.get("sample_rate", 44100))
                                    keep = int(round(n / 24 * sr))
                                    aud["waveform"] = aud["waveform"][..., :keep]
                    except Exception:
                        pass
            # Resize REF2VA visuals to the sampler megapixel area (width x height),
            # aspect-preserving cover resize + center-crop (never stretch).
            def _resize_ref_to_target(img_batch):
                _, h, w, _ = img_batch.shape
                ratio = float(w) / float(max(1, h))
                total_pixels = int(width) * int(height)
                tw = math.sqrt(total_pixels * ratio)
                th = math.sqrt(total_pixels / max(ratio, 1e-8))
                tw = max(32, int(round(tw / 32) * 32))
                th = max(32, int(round(th / 32) * 32))
                if tw == w and th == h:
                    return img_batch
                return cls._resize_crop_cover(img_batch, tw, th)

            ref_images_resized = {}
            for k, img in (config["ref_images"] or {}).items():
                try:
                    ref_images_resized[k] = _resize_ref_to_target(img[:1]) if img is not None else img
                except Exception:
                    ref_images_resized[k] = img
            ref_videos_resized = {}
            for k, vid in (config["ref_videos"] or {}).items():
                try:
                    ref_videos_resized[k] = _resize_ref_to_target(vid) if vid is not None else vid
                except Exception:
                    ref_videos_resized[k] = vid

            # Per-run content fingerprints: makes stale-vs-fresh reference media
            # visible in the console when diagnosing cache/repeat-run issues.
            ref_digests = []
            for k, img in ref_images_resized.items():
                try:
                    ref_digests.append(
                        f"{k}: {int(img.shape[2])}x{int(img.shape[1])} mean={float(img.float().mean()):.6f}"
                    )
                except Exception:
                    ref_digests.append(f"{k}: <unreadable>")
            for k, vid in ref_videos_resized.items():
                try:
                    ref_digests.append(
                        f"{k}: {int(vid.shape[0])}f {int(vid.shape[2])}x{int(vid.shape[1])} mean={float(vid.float().mean()):.6f}"
                    )
                except Exception:
                    ref_digests.append(f"{k}: <unreadable>")
            if ref_digests:
                cls._log("Ref content: " + " | ".join(ref_digests), level="ok")

            # Patch adapt_canvas for this call so ref videos also respect the
            # target MP area instead of the fixed 768 short edge. The native
            # ref-video path resizes with crop="disabled" (stretch) when the
            # aspect differs, so pre-cover-crop the frames to the patched
            # canvas aspect and hand the native node an exact-aspect input.
            import comfy_extras.nodes_minimax_h3 as _h3_nodes
            _orig_adapt = _h3_nodes.adapt_canvas
            def _patched_adapt(vw, vh):
                ratio = float(vw) / float(max(1, vh))
                total_pixels = int(width) * int(height)
                cw = math.sqrt(total_pixels * ratio)
                ch = math.sqrt(total_pixels / max(ratio, 1e-8))
                cw = max(32, int(round(cw / 32) * 32))
                ch = max(32, int(round(ch / 32) * 32))
                return int(cw), int(ch)
            _h3_nodes.adapt_canvas = _patched_adapt
            # Pre-cover-crop each ref video to its patched canvas so the
            # native resize never stretches (crop="disabled" on exact aspect
            # is a no-op).
            for k in list(ref_videos_resized.keys()):
                vid = ref_videos_resized[k]
                if vid is None or not hasattr(vid, "shape"):
                    continue
                try:
                    cw, ch = _patched_adapt(int(vid.shape[2]), int(vid.shape[1]))
                    if (int(vid.shape[2]), int(vid.shape[1])) != (cw, ch):
                        ref_videos_resized[k] = cls._resize_crop_cover(vid, cw, ch)
                except Exception:
                    pass
            try:
                outputs = cls._result_tuple(
                    MiniMaxH3ReferenceToVideo.execute(
                        clip=clip,
                        vae=vae,
                        audio_vae=audio_vae,
                        prompt=prompt,
                        width=int(width),
                        height=int(height),
                        length=int(length),
                        ref_image_size="match",
                        ref_images=ref_images_resized,
                        ref_videos=ref_videos_resized,
                        ref_video_audios=config["ref_video_audios"],
                        ref_audios=config["ref_audios"],
                    )
                )
            finally:
                _h3_nodes.adapt_canvas = _orig_adapt
            positive, latent = outputs[0], outputs[1]
            if has_refs:
                cls._log(
                    "REF2VA refs: "
                    f"{len(config['ref_images'])} image(s), {len(config['ref_videos'])} video(s), "
                    f"{len(config['ref_audios'])} standalone audio(s)",
                    level="ok",
                )
            else:
                cls._log("REF2VA without any reference media; running prompt-only.", level="warn")
        else:
            first_frame = config["first_frame"] if mode == MODE_FL2VA else None
            last_frame = config["last_frame"] if mode == MODE_FL2VA else None
            if mode == MODE_FL2VA and first_frame is None and last_frame is None:
                cls._log(
                    "FL2VA selected without keyframes; falling back to prompt-only T2V conditioning.",
                    level="warn",
                )
            outputs = cls._result_tuple(
                MiniMaxH3ImageToVideo.execute(
                    clip=clip,
                    vae=vae,
                    prompt=prompt,
                    width=int(width),
                    height=int(height),
                    length=int(length),
                    first_frame=first_frame,
                    last_frame=last_frame,
                )
            )
            positive, latent = outputs[0], outputs[1]

        if cache_key is not None:
            cls._store_text_conditioning_cache(cache_key, clip, vae, audio_vae, (positive, latent))
        return positive, latent

    def sample(
        self,
        models_pipe,
        config_pipe,
        workflow_mode,
        steps,
        steps_turbo,
        turbo,
        enable_sol_attn,
        enable_chunk_ff,
        enable_spectrum,
        live_preview,
        vae_decode_tiled,
        unload_before_decode,
        low_vram,
        megapixels_target,
        aspect_ratio,
        fl_aspect_mode,
        length_frames,
        audio_frames_override,
        video_frames_override,
        generated_audio_gain_db=0.0,
        enable_derope=False,
        derope_preset=DEROPE_PRESET_DEFAULT,
        derope_inject=DEROPE_INJECT_DEFAULT,
        derope_audio="keep pass-1 audio (safe default)",
        derope_mode="adaptive (jerk oracle)",
        derope_profile=DEROPE_PROFILE_DEFAULT,
        derope_abstain=DEROPE_ABSTAIN_DEFAULT,
        derope_protect_tail=0,
        enable_derope_adapter=False,
        enable_negpip=True,
        shift_video=SHIFT_VIDEO,
        shift_audio=SHIFT_AUDIO,
        **_other_inputs,
    ):
        mode = str(workflow_mode)
        if mode not in WORKFLOW_MODES:
            raise ValueError(f"Unknown workflow_mode: {mode}")
        total_steps = 6

        kickoff_taeh3_download()
        wipe_all_caches()

        self._log(f"Starting mode: {mode}")
        live_preview = bool(live_preview)
        previous_preview_method = latent_preview.args.preview_method
        try:
            # Core binary previews render as a still-per-step in the modern
            # frontend; live preview is delivered by the animated-WebP override
            # wrapper instead, so the core path stays off entirely.
            latent_preview.set_preview_method("none")

            images, audio = self._sample_inner(
                models_pipe,
                config_pipe,
                mode,
                steps,
                steps_turbo,
                turbo,
                enable_sol_attn,
                enable_chunk_ff,
                enable_spectrum,
                vae_decode_tiled,
                unload_before_decode,
                low_vram,
                megapixels_target,
                aspect_ratio,
                fl_aspect_mode,
                length_frames,
                audio_frames_override,
                video_frames_override,
                generated_audio_gain_db,
                live_preview=bool(live_preview),
                unique_id=_other_inputs.get("unique_id"),
                enable_derope=bool(enable_derope),
                derope_preset=str(derope_preset),
                derope_inject=float(derope_inject),
                derope_audio=str(derope_audio),
                derope_mode=str(derope_mode),
                derope_profile=str(derope_profile),
                derope_abstain=float(derope_abstain),
                derope_protect_tail=int(derope_protect_tail),
                enable_derope_adapter=bool(enable_derope_adapter),
                enable_negpip=bool(enable_negpip),
                shift_video=float(shift_video),
                shift_audio=float(shift_audio),
            )
        finally:
            # Restore the user's global preview method; our override must not
            # leak into other workflows after the run.
            latent_preview.args.preview_method = previous_preview_method
        return images, audio

    def _sample_inner(
        self,
        models_pipe,
        config_pipe,
        mode,
        steps,
        steps_turbo,
        turbo,
        enable_sol_attn,
        enable_chunk_ff,
        enable_spectrum,
        vae_decode_tiled,
        unload_before_decode,
        low_vram,
        megapixels_target,
        aspect_ratio,
        fl_aspect_mode,
        length_frames,
        audio_frames_override,
        video_frames_override,
        generated_audio_gain_db=0.0,
        enable_derope=False,
        derope_preset=DEROPE_PRESET_DEFAULT,
        derope_inject=DEROPE_INJECT_DEFAULT,
        derope_audio="keep pass-1 audio (safe default)",
        derope_mode="adaptive (jerk oracle)",
        derope_profile=DEROPE_PROFILE_DEFAULT,
        derope_abstain=DEROPE_ABSTAIN_DEFAULT,
        derope_protect_tail=0,
        enable_derope_adapter=False,
        enable_negpip=True,
        shift_video=SHIFT_VIDEO,
        shift_audio=SHIFT_AUDIO,
        live_preview=False,
        unique_id=None,
    ):
        total_steps = 6 if enable_derope else 4

        models = self._unpack_models_pipe(models_pipe, mode)
        config = self._unpack_config_pipe(config_pipe)
        cnet = self._unpack_cnet(models_pipe)

        use_turbo = bool(turbo) and models["turbo_model"] is not None
        if bool(turbo) and models["turbo_model"] is None:
            if models["base_model"] is not None:
                self._log(
                    "Turbo enabled but the Turbo variant of the active family is not connected; "
                    "using the base model with full steps.",
                    level="warn",
                )
        if use_turbo:
            step_count = int(steps_turbo)
        else:
            step_count = int(steps)
        base_model = None
        if models["base_model"] is None or models["turbo_model"] is None:
            # Socket visibility: shows exactly which variants the pipe actually
            # delivered. A ✓ on the active family's missing variant means the
            # pipe served a stale cache — the IS_CHANGED presence hash now
            # busts that, so this should only appear on genuine wiring gaps.
            received = {
                k: ("✓" if (models_pipe.get(k) if isinstance(models_pipe, dict) else None) is not None else "✗")
                for k in ("fl2va_model", "fl2va_turbo_model", "ref2va_model", "ref2va_turbo_model")
            }
            self._log(
                "Models Pipe sockets received: "
                + " ".join(f"{k}={v}" for k, v in received.items()),
                level="warn",
            )
        if models["base_model"] is None and models["turbo_model"] is None:
            # Last resort: try any model wired to the pipe, even from the
            # other family, so a graph with only FL2VA wired can still run
            # an R2V request instead of hard-crashing.
            for fallback_key in ("fl2va_model", "fl2va_turbo_model", "ref2va_model", "ref2va_turbo_model"):
                fb = models_pipe.get(fallback_key) if isinstance(models_pipe, dict) else None
                if fb is not None:
                    self._log(
                        f"Required {models['family']} model missing; falling back to {fallback_key} (may be suboptimal for {mode}).",
                        level="warn",
                    )
                    base_model = fb
                    if fallback_key.endswith("_turbo_model") and not use_turbo:
                        step_count = int(steps)
                    break
            if base_model is None:
                raise ValueError(
                    f"models_pipe has no '{models['family']}_model' (or '{models['family']}_turbo_model') "
                    f"connected, required by {mode}. Connect the matching AutoDL model loader output to "
                    "that socket on the MiniMax H3 US Models Pipe (CRT)."
                )
        else:
            base_model = models["turbo_model"] if use_turbo else models["base_model"]
        if base_model is None and not use_turbo and models["turbo_model"] is not None:
            self._log(
                "Turbo is OFF but the base model is not connected; falling back to the Turbo model with full steps.",
                level="warn",
            )
            base_model = models["turbo_model"]
        elif base_model is None and use_turbo and models["turbo_model"] is None and models["base_model"] is not None:
            self._log(
                "Turbo is ON but the Turbo model is not connected; falling back to the base model.",
                level="warn",
            )
            base_model = models["base_model"]
        if base_model is None:
            raise ValueError(
                f"Selected model variant for {mode} is not connected (turbo={bool(turbo)}). "
                f"Connect '{models['family']}_{'turbo_' if use_turbo else ''}model' on the Pipe or toggle Turbo."
            )
        self._log(
            f"Family: {models['family']} | variant: {'Turbo LoRA' if use_turbo else 'base'} | steps: {step_count}",
            level="ok",
        )

        self._progress(1, total_steps, "Applying sigma shift and speed patches")
        model = _apply_sigma_shift(base_model, float(shift_video), float(shift_audio))
        model = self._apply_speed_patches(model, enable_sol_attn, enable_chunk_ff, enable_spectrum)
        if live_preview:
            model = apply_h3_preview_override(model, unique_id)

        # Prompt weighting. NegPiP (when enabled) owns the whole (word:weight)
        # syntax: negative and time-ranged groups are lifted out of the prompt,
        # encoded on their own, appended as extra text rows, and their attention
        # value vectors are flipped on the DiT side. It must see the RAW prompt,
        # so the in-place parser below only runs when NegPiP is off or failed.
        use_negpip = bool(enable_negpip) and _NegPiP is not None
        if bool(enable_negpip) and _NegPiP is None:
            self._log(
                "NegPiP helper failed to import at startup; (word:weight) falls back "
                "to the in-place weighting.", level="warn")
        if use_negpip:
            # Hardcoded per design: exact-flip strength, positive emphasis scaled,
            # text rows not protected (a few-percent-per-step query split that only
            # pays off when strong negatives invert; exact flip rarely does).
            neg_cfg = {
                "value_strength": 1.0,
                "apply_positive_weights": True,
                "protect_text_rows": False,
            }
            try:
                neg_clip = self._get_negpip_clip(models["clip"])
                model = _NegPiP.patch_model(model, neg_cfg)
                models["clip"] = neg_clip
                self._log("NegPiP active: (word:-N) flips value vectors; "
                          "@start-end / @v / @a time ranges supported.", level="ok")
            except Exception as e:
                use_negpip = False
                self._log(f"NegPiP setup failed ({type(e).__name__}: {e}); "
                          "falling back to the in-place weighting.", level="warn")
        if not use_negpip:
            weight_pairs, clean_prompt = _h3_parse_prompt_weights(models["clip"], config["prompt"], self._log)
            if weight_pairs:
                config["prompt"] = clean_prompt
                to = model.model_options.get("transformer_options", {}).copy()
                to["minimax_h3_token_weights"] = weight_pairs
                model.model_options["transformer_options"] = to
                dm = model.get_model_object("diffusion_model")
                patch = _H3WeightPatch()
                for idx, block in enumerate(dm.blocks):
                    model.add_object_patch(
                        f"diffusion_model.blocks.{idx}.attn.forward",
                        patch.__get__(block.attn, block.attn.__class__),
                    )
                self._log(f"Prompt weighting active on {len(weight_pairs)} token(s).", level="ok")

        override_megapixels = config["override_megapixels"]
        if override_megapixels is not None:
            megapixels_target = float(override_megapixels)
            self._log(f"Overriding megapixels_target from US Config -> {megapixels_target}", level="ok")
        elif cnet is not None:
            megapixels_target = float(cnet["megapixels"])
            self._log(f"ControlNet: megapixels target {megapixels_target} (from CNET pipe; "
                      "US Config MegaPixels override wins if wired)", level="ok")
        # REF2VA reference validation runs before any length source reads them.
        if mode == MODE_REF2VA:
            for socket, video_frames in sorted(config["ref_videos"].items()):
                if len(video_frames.shape) != 4:
                    raise ValueError(
                        f"'{socket} (REF2VA)' must be an IMAGE batch of video frames."
                    )
                if int(video_frames.shape[0]) < 5:
                    raise ValueError(
                        f"'{socket} (REF2VA)' needs at least 5 frames (~0.2s at 24 fps); got {int(video_frames.shape[0])}."
                    )
            for socket, ref_audio in sorted(config["ref_audios"].items()):
                waveform = ref_audio.get("waveform", None) if isinstance(ref_audio, dict) else None
                if waveform is None or not isinstance(waveform, torch.Tensor):
                    raise ValueError(f"'{socket} (REF2VA)' carries no audio waveform.")

        # Frame-count precedence: CNET control video > US Config Frames override
        # > Video Length Override (longest ref video) > Audio Length Override
        # (longest ref audio) > the Unified Sampler Duration. Every branch logs
        # its source so the workflow's logic is visible in the console.
        override_frames = config["override_frames"]
        cnet_frames = int(cnet.get("frames", 0)) if cnet is not None else 0
        if cnet_frames > 0:
            length_frames = cnet_frames
            if override_frames is not None and int(override_frames) != cnet_frames:
                self._log(
                    "CNET Pipeline uses the control_video quantized frame count "
                    f"({cnet_frames}); the US Config Frames override "
                    f"({int(override_frames)}) is ignored.", level="warn")
            else:
                self._log(
                    "CNET Pipeline uses the control_video quantized frame count "
                    f"-> {cnet_frames} frames.", level="ok")
        elif override_frames is not None:
            length_frames = int(override_frames)
            self._log(f"US Config Frames override -> {length_frames} frames.", level="ok")
        elif mode == MODE_REF2VA and bool(video_frames_override) and config["ref_videos"]:
            longest_video = max(int(v.shape[0]) for v in config["ref_videos"].values())
            length_frames = max(5, longest_video)
            self._log(
                f"Video Length Override: longest Ref Video -> {length_frames} frames.",
                level="ok")
        elif mode == MODE_REF2VA and bool(audio_frames_override) and config["ref_audios"]:
            longest_audio_frames = 0
            for ref_audio in config["ref_audios"].values():
                waveform = ref_audio.get("waveform", None) if isinstance(ref_audio, dict) else None
                sample_rate = ref_audio.get("sample_rate", None) if isinstance(ref_audio, dict) else None
                if waveform is None or not sample_rate:
                    continue
                seconds = float(waveform.shape[-1]) / float(sample_rate)
                longest_audio_frames = max(
                    longest_audio_frames, int(round(seconds * FPS))
                )
            if longest_audio_frames > 0:
                length_frames = max(5, longest_audio_frames)
                self._log(
                    f"Audio Length Override: longest Ref Audio -> {length_frames} frames.",
                    level="ok")
            else:
                self._log(
                    "No Ref Audio length found; falling back to the Unified Sampler "
                    f"Duration ({length_frames} frames).", level="ok")
        elif mode == MODE_REF2VA and (bool(video_frames_override) or bool(audio_frames_override)):
            self._log(
                "No Ref Video connected; falling back to the Unified Sampler "
                f"Duration ({length_frames} frames).", level="ok")
        else:
            self._log(
                f"Using the Unified Sampler Duration ({length_frames} frames).", level="ok")

        ref_aspect = self._resolve_ref_aspect(aspect_ratio, config)
        i2v_frames = mode == MODE_FL2VA and (
            config["first_frame"] is not None or config["last_frame"] is not None)
        if cnet is not None and cnet.get("aspect") is not None:
            # The control video's aspect defines the canvas; the tab aspect
            # widget is silenced. Megapixels is still effective (pipe input or
            # US Config override). In I2V, fl_aspect_mode "ControlNet" keeps the
            # keyframe canvas instead and lets the native patch resize the
            # control video to it.
            if i2v_frames and str(fl_aspect_mode) == "ControlNet":
                width, height, config["first_frame"], config["last_frame"] = (
                    self._resolve_i2v_canvas(
                        config["first_frame"], config["last_frame"],
                        megapixels_target, "Optimal"))
                self._log(
                    f"ControlNet canvas: {width}x{height} (I2V keyframe aspect; "
                    "control video resized to it)", level="ok")
            else:
                width, height = self._dims_from_aspect_round32(
                    megapixels_target, cnet["aspect"][0], cnet["aspect"][1])
                if i2v_frames:
                    if config["first_frame"] is not None:
                        config["first_frame"] = self._resize_crop_cover(
                            config["first_frame"], width, height)
                    if config["last_frame"] is not None:
                        config["last_frame"] = self._resize_crop_cover(
                            config["last_frame"], width, height)
                    self._log(
                        f"ControlNet canvas: {width}x{height} (aspect from control video); "
                        "I2V keyframes cover-resized (lanczos + center-crop), "
                        "fl_aspect_mode ignored", level="ok")
                else:
                    self._log(
                        f"ControlNet canvas: {width}x{height} (aspect from control video; "
                        "tab aspect ignored)", level="ok")
        elif ref_aspect is not None:
            width, height = self._dims_from_aspect_round32(
                megapixels_target, ref_aspect[0], ref_aspect[1])
            if i2v_frames:
                if config["first_frame"] is not None:
                    config["first_frame"] = self._resize_crop_cover(
                        config["first_frame"], width, height)
                if config["last_frame"] is not None:
                    config["last_frame"] = self._resize_crop_cover(
                        config["last_frame"], width, height)
                self._log(
                    f"Canvas from {str(aspect_ratio)}: {width}x{height} "
                    f"(aspect {ref_aspect[0]}:{ref_aspect[1]}, nearest 32); "
                    "I2V keyframes cover-resized", level="ok")
            else:
                self._log(
                    f"Canvas from {str(aspect_ratio)}: {width}x{height} "
                    f"(aspect {ref_aspect[0]}:{ref_aspect[1]}, nearest 32)",
                    level="ok")
        elif i2v_frames:
            width, height, config["first_frame"], config["last_frame"] = (
                self._resolve_i2v_canvas(
                    config["first_frame"],
                    config["last_frame"],
                    megapixels_target,
                    fl_aspect_mode,
                )
            )
        else:
            width, height = self._dims_from_megapixels_aspect(megapixels_target, aspect_ratio)
        length = self._length_from_frames(length_frames)
        if length > 362:
            self._log(
                f"{length} frames exceeds the trained range (362); generation quality may degrade.",
                level="warn",
            )
        self._log(f"Canvas: {width}x{height} | length: {length} frames @ {FPS:.0f} fps", level="ok")

        if cnet is not None:
            model = self._apply_cnet_control(
                model, cnet, models["vae"], mode, use_turbo)

        if mode != MODE_FL2VA and (config["first_frame"] is not None or config["last_frame"] is not None):
            self._log("First/Last Frame inputs are I2V-only; ignoring them in this mode.", level="warn")
        if mode != MODE_REF2VA and (
            config["ref_images"] or config["ref_videos"] or config["ref_video_audios"] or config["ref_audios"]
        ):
            self._log("Ref * inputs are R2V-only. We can still use them to define the aspect ratio.", level="warn")

        self._progress(2, total_steps, "Building conditioning and AV latent")
        positive, latent = self._build_conditioning_and_latent(
            mode,
            models["clip"],
            models["vae"],
            models["audio_vae"],
            config["prompt"],
            config,
            width,
            height,
            length,
        )

        if low_vram:
            self._log("Low VRAM: unloading CLIP after conditioning", level="ok")
            self._offload_clip(models["clip"])

        noise_obj = self._result_tuple(RandomNoise.execute(config["seed"]))[0]
        sampler_name = "euler" if use_turbo else SAMPLER_NAME
        sampler_obj = self._result_tuple(KSamplerSelect.execute(sampler_name))[0]
        sigmas_obj = self._result_tuple(
            BasicScheduler.execute(model, SCHEDULER_NAME, step_count, 1.0)
        )[0]
        guider_obj = self._result_tuple(BasicGuider.execute(model, positive))[0]
        # Wrap guider so the previewer sees correct latent shapes / fps for video sweep
        if live_preview:
            try:
                guider_obj = _PreviewFixGuider(guider_obj, fps_override=FPS)
                _PREVIEW_STATE.fps_override = float(FPS)
            except Exception:
                pass

        if low_vram:
            self._log("Low VRAM: unloading VAEs before sampling", level="ok")
            self._offload_vae(models["vae"])
            self._offload_vae(models["audio_vae"])
            mm.soft_empty_cache()

        self._progress(3, total_steps, f"Sampling ({sampler_name}/{SCHEDULER_NAME}, {step_count} steps)")
        sampled = self._result_tuple(
            SamplerCustomAdvanced.execute(noise_obj, guider_obj, sampler_obj, sigmas_obj, latent)
        )
        output_latent = sampled[0] if len(sampled) > 0 else latent

        samples = output_latent.get("samples") if isinstance(output_latent, dict) else None
        if samples is None:
            raise RuntimeError("Sampler returned no packed latent output.")
        streams = samples.unbind() if getattr(samples, "is_nested", False) else (samples,)
        video_stream = streams[0]
        audio_stream = streams[-1] if len(streams) > 1 else None

        if enable_derope:
            # The derope cycle needs pass-1 pixels/audio to smear and the VAEs
            # for the re-encode, so they come back now (the VAEs were offloaded
            # before sampling only in the low_vram flow; the model stays for
            # pass 2 and is unloaded after it, not before decode).
            if low_vram:
                self._log("Reloading VAEs for derope", level="ok")
                self._reload_vae(models["vae"])
                self._reload_vae(models["audio_vae"])

            video_latent = {"samples": video_stream}
            if vae_decode_tiled:
                images = nodes.VAEDecodeTiled().decode(
                    models["vae"], video_latent, 512, 64, 64, 8)[0]
            else:
                images = nodes.VAEDecode().decode(models["vae"], video_latent)[0]
            audio = None
            if audio_stream is not None:
                audio = vae_decode_audio(models["audio_vae"], {"samples": audio_stream})

            images, audio = self._derope_pass2(
                model=model,
                models=models,
                config=config,
                mode=mode,
                positive=positive,
                pass1_latent=output_latent,
                images1=images,
                audio1=audio,
                width=width,
                height=height,
                length=length,
                use_turbo=use_turbo,
                step_count=step_count,
                sampler_obj=sampler_obj,
                vae_decode_tiled=vae_decode_tiled,
                unload_before_decode=unload_before_decode,
                total_steps=total_steps,
                derope_preset=derope_preset,
                derope_inject=derope_inject,
                derope_audio=derope_audio,
                derope_mode=derope_mode,
                derope_profile=derope_profile,
                derope_abstain=derope_abstain,
                derope_protect_tail=derope_protect_tail,
                derope_adapter=enable_derope_adapter,
            )
        else:
            if unload_before_decode:
                self._log("Unload-before-decode enabled: unloading diffusion model", level="ok")
                unload_errors = self._unload_sampling_model(model)
                if unload_errors:
                    self._log("Unload-before-decode warnings: " + " | ".join(unload_errors), level="warn")

            # unload_before_decode evicts every loaded model (VAEs included) via
            # unload_all_models, so the VAEs must be pulled back whenever that ran,
            # not only in the low_vram flow.
            if low_vram or unload_before_decode:
                self._log("Reloading VAEs for decode", level="ok")
                self._reload_vae(models["vae"])
                self._reload_vae(models["audio_vae"])

            self._progress(4, total_steps, "Decoding video/audio")
            video_latent = {"samples": video_stream}
            if vae_decode_tiled:
                images = nodes.VAEDecodeTiled().decode(
                    models["vae"],
                    video_latent,
                    512,
                    64,
                    64,
                    8,
                )[0]
            else:
                images = nodes.VAEDecode().decode(models["vae"], video_latent)[0]

            audio = None
            if audio_stream is not None:
                audio = vae_decode_audio(models["audio_vae"], {"samples": audio_stream})

        gain = 10.0 ** (float(generated_audio_gain_db) / 20.0)
        waveform = audio.get("waveform", None) if audio is not None else None
        if waveform is not None:
            audio = dict(audio)
            audio["waveform"] = waveform * gain

        self._progress(total_steps, total_steps, f"{mode} complete")
        if audio is None:
            raise RuntimeError("Packed latent contained no audio stream to decode.")
        return (images, audio)

    def _derope_pass2(
        self,
        model,
        models,
        config,
        mode,
        positive,
        pass1_latent,
        images1,
        audio1,
        width,
        height,
        length,
        use_turbo,
        step_count,
        sampler_obj,
        vae_decode_tiled,
        unload_before_decode,
        total_steps,
        derope_preset,
        derope_inject,
        derope_audio,
        derope_mode,
        derope_profile,
        derope_abstain,
        derope_protect_tail,
        derope_adapter,
    ):
        """De-rope second pass: jerk oracle -> hold map -> time smear -> re-encode ->
        partial-denoise re-render -> exact recovery. Returns (images, audio) at the
        original world length (pass-1 output when nothing needs holding)."""
        # A named preset owns every dial; "Custom" (or an unknown legacy value)
        # falls through to the individual inputs. Legacy preset names keep their
        # old q / d_max but leave the other knobs to the individual inputs.
        preset = DEROPE_PRESETS.get(str(derope_preset))
        if preset is not None:
            q = preset["q"]
            d_max = preset["d_max"]
            derope_mode = preset["mode"]
            derope_inject = preset["inject"]
            derope_profile = preset["profile"]
            derope_abstain = preset["abstain"]
            derope_protect_tail = preset["protect_tail"]
            derope_audio = preset["audio"]
        else:
            q, d_max = DEROPE_LEGACY_PRESETS.get(str(derope_preset), (0.75, 4))

        self._progress(4, total_steps, "Derope: oracle + smear + encode")
        z = _derope_video_component(pass1_latent)
        prof = _derope_jerk_profile(z, derope_profile)
        if str(derope_mode).startswith("uniform"):
            holds = [DEROPE_UNIFORM_DILATION] * length
        else:
            contrast = float(prof.max() / max(float(prof.mean()), 1e-8))
            if derope_abstain > 0 and contrast < derope_abstain:
                self._log(
                    f"Derope: jerk contrast {contrast:.2f} < {derope_abstain:g}; motion is "
                    "not fast enough to de-rope, returning pass-1 output.", level="warn")
                return images1, audio1
            holds = _derope_compile_holds(prof, length, q, d_max)
        holds = _derope_expand_hold_map_to_end(holds)
        # protect_tail runs after expand_to_end so the trailing expansion cannot
        # re-dilate the frames it was asked to keep at real time. Adaptive only:
        # uniform's whole point is that nothing is at real time.
        if derope_protect_tail > 0 and not str(derope_mode).startswith("uniform"):
            holds = list(holds)
            for i in range(max(0, len(holds) - int(derope_protect_tail)), len(holds)):
                holds[i] = 1

        n_held = sum(1 for h in holds if h > 1)
        if n_held == 0:
            self._log("Derope: hold map dilates nothing; returning pass-1 output.",
                      level="warn")
            return images1, audio1

        smeared, holds_used = _derope_smear(images1, holds)
        dilated = len(smeared)
        time_x = (_derope_token_count(dilated) / _derope_token_count(length)) ** DEROPE_COST_EXP
        self._log(
            f"Derope: {length}f -> {dilated}f effective regen "
            f"({dilated / length:.2f}x frames, {time_x:.1f}x time per step); "
            f"{n_held} of {length} frames held", level="ok")

        # Re-time any active Fun ControlNet onto the dilated clock so pass 2
        # re-renders the smeared spans with time-aligned guidance. Fails soft:
        # on error the pass-2 control keeps its original timing.
        try:
            n_retimed = _derope_smear_control(model, length, holds_used)
        except Exception as exc:
            n_retimed = 0
            self._log(
                f"Derope: Fun ControlNet re-timing failed ({type(exc).__name__}: {exc}); "
                "pass-2 control keeps its original timing.", level="warn")
        if n_retimed:
            self._log(f"Derope: re-timed {n_retimed} Fun ControlNet(s) to the dilated clock.",
                      level="ok")

        # The Motion Adapter is trained for the de-rope task, so it goes on the
        # pass-2 model only. Applied after the control re-timing so the original
        # patcher (which carries the Fun ControlNet list) stays the one re-timed.
        if derope_adapter:
            try:
                model = _apply_derope_adapter(model, self._log)
            except Exception as exc:
                self._log(f"Derope: Motion Adapter unavailable "
                          f"({type(exc).__name__}: {exc}); pass 2 runs without it.",
                          level="warn")

        video_z = models["vae"].encode(smeared)

        # Seed pass-2's audio rows with the slowed performance so mouths and
        # foley land on the dilated clock; the OUTPUT track keeps pass-1 by
        # default (regenerated audio quality varies).
        audio_latent = None
        seed_audio = audio1 is not None and str(derope_audio) != "pass-2 invents audio"
        if seed_audio:
            try:
                smeared_audio = _derope_audio_smear(audio1, holds_used, fps=24)
                audio_latent = _derope_audio_vae_encode(models["audio_vae"], smeared_audio)
            except Exception as e:
                seed_audio = False
                self._log(f"Derope: audio seeding skipped ({type(e).__name__}: {e}); "
                          "pass 2 will invent its own audio, the output keeps pass-1.",
                          level="warn")

        latent2 = _derope_nested_av_latent(
            video_z, audio_latent, dilated, 0.5 if seed_audio else 1.0)

        # Time-anchored conditioning must be re-built on the dilated clock: a
        # last-frame keyframe or REF2VA refs that sat at real-time positions
        # now sit at dilated ones. Text-only and first-frame conditioning is
        # length-independent and is reused as-is.
        needs_reanchor = bool(
            (mode == MODE_FL2VA and config.get("last_frame") is not None)
            or (mode == MODE_REF2VA and (
                config["ref_images"] or config["ref_videos"]
                or config["ref_video_audios"] or config["ref_audios"]))
        )
        if needs_reanchor:
            positive2, _ = self._build_conditioning_and_latent(
                mode, models["clip"], models["vae"], models["audio_vae"],
                config["prompt"], config, width, height, dilated)
        else:
            positive2 = positive

        derope_scheduler = "beta" if use_turbo else "simple"
        sigmas2 = _derope_inject_sigmas(model, derope_scheduler, step_count, derope_inject)
        noise2 = self._result_tuple(RandomNoise.execute(int(config["seed"]) + 1))[0]
        guider2 = self._result_tuple(BasicGuider.execute(model, positive2))[0]

        self._progress(5, total_steps, "Derope: pass 2 sampling")
        sampled2 = self._result_tuple(
            SamplerCustomAdvanced.execute(noise2, guider2, sampler_obj, sigmas2, latent2)
        )
        output_latent2 = sampled2[0] if len(sampled2) > 0 else latent2
        samples2 = output_latent2.get("samples") if isinstance(output_latent2, dict) else None
        if samples2 is None:
            raise RuntimeError("Derope pass 2 returned no packed latent output.")
        streams2 = samples2.unbind() if getattr(samples2, "is_nested", False) else (samples2,)
        video_stream2 = streams2[0]
        audio_stream2 = streams2[-1] if len(streams2) > 1 else None

        if unload_before_decode:
            self._log("Unload-before-decode: unloading diffusion model after pass 2",
                      level="ok")
            unload_errors = self._unload_sampling_model(model)
            if unload_errors:
                self._log("Unload-before-decode warnings: " + " | ".join(unload_errors),
                          level="warn")
            self._log("Reloading VAEs for derope decode", level="ok")
            self._reload_vae(models["vae"])
            self._reload_vae(models["audio_vae"])

        self._progress(total_steps, total_steps, "Decoding video/audio")
        video_latent2 = {"samples": video_stream2}
        if vae_decode_tiled:
            images2 = nodes.VAEDecodeTiled().decode(
                models["vae"], video_latent2, 512, 64, 64, 8)[0]
        else:
            images2 = nodes.VAEDecode().decode(models["vae"], video_latent2)[0]
        audio2 = None
        if audio_stream2 is not None:
            audio2 = vae_decode_audio(models["audio_vae"], {"samples": audio_stream2})

        images = _derope_recover(images2, holds_used)

        audio = audio1
        if audio2 is not None and audio1 is not None and str(derope_audio) == "pass-2 foley (seeded)":
            try:
                audio = _derope_audio_recover(audio2, holds_used, fps=24,
                                              reference=audio1, reference_mix=0.0)
            except Exception as e:
                self._log(f"Derope: audio recover failed ({type(e).__name__}: {e}); "
                          "keeping pass-1 audio.", level="warn")
                audio = audio1
        return images, audio
