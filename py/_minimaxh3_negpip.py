"""NegPiP for MiniMax H3, embedded for the CRT unified sampler.

NegPiP (hako-mikan/sd-webui-negpip technique ported to H3) makes a token
*subtract* its concept instead of adding it, by flipping the sign of its
attention value vector. MiniMax H3 is a single-stream packed-token DiT: every
block runs one joint self-attention over

    [text | cond rows | audio | video]

with text rows first at [0, text_len). Flipping the V slice of a text row
inside the block attention is exactly NegPiP, so no cross-attention is needed.

Two halves:
- CLIP side: MiniMaxH3Tokenizer tokenizes with disable_weights=True, so
  (word:-1.0) does nothing today. patch_clip re-enables weighting, encodes
  with neutral weights (the conditioning is a Qwen3-VL hidden state, so
  scaling it directly fights the LLM's normalisation), and ships per-token
  weights along as conditioning extras. Negative groups are lifted out of the
  prompt, encoded on their own and appended as extra text rows; time-ranged
  groups are always lifted too.
- DiT side: patch_model adds wrapper hooks that, per sampling call, multiply
  the value vectors of those text rows by the token weight (negative flips).
  Time ranges split attention by query so only the covered rows see the flip.

Exposed API: patch_clip(clip) -> CLIP clone with parsing+weights enabled;
patch_model(model, cfg) -> MODEL clone with the attention wrappers.
"""


import copy
import math
import numbers
import re
import threading
from typing import Any, NamedTuple

import torch

KEY = "minimax_h3_negpip"
COND_WEIGHTS_KEY = "minimax_h3_negpip_weights"
COND_RANGES_KEY = "minimax_h3_negpip_ranges"
COND_APPENDED_KEY = "minimax_h3_negpip_appended"
TOKENS_LIFTED_KEY = "minimax_h3_negpip_lifted"
TEXT_ENCODER_NAME = "qwen3vl_32b"
PAD_TOKEN = 151643
_TEXT_TOKEN_TAG = 1  # adaLN modality tag of a text row

MAX_STRENGTH = 8.0
FPS = 24.0             # video pixel frames per second
AUDIO_LATENT_FPS = 40.0  # audio latent frames per second
OPEN_END = 1.0e9

_MAX_BLOCKS = 999      # block_end default: all blocks (clamped at the model)


def _log(message, *args):
    prefix = "[CRT MiniMaxH3 NegPiP] "
    if args:
        message = message % args
    print(prefix + message)


# --------------------------------------------------------------------- CLIP
NEGPIP_GROUP_RE = re.compile(
    r"(?<!\\)\(([^()]*?):\s*(-?\d+(?:\.\d+)?)\s*"
    r"(?:@\s*([vaVA]?)\s*(\d*(?:\.\d+)?)\s*-\s*(\d*(?:\.\d+)?)\s*)?\)")


def _tidy(text: str) -> str:
    """Close the hole a lifted group leaves behind; the LLM does read the
    punctuation."""
    for _ in range(3):
        previous = text
        text = re.sub(r"(?:,[ \t]*){2,}", ", ", text)
        text = re.sub(r"([.!?;:])[ \t]*,", r"\1", text)
        text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        if text == previous:
            break
    return text.strip(" \t,")


def _split_lifted_groups(text: str):
    """Lift negatives and time-ranged groups out; plain postives stay."""
    lifted = []

    def replace(match):
        phrase = match.group(1).strip()
        weight = float(match.group(2))
        time_range = None
        if match.group(4) is not None or match.group(5) is not None:
            stream = {"v": "video", "a": "audio", "": "both"}[(match.group(3) or "").lower()]
            start = float(match.group(4)) if match.group(4) else 0.0
            end = float(match.group(5)) if match.group(5) else OPEN_END
            time_range = TimeRange(stream, start, max(start, end))
        if weight >= 0.0 and time_range is None:
            return match.group(0)  # plain emphasis belongs in the prompt
        if phrase:
            lifted.append((phrase, weight, time_range))
        return ""

    return _tidy(NEGPIP_GROUP_RE.sub(replace, text)), lifted


def _is_plain_token(x: Any) -> bool:
    return (not torch.is_tensor(x)) and isinstance(x, numbers.Integral)


def _isolated_tokens(inner_tokenizer, phrase: str):
    """Token ids of one phrase on its own (no prompt weighting)."""
    if inner_tokenizer is None:
        return []
    original = getattr(inner_tokenizer.tokenize_with_weights, "_negpip_original_tokenize",
                       inner_tokenizer.tokenize_with_weights)
    tokens = []
    for batch in original(phrase, return_word_ids=False, disable_weights=True):
        for entry in batch:
            if _is_plain_token(entry[0]):
                tokens.append(int(entry[0]))
    return tokens


def _keep_weights(inner_tokenizer):
    """MiniMaxQwenSDTokenizer hardcodes disable_weights=True on text segments."""
    original = getattr(inner_tokenizer, "_negpip_original_tokenize", inner_tokenizer.tokenize_with_weights)

    def tokenize_with_weights(text, *args, **kwargs):
        kwargs["disable_weights"] = False
        return original(text, *args, **kwargs)

    tokenize_with_weights._negpip_original_tokenize = original
    return tokenize_with_weights


def _make_tokenize_with_weights(tokenizer, original_function):
    def tokenize_with_weights(text, *args, **kwargs):
        lifted = []
        if isinstance(text, str):
            text, lifted = _split_lifted_groups(text)
        out = original_function(tokenizer, text, *args, **kwargs)
        if not lifted or not isinstance(out, dict):
            return out
        inner = getattr(tokenizer, TEXT_ENCODER_NAME, None)
        entries = []
        for phrase, weight, time_range in lifted:
            tokens = _isolated_tokens(inner, phrase)
            if tokens:
                entries.append({"tokens": tokens, "weight": weight, "range": time_range})
            else:
                _log("NegPiP: %r tokenized to nothing, ignored.", phrase)
        if entries:
            out = dict(out)
            out[TOKENS_LIFTED_KEY] = entries
        return out

    return tokenize_with_weights


def _strip_weights(section):
    return [(entry[0], 1.0) + tuple(entry[2:]) for entry in section]


def _expanded_weights(section, embeds_info, seq_len: int):
    """(token, weight) pairs -> one weight per encoded hidden state.

    process_tokens() splices image embeds (each expands to many rows) into the
    token stream, so pair index != hidden-state index. embeds_info lists each
    splice's final index and size, reproducing layout exactly.
    """
    weights = []
    embed_weights = []
    for entry in section:
        if _is_plain_token(entry[0]):
            weights.append(float(entry[1]))
        else:
            embed_weights.append(float(entry[1]))

    infos = list(embeds_info or [])
    aligned = len(infos) == len(embed_weights)
    for n, info in enumerate(infos):
        index = int(info.get("index", -1))
        size = int(info.get("size", 0))
        if index < 0 or index > len(weights) or size < 0:
            return None
        w = embed_weights[n] if aligned else 1.0
        weights[index:index] = [w] * size
    if len(weights) < seq_len:
        weights += [1.0] * (seq_len - len(weights))
    if len(weights) != seq_len:
        return None
    return weights


def _encode_lifted(original_encode, name, clip_model, lifted):
    """Encode every lifted phrase on its own; append as extra text rows."""
    rows = [entry["tokens"] for entry in lifted]
    width = max(len(row) for row in rows)
    pad_id = int(getattr(clip_model, "special_tokens", {}).get("pad", PAD_TOKEN))
    batch = [[(int(t), 1.0) for t in row] + [(pad_id, 1.0)] * (width - len(row))
             for row in rows]
    out = original_encode({name: batch})
    encoded = out[0]
    if int(encoded.shape[1]) != width * len(rows):
        return None, [], []
    pieces, weights, ranges = [], [], []
    for k, entry in enumerate(lifted):
        length = len(rows[k])
        pieces.append(encoded[:, k * width:k * width + length, :])
        weights += [float(entry["weight"])] * length
        ranges += [entry["range"]] * length
    return torch.cat(pieces, dim=1), weights, ranges


def _extend_token_tags(extra, count):
    tags = extra.get("minimax_token_tags")
    if tags is None or count <= 0:
        return
    tags = tags.view(-1)
    extra["minimax_token_tags"] = torch.cat(
        [tags, torch.full((count,), _TEXT_TOKEN_TAG, dtype=tags.dtype, device=tags.device)])


def _make_encode_token_weights(cond_stage_model):
    original = getattr(cond_stage_model, "_negpip_original_encode", cond_stage_model.encode_token_weights)
    name = getattr(cond_stage_model, "clip_name", TEXT_ENCODER_NAME)

    def encode_token_weights(token_weight_pairs):
        sections = token_weight_pairs.get(name) if isinstance(token_weight_pairs, dict) else None
        if not sections:
            return original(token_weight_pairs)

        stripped = dict(token_weight_pairs)
        stripped[name] = [_strip_weights(section) for section in sections]

        clip_name = getattr(cond_stage_model, "clip", name)
        clip_model = getattr(cond_stage_model, clip_name)
        captured = []
        had_own = "process_tokens" in clip_model.__dict__
        original_process_tokens = clip_model.process_tokens

        def process_tokens(tokens, device):
            out0 = original_process_tokens(tokens, device)
            captured.append(out0[3])
            return out0

        clip_model.process_tokens = process_tokens
        try:
            out = original(stripped)
        finally:
            if had_own:
                clip_model.process_tokens = original_process_tokens
            else:
                try:
                    del clip_model.process_tokens
                except AttributeError:
                    pass

        if len(sections) != 1:
            _log("NegPiP: %d prompt sections, weights ignored.", len(sections))
            return out

        cond = out[0]
        weights = _expanded_weights(sections[0], captured[-1] if captured else [],
                                    int(cond.shape[1]))
        if weights is None:
            _log("NegPiP: could not map prompt weights onto the conditioning; ignoring them.")
            weights = [1.0] * int(cond.shape[1])
        ranges = [None] * len(weights)

        lifted = token_weight_pairs.get(TOKENS_LIFTED_KEY) if isinstance(token_weight_pairs, dict) else None
        appended = 0
        extra = dict(out[2]) if len(out) > 2 and isinstance(out[2], dict) else {}
        if lifted:
            rows, row_weights, row_ranges = _encode_lifted(original, name, clip_model, lifted)
            if rows is not None:
                cond = torch.cat([cond, rows.to(device=cond.device, dtype=cond.dtype)], dim=1)
                weights += row_weights
                ranges += row_ranges
                appended = len(row_weights)
                _extend_token_tags(extra, appended)

        if all(w == 1.0 for w in weights):
            return out

        extra[COND_WEIGHTS_KEY] = [float(w) for w in weights]
        extra[COND_APPENDED_KEY] = appended
        timed = [[i, r.stream, r.start, r.end] for i, r in enumerate(ranges) if r is not None]
        if timed:
            extra[COND_RANGES_KEY] = timed
        _log("NegPiP: %d weighted token(s) of %d, %d negative, %d appended, %d time ranged.",
             sum(1 for w in weights if w != 1.0), int(cond.shape[1]),
             sum(1 for w in weights if w < 0.0), appended, len(timed))
        return (cond, out[1], extra)

    encode_token_weights._negpip_original_encode = original
    return encode_token_weights


def patch_clip(clip):
    """Return a CLIP clone whose tokenizer parses (word:weight) and whose
    encoder carries the per-token weights / lifted rows as conditioning extras.
    The original CLIP is untouched."""
    tokenizer = getattr(clip, "tokenizer", None)
    cond_stage_model = getattr(clip, "cond_stage_model", None)
    if (cond_stage_model is None or tokenizer is None
            or not hasattr(tokenizer, TEXT_ENCODER_NAME)
            or not hasattr(cond_stage_model, TEXT_ENCODER_NAME)):
        raise RuntimeError(
            "NegPiP needs the MiniMax H3 CLIP (CLIPLoader / DualCLIPLoader type "
            "minimax_h3).")

    new_clip = clip.clone() if hasattr(clip, "clone") else copy.copy(clip)
    new_clip.tokenizer = copy.copy(tokenizer)
    inner_tokenizer = copy.copy(getattr(tokenizer, TEXT_ENCODER_NAME))
    inner_tokenizer.tokenize_with_weights = _keep_weights(getattr(tokenizer, TEXT_ENCODER_NAME))
    setattr(new_clip.tokenizer, TEXT_ENCODER_NAME, inner_tokenizer)
    new_clip.tokenizer.tokenize_with_weights = _make_tokenize_with_weights(
        new_clip.tokenizer, type(tokenizer).tokenize_with_weights)

    new_clip.cond_stage_model = copy.copy(cond_stage_model)
    new_clip.cond_stage_model.encode_token_weights = _make_encode_token_weights(cond_stage_model)
    return new_clip


# ------------------------------------------------------------------------- DiT part

class TimeRange(NamedTuple):
    stream: str  # "both" | "video" | "audio"
    start: float
    end: float


def _minimax_module():
    import comfy.ldm.minimax.model as minimax_model
    return minimax_model


def _is_minimax_h3(dm):
    try:
        return isinstance(dm, _minimax_module().MiniMaxH3Model)
    except Exception:
        return False


def _layout_for_call(dm, payload, context, x):
    minimax_model = _minimax_module()
    video, audio = x[0], x[1]
    text_len = int(context.shape[1])
    latent_t = int(video.shape[2])
    # the layout the forward will build uses the patch-padded spatial dims
    lat_h = (int(video.shape[3]) + 1) // 2 * 2
    lat_w = (int(video.shape[4]) + 1) // 2 * 2
    audio_t = int(audio.shape[-1])
    signature = (text_len, latent_t, lat_h, lat_w, audio_t)
    layout = payload.get("layout")
    if layout is None or layout.signature != signature:
        layout = minimax_model.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                            keyframes=payload.get("keyframes"),
                                            refs=payload.get("refs"))
    return layout, latent_t, audio_t


def _frame_starts(latent_t):
    """Pixel frame each latent frame starts at: FRAME_PER_TOKEN repeats 1,4,4,4,4."""
    per_token = _minimax_module().FRAME_PER_TOKEN
    starts, acc = [], 0
    for k in range(latent_t):
        starts.append(acc)
        acc += per_token[k % len(per_token)]
    return starts, per_token


def _range_rows(time_range, layout, latent_t, audio_t):
    """Seconds -> packed rows of the generated streams inside that window."""
    segments = {kind: (a, b) for a, b, kind in layout.segments}
    rows = []

    video = segments.get("video")
    if time_range.stream in ("both", "video") and video is not None and latent_t > 0:
        va, vb = video
        frame_rows = (vb - va) // latent_t
        starts, per_token = _frame_starts(latent_t)
        first = time_range.start * FPS
        last = time_range.end * FPS
        hit = [k for k in range(latent_t)
               if starts[k] + per_token[k % len(per_token)] > first and starts[k] < last]
        if hit:
            rows.append((va + hit[0] * frame_rows, va + (hit[-1] + 1) * frame_rows))

    audio = segments.get("audio")
    if time_range.stream in ("both", "audio") and audio is not None and audio_t > 0:
        aa, _ = audio
        i0 = max(0, int(math.floor(time_range.start * AUDIO_LATENT_FPS)))
        i1 = min(audio_t, int(math.ceil(time_range.end * AUDIO_LATENT_FPS)))
        if i1 > i0:
            rows.append((aa + i0, aa + i1))
            rows.append((aa + audio_t + i0, aa + audio_t + i1))
    return rows


def _complement_rows(time_range, layout, latent_t, audio_t, seq_len):
    """Every packed row the range does not cover (text and cond rows included)."""
    inside = sorted(_range_rows(time_range, layout, latent_t, audio_t))
    rows, cursor = [], 0
    for a, b in inside:
        if a > cursor:
            rows.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < seq_len:
        rows.append((cursor, seq_len))
    return rows


def _chunks_by_active_set(group_rows, seq_len):
    """Partition [0, seq_len) into runs where the set of active groups is constant."""
    bounds = {0, seq_len}
    for rows in group_rows:
        for a, b in rows:
            bounds.add(max(0, min(a, seq_len)))
            bounds.add(max(0, min(b, seq_len)))
    ordered = sorted(bounds)
    chunks = []
    for a, b in zip(ordered, ordered[1:]):
        if b <= a:
            continue
        active = frozenset(i for i, rows in enumerate(group_rows)
                           if any(s <= a and b <= t for s, t in rows))
        if chunks and chunks[-1][2] == active:
            chunks[-1] = (chunks[-1][0], b, active)
        else:
            chunks.append((a, b, active))
    return chunks


def _multipliers(weights, ranges, appended, cfg):
    value_strength = float(cfg.get("value_strength", 1.0))
    apply_positive = bool(cfg.get("apply_positive_weights", True))
    protect = bool(cfg.get("protect_text_rows", False))
    everything = TimeRange("both", 0.0, OPEN_END)
    first_appended = len(weights) - max(0, int(appended or 0))

    glob = ([], [])
    groups = {}
    for i, w in enumerate(weights):
        w = float(w)
        is_appended = i >= first_appended
        if w == 1.0 and not is_appended:
            continue
        if w < 0.0:
            m = w * value_strength
        elif is_appended or apply_positive:
            m = w
        else:
            continue
        time_range = ranges[i] if ranges else None
        if time_range is None and not protect:
            glob[0].append(i)
            glob[1].append(m)
            continue
        window = time_range or everything
        target = groups.setdefault(("range", window), ([], []))
        target[0].append(i)
        target[1].append(m)
        if is_appended:
            silence = groups.setdefault(("silence", window), ([], []))
            silence[0].append(i)
            silence[1].append(0.0)
    return glob, list(groups.items())


def _weights_for_call(cfg, transformer_options, context):
    by_uuid = cfg.get("_by_uuid") or {}
    if not by_uuid:
        return None, None, 0
    rows = [by_uuid.get(str(u)) for u in (transformer_options.get("uuids") or [])]
    rows = [r for r in rows if r]
    if not rows:
        return None, None, 0
    weights, ranges, appended = rows[0]
    text_len = int(context.shape[1])
    if len(weights) != text_len:
        _log("NegPiP: conditioning is %d tokens but %d weights carried, skipped.",
             text_len, len(weights))
        return None, None, 0
    return weights, ranges, appended


def _index_tensors(state, device, dtype):
    cached = state["cache"].get((device, dtype))
    if cached is None:
        def build(positions, mults):
            return (torch.tensor(positions, device=device, dtype=torch.long),
                    torch.tensor(mults, device=device, dtype=dtype).view(1, 1, -1, 1))

        cached = {
            "global": build(*state["global"]) if state["global"][0] else None,
            "groups": [build(positions, mults) for positions, mults in state["groups"]],
            "timed": (torch.tensor(state["timed_positions"], device=device, dtype=torch.long)
                      if state["timed_positions"] else None),
        }
        state["cache"][(device, dtype)] = cached
    return cached


def _make_attention_hook(original_attention, container_class):
    def attention(q, k, v, heads, mask=None, skip_reshape=False, transformer_options={}, **kwargs):
        state = getattr(_ACTIVE, "state", None)
        if state is None:
            return original_attention(q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                                      transformer_options=transformer_options, **kwargs)

        wrapped = isinstance(q, container_class)
        qt = q.take() if wrapped else q
        kt = k.take() if wrapped else k
        vt = v.take() if wrapped else v

        def wrap(t):
            return container_class(t) if wrapped else t

        def run(query, values):
            return original_attention(wrap(query), wrap(kt), wrap(values), heads, mask=mask,
                                      skip_reshape=skip_reshape,
                                      transformer_options=transformer_options, **kwargs)

        index = _index_tensors(state, qt.device, qt.dtype)
        if index["global"] is not None:
            positions, mults = index["global"]
            vt = vt.clone()
            vt[..., positions, :] *= mults

        chunks = state["chunks"]
        if not (chunks and mask is None and skip_reshape
                and qt.ndim == 4 and int(qt.shape[2]) == state["seq_len"]):
            return run(qt, vt)

        timed = index["timed"]
        if timed.numel() == 0:
            return run(qt, vt)

        pristine = vt[..., timed, :].clone()
        outs = []
        for a, b, active in chunks:
            vt_c = vt.clone()
            vt_c[..., timed, :] = pristine
            for group in active:
                positions, mults = index["groups"][group]
                vt_c[..., positions, :] *= mults
            outs.append(run(qt[:, :, a:b, :], vt_c))
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

    return attention


_ACTIVE = threading.local()
_LOG_STATE = {}


def _restore_modules(installed):
    for module, original_forward, had_own in installed:
        if had_own:
            module.forward = original_forward
        else:
            try:
                del module.forward
            except AttributeError:
                pass
        try:
            del module._negpip_patched
        except AttributeError:
            pass


def _patch_attention_module(attn, state):
    if getattr(attn, "_negpip_patched", False):
        return None
    original_forward = attn.forward
    had_own = "forward" in attn.__dict__

    def forward(*args, **kwargs):
        previous = getattr(_ACTIVE, "state", None)
        _ACTIVE.state = state
        try:
            return original_forward(*args, **kwargs)
        finally:
            _ACTIVE.state = previous

    attn.forward = forward
    attn._negpip_patched = True
    return (attn, original_forward, had_own)


def _block_indices(cfg, count):
    start = max(0, int(cfg.get("block_start", 0)))
    end = min(count - 1, int(cfg.get("block_end", _MAX_BLOCKS)))
    stride = max(1, int(cfg.get("block_stride", 1)))
    return range(start, end + 1, stride)


def _collect_weights_by_uuid(conds):
    out = {}
    if not isinstance(conds, list):
        return out
    for group in conds:
        if not isinstance(group, list):
            continue
        for cond in group:
            if not isinstance(cond, dict):
                continue
            weights = cond.get(COND_WEIGHTS_KEY)
            cond_uuid = cond.get("uuid")
            if not weights or cond_uuid is None:
                continue
            try:
                values = [float(w) for w in weights]
            except (TypeError, ValueError):
                continue
            ranges = [None] * len(values)
            for item in cond.get(COND_RANGES_KEY) or []:
                try:
                    position, stream, start, end = int(item[0]), str(item[1]), float(item[2]), float(item[3])
                except (TypeError, ValueError, IndexError):
                    continue
                if 0 <= position < len(ranges):
                    ranges[position] = TimeRange(stream, start, end)
            try:
                appended = int(cond.get(COND_APPENDED_KEY) or 0)
            except (TypeError, ValueError):
                appended = 0
            out[str(cond_uuid)] = (values, ranges, appended)
    return out


def diffusion_model_wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
    cfg = (transformer_options or {}).get(KEY) or {}
    if not cfg.get("enabled", False):
        return executor(x, timestep, context, transformer_options, **kwargs)

    dm = getattr(executor, "class_obj", None)
    if not _is_minimax_h3(dm):
        _log("NegPiP: this is not a MiniMax H3 DiT, doing nothing.")
        return executor(x, timestep, context, transformer_options, **kwargs)

    weights, ranges, appended = _weights_for_call(cfg, transformer_options, context)
    if weights is None:
        return executor(x, timestep, context, transformer_options, **kwargs)

    glob, groups = _multipliers(weights, ranges, appended, cfg)
    if not glob[0] and not groups:
        return executor(x, timestep, context, transformer_options, **kwargs)

    layout, latent_t, audio_t = _layout_for_call(dm, kwargs.get("minimax_payload") or {}, context, x)
    seq_len = int(layout.seq_len)
    group_rows = []
    for (kind, window), _ in groups:
        if kind == "range":
            rows = _range_rows(window, layout, latent_t, audio_t)
            if not rows:
                _log("NegPiP: %s%s covers no frame of this clip.",
                     window.stream, "-%g" % window.end if window.end < OPEN_END else "")
        else:
            rows = _complement_rows(window, layout, latent_t, audio_t, seq_len)
        group_rows.append(rows)

    state = {
        "seq_len": seq_len,
        "global": glob,
        "groups": [mults for _, mults in groups],
        "timed_positions": sorted({i for _, (positions, _) in groups for i in positions}),
        "chunks": _chunks_by_active_set(group_rows, seq_len) if groups else [],
        "cache": {},
    }

    blocks = _block_indices(cfg, len(dm.blocks))
    _log("NegPiP: %d always-on and %d row-restricted token(s) of %d "
         "(%d appended), %d/%d blocks.",
         len(glob[0]), len(state["timed_positions"]), int(context.shape[1]),
         int(appended or 0), len(list(blocks)), len(dm.blocks))

    minimax_model = _minimax_module()
    original_attention = minimax_model.optimized_attention
    minimax_model.optimized_attention = _make_attention_hook(
        original_attention, minimax_model.AttentionTensorContainer)
    installed = []
    try:
        for i in blocks:
            entry = _patch_attention_module(dm.blocks[i].attn, state)
            if entry is not None:
                installed.append(entry)
        return executor(x, timestep, context, transformer_options, **kwargs)
    finally:
        _restore_modules(installed)
        minimax_model.optimized_attention = original_attention


def calc_cond_batch_wrapper(executor, model, conds, x_in, timestep, model_options):
    by_uuid = _collect_weights_by_uuid(conds)
    if by_uuid:
        model_options = dict(model_options)
        transformer_options = dict(model_options.get("transformer_options", {}))
        cfg = dict(transformer_options.get(KEY, {}))
        cfg["_by_uuid"] = by_uuid
        transformer_options[KEY] = cfg
        model_options["transformer_options"] = transformer_options
    return executor(model, conds, x_in, timestep, model_options)


def _outer_sample_wrapper(executor, *args, **kwargs):
    _LOG_STATE.clear()
    return executor(*args, **kwargs)


def patch_model(model, cfg={}):
    """Return a MODEL clone carrying the NegPiP DiT-side wrappers.
    cfg: {value_strength, apply_positive_weights, protect_text_rows,
          block_start, block_end, block_stride}."""
    from comfy.patcher_extension import WrappersMP

    patched = model.clone()
    wrapper_cfg = dict(cfg)
    wrapper_cfg.setdefault("enabled", True)
    patched.model_options.setdefault("transformer_options", {})[KEY] = wrapper_cfg
    for wrapper_type, wrapper in ((WrappersMP.OUTER_SAMPLE, _outer_sample_wrapper),
                                  (WrappersMP.CALC_COND_BATCH, calc_cond_batch_wrapper),
                                  (WrappersMP.DIFFUSION_MODEL, diffusion_model_wrapper)):
        if hasattr(patched, "remove_wrappers_with_key"):
            patched.remove_wrappers_with_key(wrapper_type, KEY)
        patched.add_wrapper_with_key(wrapper_type, KEY, wrapper)
    return patched


__all__ = ["patch_clip", "patch_model", "KEY", "MAX_STRENGTH", "NEGPIP_GROUP_RE"]