import os
from pathlib import Path

import torch

from ._crawl_common import pick_folder, scan_tree, to_int


def _load_audio_file(path):
    """Decode audio -> (waveform [channels, samples] float32, sample_rate).

    Mirrors ComfyUI core's own audio loader (comfy_extras/nodes_audio.py):
    pure PyAV, no torchaudio — torchaudio 2.11+ hard-requires torchcodec,
    which is not installed in this environment.
    """
    import av

    with av.open(str(path)) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = af.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames = []
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()
            frames.append(buf)

        if not frames:
            raise ValueError("No audio frames decoded.")

        waveform = torch.cat(frames, dim=1)

    # f32 PCM conversion (as in core's f32_pcm)
    if waveform.dtype == torch.int16:
        waveform = waveform.float() / (2 ** 15)
    elif waveform.dtype == torch.int32:
        waveform = waveform.float() / (2 ** 31)
    elif not waveform.dtype.is_floating_point:
        raise ValueError(f"Unsupported wav dtype: {waveform.dtype}")

    return waveform, int(sample_rate)


class AudioLoaderCrawl:
    def __init__(self):
        # Instance-level cache to store file lists and folder modification times.
        self.cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to the folder containing audio files"}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Selects the file. With no subfolder seed connected, crawls each folder's first file, then each folder's second file, etc.",
                    },
                ),
                "file_extension": (
                    ["wav", "mp3", "flac", "ogg"],
                    {"default": "wav", "tooltip": "File extension to filter for"},
                ),
                "max_depth": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1,
                        "max": 100,
                        "tooltip": "Subfolder crawl depth. 0 = only the root folder, 1 = one level deep, -1 = infinite.",
                    },
                ),
                "remove_extension": ("BOOLEAN", {"default": False, "tooltip": "Output filename without extension"}),
                "max_length_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "step": 0.1,
                        "tooltip": "Maximum length of the audio in seconds (0 for no limit)",
                    },
                ),
                "start_offset_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "step": 0.1,
                        "tooltip": "Start loading the audio from this offset in seconds",
                    },
                ),
                "gain_db": (
                    "FLOAT",
                    {"default": 0.0, "min": -120.0, "max": 120.0, "step": 0.1, "tooltip": "Gain in decibels (dB)"},
                ),
            },
            "optional": {
                "subfolder_seed": (
                    "INT",
                    {
                        "forceInput": True,
                        "tooltip": "Selects the folder as seed % number_of_folders. Leave unconnected to crawl folders column-major with the main seed.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "file_name", "file_path")
    FUNCTION = "load_audio"
    CATEGORY = "CRT/Load"

    def load_audio(
        self,
        folder_path,
        seed,
        file_extension,
        max_depth,
        remove_extension,
        max_length_seconds,
        start_offset_seconds,
        gain_db,
        subfolder_seed=None,
    ):
        # Failure returns None for audio: downstream reference nodes (e.g. MiniMax H3)
        # skip None audio cleanly. Do NOT substitute fake silence here — a tiny silent
        # tensor resamples to 0 samples at 32 kHz and crashes the audio VAE instead.
        safe_return = (None, "", "")

        if not folder_path or not folder_path.strip():
            print("[ERROR] Error: Folder path is empty.")
            return safe_return

        folder = Path(folder_path.strip())
        if not folder.is_dir():
            print(f"[ERROR] Error: Folder '{folder}' not found or is not a directory.")
            return safe_return

        # Ensure file extension has a dot
        if not file_extension.startswith('.'):
            file_extension = f".{file_extension}"

        seed = to_int(seed)
        max_depth = to_int(max_depth)
        sub_seed = to_int(subfolder_seed) if subfolder_seed is not None else None

        try:
            # --- Smart Caching Logic ---
            cache_key = (str(folder.resolve()), max_depth)
            current_mtime = folder.stat().st_mtime

            if cache_key not in self.cache or self.cache[cache_key]['mtime'] != current_mtime:
                print(f"[INFO] Folder changed or not cached. Scanning '{folder}' for '{file_extension}' files...")
                folders, files_by_folder = scan_tree(folder, max_depth)
                self.cache[cache_key] = {'folders': folders, 'files': files_by_folder, 'mtime': current_mtime}
                print(f"[OK] Cached folder tree from '{folder}'.")
            folders = self.cache[cache_key]['folders']
            files_by_folder = self.cache[cache_key]['files']

            selected_folder, inner_seed = pick_folder(folders, sub_seed, seed)
            if selected_folder is None:
                print("[ERROR] No folders found under the given path.")
                return safe_return

            files = [f for f in files_by_folder.get(os.path.normpath(str(selected_folder)), []) if f.suffix.lower() == file_extension]
            files.sort()

            if not files:
                print(f"[ERROR] Warning: No files with extension '{file_extension}' found in '{selected_folder}'.")
                return safe_return

            # --- Deterministic and Safe Selection ---
            num_files = len(files)
            selected_index = inner_seed % num_files
            selected_file = files[selected_index]
            # --- End Selection ---

            print(f"[OK] Seed {seed} -> Folder '{selected_folder.name}' File {selected_index + 1}/{num_files}: '{selected_file.name}'")

            # --- Load and Process Audio (pure PyAV, like core LoadAudio) ---
            waveform, sample_rate = _load_audio_file(selected_file)

            # Apply start offset
            if start_offset_seconds > 0:
                offset_samples = int(start_offset_seconds * sample_rate)
                if offset_samples < waveform.shape[1]:
                    waveform = waveform[:, offset_samples:]
                else:
                    print("[WARN] Warning: Start offset is beyond the audio duration. Returning no audio.")
                    return safe_return

            # Apply max length
            if max_length_seconds > 0:
                max_samples = int(max_length_seconds * sample_rate)
                if waveform.shape[1] > max_samples:
                    waveform = waveform[:, :max_samples]

            # Apply gain
            if gain_db != 0.0:
                gain_multiplier = 10 ** (gain_db / 20.0)
                waveform = waveform * gain_multiplier
                waveform = torch.clamp(waveform, -1.0, 1.0)

            # --- Format for ComfyUI ---
            waveform = waveform.unsqueeze(0)

            audio_out = {"waveform": waveform, "sample_rate": sample_rate}

            file_name = selected_file.stem if remove_extension else selected_file.name
            file_path_str = str(selected_file.parent.resolve())

            return (audio_out, file_name, file_path_str)

        except Exception as e:
            print(f"[ERROR] An unexpected error occurred in AudioLoaderCrawl: {str(e)}")
            return safe_return

# Node mappings
NODE_CLASS_MAPPINGS = {"AudioLoaderCrawl": AudioLoaderCrawl}

NODE_DISPLAY_NAME_MAPPINGS = {"AudioLoaderCrawl": "Audio Loader Crawl (CRT)"}
