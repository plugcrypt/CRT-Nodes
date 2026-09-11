from concurrent.futures import Future, ThreadPoolExecutor
import os
import re
import threading
from pathlib import Path

import torch
import torchaudio.functional as TAF

from .Audio_Loader_Crawl import _load_audio_file
from ._crawl_common import scan_tree, to_int


VALID_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")


class CRT_AudioLoaderCrawlBatch:
    def __init__(self):
        self.cache = {}
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="crt-audio-prefetch",
        )
        self._prefetch_future: Future | None = None
        self._prefetch_key = None
        self._prefetch_lock = threading.Lock()

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def natural_sort_key(path):
        return [
            int(token) if token.isdigit() else token.lower()
            for token in re.split(r"([0-9]+)", path.name)
        ]

    @staticmethod
    def _process_waveform(
        waveform,
        sample_rate,
        target_sample_rate,
        start_offset_seconds,
        max_length_seconds,
        gain_db,
    ):
        if start_offset_seconds > 0:
            offset_samples = int(start_offset_seconds * sample_rate)
            if offset_samples >= waveform.shape[1]:
                raise ValueError("Start offset is beyond the audio duration.")
            waveform = waveform[:, offset_samples:]

        if max_length_seconds > 0:
            max_samples = int(max_length_seconds * sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

        if gain_db != 0.0:
            gain_multiplier = 10 ** (gain_db / 20.0)
            waveform = torch.clamp(waveform * gain_multiplier, -1.0, 1.0)

        if sample_rate != target_sample_rate:
            waveform = TAF.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=target_sample_rate,
            )

        return waveform

    def _prepare_batch(
        self,
        files,
        selected_indices,
        target_sample_rate,
        start_offset_seconds,
        max_length_seconds,
        gain_db,
    ):
        waveforms = []
        errors = []

        for index in selected_indices:
            path = files[index]
            try:
                waveform, sample_rate = _load_audio_file(path)
                waveform = self._process_waveform(
                    waveform,
                    sample_rate,
                    target_sample_rate,
                    start_offset_seconds,
                    max_length_seconds,
                    gain_db,
                )
                waveforms.append(waveform)
                errors.append(None)
            except Exception as exc:
                # Keep batch positions aligned with the file window; the error
                # is reported through file_names.
                waveforms.append(
                    torch.zeros(1, target_sample_rate, dtype=torch.float32)
                )
                errors.append(str(exc))

        max_channels = max(waveform.shape[0] for waveform in waveforms)
        max_length = max(waveform.shape[1] for waveform in waveforms)

        normalized = []
        for waveform in waveforms:
            if waveform.shape[0] < max_channels:
                waveform = waveform.repeat(max_channels, 1)
            if waveform.shape[1] < max_length:
                waveform = torch.nn.functional.pad(
                    waveform, (0, max_length - waveform.shape[1])
                )
            normalized.append(waveform)

        batch = torch.stack(normalized, dim=0)
        return batch, errors

    @staticmethod
    def _batch_key(files, selected_indices, *options):
        return (
            tuple(str(files[index]) for index in selected_indices),
            tuple(float(option) for option in options),
        )

    def _consume_prefetch_or_load(self, key, files, selected_indices, *options):
        with self._prefetch_lock:
            if (
                self._prefetch_key == key
                and self._prefetch_future is not None
            ):
                future = self._prefetch_future
                self._prefetch_key = None
                self._prefetch_future = None
            else:
                future = None

        if future is not None:
            try:
                return future.result()
            except Exception as exc:
                print(
                    "[CRT Audio Loader Crawl Batch] "
                    f"Prefetch fallback: {exc}"
                )

        return self._prepare_batch(files, selected_indices, *options)

    def _schedule_prefetch(self, key, files, selected_indices, *options):
        with self._prefetch_lock:
            if self._prefetch_future is not None:
                self._prefetch_future.cancel()

            self._prefetch_key = key
            self._prefetch_future = self._prefetch_executor.submit(
                self._prepare_batch,
                files,
                selected_indices,
                *options,
            )

    def _cancel_prefetch(self):
        with self._prefetch_lock:
            if self._prefetch_future is not None:
                self._prefetch_future.cancel()
            self._prefetch_key = None
            self._prefetch_future = None

    def __del__(self):
        try:
            self._prefetch_executor.shutdown(
                wait=False,
                cancel_futures=True,
            )
        except Exception:
            pass

    # -- Node definition -------------------------------------------------------

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": (
                    "STRING",
                    {"default": "", "tooltip": "Path to the folder containing audio files"},
                ),
                "batch_count": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 64,
                        "tooltip": (
                            "Number of audio files to load. Window starts at "
                            "seed × batch_count and wraps around the folder. "
                            "Batches are zero-padded to the longest file."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": (
                            "Selects the starting file: "
                            "index = (seed × batch_count) % total_files."
                        ),
                    },
                ),
                "file_extension": (
                    ["all", "wav", "mp3", "flac", "ogg"],
                    {"default": "all", "tooltip": "File extension to filter for. 'all' includes wav, mp3, flac and ogg."},
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
                "remove_extension": ("BOOLEAN", {"default": False}),
                "sample_rate": (
                    "INT",
                    {
                        "default": 44100,
                        "min": 8000,
                        "max": 192000,
                        "tooltip": "All files are resampled to this rate so they can be stacked into one batch.",
                    },
                ),
                "max_length_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "step": 0.1,
                        "tooltip": "Maximum length of each audio in seconds (0 for no limit)",
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
                "print_index": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Print each selected file index and name to the console.",
                    },
                ),
            },
            "optional": {
                "subfolder_seed": (
                    "INT",
                    {
                        "forceInput": True,
                        "tooltip": "Restricts the batch window to the folder selected by seed % number_of_folders. Leave unconnected to batch across the whole tree.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("audio", "file_names", "file_paths", "batch_count", "total_files")
    OUTPUT_IS_LIST = (False, True, True, False, False)
    FUNCTION = "load_batch"
    CATEGORY = "CRT/Load"
    DESCRIPTION = (
        "Loads a window of audio files as one padded batch. file_names and "
        "file_paths are per-file lists that pair item by item with batched "
        "transcripts in SaveTextWithPath."
    )

    # -- Main ------------------------------------------------------------------

    def load_batch(
        self,
        folder_path,
        batch_count,
        seed,
        file_extension,
        max_depth,
        remove_extension,
        sample_rate,
        max_length_seconds,
        start_offset_seconds,
        gain_db,
        print_index,
        subfolder_seed=None,
    ):
        tag = "[CRT Audio Loader Crawl Batch]"

        def blank():
            return {
                "waveform": torch.zeros(1, 2, 1, dtype=torch.float32),
                "sample_rate": int(sample_rate),
            }

        if not folder_path or not folder_path.strip():
            print(f"{tag} ERROR: Folder path is empty.")
            return (blank(), ["Error: folder path is empty"], [""], 0, 0)

        folder = Path(folder_path.strip()).expanduser()
        if not folder.is_dir():
            print(f"{tag} ERROR: Folder '{folder}' not found.")
            return (blank(), ["Error: folder not found"], [""], 0, 0)
        folder = folder.resolve()

        # -- File-list cache ---------------------------------------------------
        # Two-tier invalidation so downstream stages writing non-audio files
        # next to the sources (e.g. .txt transcripts) don't force a rescan
        # every run: an unchanged mtime reuses the cache with no directory
        # walk, and a changed mtime only triggers a full rescan when the
        # number of audio files actually changed.
        raw_ext = file_extension.strip().lower()
        is_all = raw_ext == "all"
        if is_all:
            extension = "all"
        else:
            extension = raw_ext
            if not extension.startswith("."):
                extension = f".{extension}"
        current_mtime = folder.stat().st_mtime_ns
        seed = to_int(seed)
        max_depth = to_int(max_depth)
        sub_seed = to_int(subfolder_seed) if subfolder_seed is not None else None
        cache_key = f"{folder}_{max_depth}_{extension}_s{sub_seed if sub_seed is not None else 'x'}"

        def _tree_files():
            folders, files_by_folder = scan_tree(folder, max_depth)
            if sub_seed is not None:
                if not folders:
                    return []
                sel_folder = folders[sub_seed % len(folders)]
                return files_by_folder.get(os.path.normpath(str(sel_folder)), [])
            return [p for fl in files_by_folder.values() for p in fl]

        def _collect_files():
            tree_files = _tree_files()
            if is_all:
                seen = set()
                return [p for p in tree_files if p.suffix.lower() in VALID_EXTENSIONS and not (p in seen or seen.add(p))]
            return [p for p in tree_files if p.suffix.lower() == extension]

        def _count_files():
            return len(_collect_files())

        cached = self.cache.get(cache_key)
        need_rescan = True
        if cached is not None and cached.get("mtime") == current_mtime:
            need_rescan = False
        elif cached is not None:
            try:
                current_count = _count_files()
            except Exception:
                current_count = -1
            if current_count == len(cached["files"]):
                need_rescan = False
                cached["mtime"] = current_mtime

        if need_rescan:
            ext_label = "audio" if is_all else f"'{extension}'"
            print(f"{tag} Scanning '{folder}' for {ext_label} files...")
            try:
                files = sorted(_collect_files(), key=self.natural_sort_key)
                self.cache[cache_key] = {
                    "files": files,
                    "mtime": current_mtime,
                }
                self._cancel_prefetch()
                print(f"{tag} Found {len(files)} files.")
            except Exception as exc:
                print(f"{tag} ERROR scanning: {exc}")
                self.cache.pop(cache_key, None)
                self._cancel_prefetch()
                return (blank(), [f"Error: {exc}"], [""], 0, 0)

        files = self.cache[cache_key]["files"]
        total = len(files)

        if total == 0:
            ext_label = "audio" if is_all else f"'{extension}'"
            print(f"{tag} No {ext_label} files found in '{folder}'.")
            return (blank(), ["No audio files found"], [""], 0, 0)

        # -- Select and load batch ---------------------------------------------
        batch_count = min(batch_count, total)
        start = (seed * batch_count) % total
        selected_indices = [
            (start + index) % total
            for index in range(batch_count)
        ]

        options = (
            int(sample_rate),
            float(start_offset_seconds),
            float(max_length_seconds),
            float(gain_db),
        )
        current_key = self._batch_key(files, selected_indices, *options)
        batch, errors = self._consume_prefetch_or_load(
            current_key,
            files,
            selected_indices,
            *options,
        )

        names = []
        paths = []
        for position, index in enumerate(selected_indices):
            path = files[index]
            error = errors[position]
            if error is None:
                name = path.stem if remove_extension else path.name
                if print_index:
                    print(f"{tag} [{index + 1}/{total}] {name}")
            else:
                print(f"{tag} ERROR loading '{path}': {error}")
                name = f"Error: {path.name}"

            names.append(name)
            # file_paths is the containing directory, one entry per file, so
            # downstream SaveTextWithPath can save each transcript next to its
            # source file even across subfolders.
            paths.append(str(path.parent))

        audio_out = {
            "waveform": batch,
            "sample_rate": int(sample_rate),
        }

        # Decode the next window while this one is being transcribed.
        next_start = ((seed + 1) * batch_count) % total
        next_indices = [
            (next_start + index) % total
            for index in range(batch_count)
        ]
        next_key = self._batch_key(files, next_indices, *options)
        self._schedule_prefetch(
            next_key,
            files,
            next_indices,
            *options,
        )

        return (
            audio_out,
            names,
            paths,
            batch.shape[0],
            total,
        )


NODE_CLASS_MAPPINGS = {
    "CRT_AudioLoaderCrawlBatch": CRT_AudioLoaderCrawlBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_AudioLoaderCrawlBatch": "Audio Loader Crawl Batch (CRT)",
}
