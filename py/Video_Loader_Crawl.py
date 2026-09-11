import math
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ._crawl_common import pick_folder, scan_tree, to_int


VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


def _silent_audio(sample_rate=44100):
    return {"waveform": torch.zeros(1, 1, 1), "sample_rate": int(sample_rate)}


def _probe_audio_stream(file_path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels",
                "-of",
                "default=noprint_wrappers=1",
                str(file_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None

        values = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key.strip()] = value.strip()
        sample_rate = int(values.get("sample_rate", 0))
        channels = int(values.get("channels", 0))
        if sample_rate > 0 and channels > 0:
            return sample_rate, channels
    except (OSError, TypeError, ValueError):
        pass
    return None


def get_audio_from_video(file_path, max_duration=None):
    started = time.perf_counter()
    duration_label = (
        f" (first {max_duration:.2f}s)"
        if max_duration is not None and max_duration > 0
        else ""
    )
    print(
        f"[CRT Video Loader] Extracting audio from '{Path(file_path).name}'"
        f"{duration_label}..."
    )
    audio_info = _probe_audio_stream(file_path)
    if audio_info is None:
        print("[CRT Video Loader] No readable audio stream; returning silence.")
        return _silent_audio()

    sample_rate, channels = audio_info
    try:
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(file_path),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
        ]
        if max_duration is not None and max_duration > 0:
            command.extend(["-t", f"{float(max_duration):.6f}"])
        command.extend(["-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0 and not result.stdout:
            error = result.stderr.decode("utf-8", errors="backslashreplace").strip()
            raise RuntimeError(error or f"ffmpeg exited with code {result.returncode}")

        raw_audio = bytearray(result.stdout)
        result = None
        sample_count = len(raw_audio) // np.dtype(np.float32).itemsize
        sample_count -= sample_count % channels
        if sample_count <= 0:
            return _silent_audio(sample_rate)
        del raw_audio[sample_count * np.dtype(np.float32).itemsize :]

        audio = torch.frombuffer(raw_audio, dtype=torch.float32, count=sample_count)
        audio = audio.reshape((-1, channels)).transpose(0, 1).unsqueeze(0)
        elapsed = time.perf_counter() - started
        duration = sample_count / float(channels * sample_rate)
        print(
            f"[CRT Video Loader] Audio ready: {duration:.2f}s, "
            f"{sample_rate} Hz, {channels} channel(s) in {elapsed:.2f}s"
        )
        return {"waveform": audio, "sample_rate": sample_rate}
    except Exception as error:
        print(f"[CRT Video Loader] Warning: could not extract audio: {error}")
        return _silent_audio(sample_rate)


class VideoLoaderCrawl:
    def __init__(self):
        # This cache intentionally contains only paths, never decoded tensors.
        self.cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": (
                    "STRING",
                    {"default": "C:\\videos", "tooltip": "Path to video files"},
                ),
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
                "max_depth": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1,
                        "max": 100,
                        "tooltip": "Subfolder crawl depth. 0 = only the root folder, 1 = one level deep, -1 = infinite.",
                    },
                ),
                "remove_extension": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Remove file extension from output name"},
                ),
                "frames_limit": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 10000,
                        "tooltip": "Maximum number of output frames. Use -1 to load the complete video.",
                    },
                ),
                "framerate": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 240.0,
                        "step": 0.1,
                        "tooltip": "-1 keeps the source rate. Lower values retain evenly spaced frames; values above the source rate keep the source rate.",
                    },
                ),
                "even_batch_picker": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 10000,
                        "step": 1,
                        "tooltip": "After Frames limit, retain this many frames evenly across that range using the same selection as Even Batch Picker (CRT). -1 or 0 keeps the complete limited range.",
                    },
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 16.0,
                        "step": 0.05,
                        "tooltip": "After even-frame selection, resize only the retained frames to this target area with Lanczos while preserving aspect ratio. Use 0 to keep the source resolution.",
                    },
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

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "FLOAT", "INT", "AUDIO")
    RETURN_NAMES = (
        "image_output",
        "file_name",
        "file_path",
        "framerate",
        "framerate_int",
        "audio",
    )
    FUNCTION = "load_video_file"
    CATEGORY = "CRT/Load"

    @staticmethod
    def _blank_output():
        blank_frame = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        return (
            blank_frame,
            "Error: See console for details",
            "",
            0.0,
            0,
            _silent_audio(),
        )

    @staticmethod
    def _even_positions(batch_size, count):
        batch_size = max(0, int(batch_size))
        count = int(count)
        if batch_size <= 1 or count <= 0 or count >= batch_size:
            return list(range(batch_size))
        if count == 1:
            return [0]
        return [
            int((index * (batch_size - 1)) / (count - 1))
            for index in range(count)
        ]

    @staticmethod
    def _target_dimensions(width, height, megapixels, quantize=8):
        width = max(1, int(width))
        height = max(1, int(height))
        megapixels = float(megapixels)
        if megapixels <= 0:
            return width, height
        megapixels = max(0.001, megapixels)
        scale = math.sqrt(megapixels * 1_000_000.0 / (width * height))
        target_width = max(quantize, round(width * scale / quantize) * quantize)
        target_height = max(quantize, round(height * scale / quantize) * quantize)
        return int(target_width), int(target_height)

    @staticmethod
    def _print_decode_progress(source_index, source_total, retained, started, next_percent):
        elapsed = max(1e-6, time.perf_counter() - started)
        if source_total > 0:
            percent = min(100, int(source_index * 100 / source_total))
            if percent < next_percent:
                return next_percent
            speed = source_index / elapsed
            remaining = max(0, source_total - source_index)
            eta = remaining / speed if speed > 0 else 0.0
            print(
                f"[CRT Video Loader] Decoding {percent:3d}% - "
                f"{retained} frame(s) retained - {speed:.1f} source fps - ETA {eta:.1f}s"
            )
            return ((percent // 10) + 1) * 10

        if source_index % 250 == 0:
            print(
                f"[CRT Video Loader] Decoded {source_index} source frame(s), "
                f"retained {retained} - {source_index / elapsed:.1f} source fps"
            )
        return next_percent

    def load_video_file(
        self,
        folder_path,
        seed,
        max_depth,
        remove_extension,
        frames_limit,
        framerate,
        even_batch_picker,
        megapixels,
        subfolder_seed=None,
    ):
        if not folder_path or not folder_path.strip():
            print("[CRT Video Loader] Error: folder path is empty.")
            return self._blank_output()

        folder = Path(folder_path.strip())
        if not folder.is_dir():
            print(f"[CRT Video Loader] Error: folder '{folder}' was not found.")
            return self._blank_output()

        seed = to_int(seed)
        max_depth = to_int(max_depth)
        sub_seed = to_int(subfolder_seed) if subfolder_seed is not None else None

        cache_key = (str(folder.resolve()), max_depth)
        current_mtime = folder.stat().st_mtime

        try:
            if cache_key not in self.cache or self.cache[cache_key]["mtime"] != current_mtime:
                print(f"[CRT Video Loader] Scanning '{folder}' (depth {max_depth})...")
                folders, files_by_folder = scan_tree(folder, max_depth)
                self.cache[cache_key] = {"folders": folders, "files": files_by_folder, "mtime": current_mtime}
                print(f"[CRT Video Loader] Cached folder tree from '{folder}'.")
            folders = self.cache[cache_key]["folders"]
            files_by_folder = self.cache[cache_key]["files"]
        except Exception as error:
            print(f"[CRT Video Loader] Error accessing '{folder}': {error}")
            self.cache.pop(cache_key, None)
            return self._blank_output()

        selected_folder, inner_seed = pick_folder(folders, sub_seed, seed)
        if selected_folder is None:
            print("[CRT Video Loader] No folders found under the given path.")
            return self._blank_output()

        files = [p for p in files_by_folder.get(os.path.normpath(str(selected_folder)), []) if p.suffix.lower() in VIDEO_EXTENSIONS]
        files.sort()

        if not files:
            print(f"[CRT Video Loader] No supported video files found in '{selected_folder}'.")
            return self._blank_output()

        selected_index = inner_seed % len(files)
        selected_file = files[selected_index]
        print(
            f"[CRT Video Loader] Seed {seed} -> folder '{selected_folder.name}' video "
            f"{selected_index + 1}/{len(files)}: '{selected_file.name}'"
        )

        cap = None
        try:
            cap = cv2.VideoCapture(str(selected_file))
            if not cap.isOpened():
                print(f"[CRT Video Loader] Could not open '{selected_file}'.")
                return self._blank_output()

            source_fps = max(0.0, float(cap.get(cv2.CAP_PROP_FPS) or 0.0))
            source_total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            if framerate <= 0 or source_fps <= 0 or framerate >= source_fps:
                frame_step = 1
                output_fps = source_fps
            else:
                frame_step = max(1, int(round(source_fps / framerate)))
                output_fps = source_fps / frame_step

            if source_total <= 0:
                print(
                    "[CRT Video Loader] Container has no frame count; "
                    "counting frames once before selection..."
                )
                while cap.grab():
                    source_total += 1
                cap.release()
                cap = cv2.VideoCapture(str(selected_file))
                if not cap.isOpened():
                    raise RuntimeError("Could not reopen video after frame counting")

            candidate_total = int(math.ceil(source_total / float(frame_step)))
            limited_count = (
                candidate_total
                if frames_limit < 0
                else min(candidate_total, max(0, int(frames_limit)))
            )
            selected_positions = self._even_positions(
                limited_count, even_batch_picker
            )
            if not selected_positions:
                raise RuntimeError(
                    "Frames limit / Even batch picker selected zero frames"
                )
            selected_source_indices = [
                position * frame_step for position in selected_positions
            ]
            selected_total = len(selected_source_indices)
            decode_total = selected_source_indices[-1] + 1

            print(
                f"[CRT Video Loader] Stage 1/3 - Frames limit: "
                f"{limited_count}/{candidate_total} frame(s) at {output_fps:.3f} fps"
            )
            print(
                f"[CRT Video Loader] Stage 2/3 - Even batch picker: "
                f"{selected_total}/{limited_count} frame(s) retained"
            )
            if float(megapixels) > 0:
                resize_label = f"{float(megapixels):.3f} MP"
            else:
                resize_label = "disabled (source resolution)"
            print(
                f"[CRT Video Loader] Stage 3/3 - Lanczos resize target: {resize_label}"
            )

            frame_buffer = None
            retained = 0
            source_index = 0
            selected_cursor = 0
            next_percent = 10
            started = time.perf_counter()

            while cap.isOpened() and selected_cursor < selected_total:
                if not cap.grab():
                    break
                current_source_index = source_index
                source_index += 1
                if current_source_index == selected_source_indices[selected_cursor]:
                    retrieved, frame = cap.retrieve()
                    if not retrieved or frame is None:
                        break

                    source_height, source_width = frame.shape[:2]
                    target_width, target_height = self._target_dimensions(
                        source_width,
                        source_height,
                        megapixels,
                    )
                    if (target_width, target_height) != (source_width, source_height):
                        frame = cv2.resize(
                            frame,
                            (target_width, target_height),
                            interpolation=cv2.INTER_LANCZOS4,
                        )

                    if frame_buffer is None:
                        frame_buffer = np.empty(
                            (selected_total, target_height, target_width, 3),
                            dtype=np.float32,
                        )
                    elif frame_buffer.shape[1:3] != (target_height, target_width):
                        raise RuntimeError("Video dimensions changed during decoding")

                    # Normalize only the selected, resized frame directly into
                    # the final ComfyUI batch. Full-resolution float frames are
                    # never allocated.
                    np.multiply(
                        frame[:, :, ::-1],
                        np.float32(1.0 / 255.0),
                        out=frame_buffer[retained],
                        casting="unsafe",
                    )
                    retained += 1
                    selected_cursor += 1

                next_percent = self._print_decode_progress(
                    source_index,
                    decode_total,
                    retained,
                    started,
                    next_percent,
                )

            if frame_buffer is None or retained == 0:
                raise RuntimeError("No selected frames could be decoded")
            if retained != selected_total:
                print(
                    f"[CRT Video Loader] Warning: requested {selected_total} selected "
                    f"frames but decoded {retained}; container frame metadata may be inaccurate."
                )

            video_tensor = torch.from_numpy(frame_buffer[:retained])
            elapsed = time.perf_counter() - started
            memory_gib = video_tensor.numel() * video_tensor.element_size() / (1024**3)
            print(
                f"[CRT Video Loader] Video ready: {retained} frame(s), "
                f"{video_tensor.shape[2]}x{video_tensor.shape[1]}, {memory_gib:.2f} GiB "
                f"in {elapsed:.2f}s"
            )

            file_name = selected_file.stem if remove_extension else selected_file.name
            loaded_duration = (
                limited_count / output_fps
                if frames_limit >= 0 and output_fps > 0
                else None
            )
            audio = get_audio_from_video(
                selected_file,
                max_duration=loaded_duration,
            )
            return (
                video_tensor,
                file_name,
                str(selected_file.parent.resolve()),
                float(output_fps),
                int(round(output_fps)),
                audio,
            )

        except FileNotFoundError:
            print(
                f"[CRT Video Loader] '{selected_file}' disappeared; "
                "invalidating the folder cache."
            )
            self.cache.pop(cache_key, None)
            return self._blank_output()
        except Exception as error:
            print(f"[CRT Video Loader] Error loading '{selected_file}': {error}")
            return self._blank_output()
        finally:
            if cap is not None:
                cap.release()
