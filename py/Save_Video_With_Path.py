import os
import secrets
import torch
import cv2
import numpy as np
import folder_paths
import tempfile
import subprocess
import json
import wave


def _write_audio(audio, temp_dir, video_duration):
    """Write the ComfyUI AUDIO dict as a 16-bit PCM wav matching the video length."""
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])

    if waveform.ndim != 3:
        waveform = waveform.unsqueeze(0)
    data = waveform[0].cpu().numpy().astype(np.float32)

    if data.ndim == 1:
        data = data[None, :]

    target_samples = int(round(video_duration * sample_rate))
    if data.shape[-1] > target_samples:
        data = data[..., :target_samples]
    elif data.shape[-1] < target_samples:
        data = np.pad(data, ((0, 0), (0, target_samples - data.shape[-1])))

    pcm = np.clip(data, -1.0, 1.0)
    pcm = (pcm.T * 32767.0).astype(np.int16)

    wav_path = os.path.join(temp_dir, "audio.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return wav_path


class SaveVideoWithPath:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        return {
            "required": {
                "image": ("IMAGE",),
                "folder_path": ("STRING", {"default": output_dir}),
                "subfolder_name": ("STRING", {"default": "videos"}),
                "filename": ("STRING", {"default": "output"}),
                "suffix": ("STRING", {"default": "", "tooltip": "Optional suffix appended to filename."}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 120}),
                "frames_limit": ("INT", {"default": -1, "min": -1, "max": 10000}),
                "activate": ("BOOLEAN", {"default": True}),
                "overwrite": ("BOOLEAN", {"default": False, "tooltip": "Overwrite an existing file with the same name. If off, a _1, _2, ... suffix is added."}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    CATEGORY = "CRT/Save"

    def save_video(
        self,
        image,
        folder_path,
        subfolder_name,
        filename,
        suffix,
        fps,
        frames_limit,
        activate,
        overwrite,
        audio=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        if not activate:
            print("[INFO] SaveVideoWithPath is deactivated. Skipping video save.")
            return ()

        if image is None:
            print("[ERROR] ERROR: No input image provided to SaveVideoWithPath.")
            return ()

        try:
            subfolder_clean = subfolder_name.strip().lstrip('/\\')
            filename_clean = filename.strip().lstrip('/\\')
            suffix_clean = suffix.strip()

            # Empty filename: auto-generate a unique name (never block the save).
            if not filename_clean:
                filename_clean = f"video_{secrets.token_hex(16)}"
                print(f"[INFO] No filename given. Using auto-generated name: {filename_clean}")

            # Empty subfolder: save directly into the base folder.
            final_dir = (
                os.path.join(folder_path, subfolder_clean)
                if subfolder_clean
                else folder_path
            )
            os.makedirs(final_dir, exist_ok=True)
            final_filepath = os.path.join(final_dir, filename_clean + suffix_clean + ".mp4")

            if not overwrite:
                counter = 1
                base = filename_clean + suffix_clean
                while os.path.exists(final_filepath):
                    final_filepath = os.path.join(final_dir, f"{base}_{counter}.mp4")
                    counter += 1

            if not os.access(final_dir, os.W_OK):
                raise IOError(f"Error: No write permissions for directory {final_dir}")

            frames = (image.cpu().numpy() * 255).astype(np.uint8)

            if frames.ndim == 3:
                frames = np.expand_dims(frames, axis=0)

            if frames_limit != -1 and len(frames) > frames_limit:
                frames = frames[:frames_limit]

            with tempfile.TemporaryDirectory() as temp_dir:
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    os.path.join(temp_dir, "frame_%06d.png"),
                ]

                audio_path = None
                if audio is not None:
                    audio_path = _write_audio(audio, temp_dir, len(frames) / fps)
                    ffmpeg_cmd.extend(["-i", audio_path])

                metadata_str = ""
                video_metadata = {}
                if prompt is not None:
                    video_metadata["prompt"] = prompt
                if extra_pnginfo is not None:
                    video_metadata.update(extra_pnginfo)

                if video_metadata:
                    metadata_str = json.dumps(video_metadata)
                    metadata_file = os.path.join(temp_dir, "metadata.txt")

                    metadata_str = metadata_str.replace("\\", "\\\\")
                    metadata_str = metadata_str.replace(";", "\\;")
                    metadata_str = metadata_str.replace("#", "\\#")
                    metadata_str = metadata_str.replace("=", "\\=")
                    metadata_str = metadata_str.replace("\n", "\\\n")

                    with open(metadata_file, "w", encoding="utf-8") as f:
                        f.write(";FFMETADATA1\n")
                        f.write(f"comment={metadata_str}")

                    ffmpeg_cmd.extend(["-i", metadata_file])

                output_options = [
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "3",
                    "-preset",
                    "fast",
                ]

                if audio is not None:
                    output_options.extend(["-c:a", "aac", "-b:a", "192k"])

                if video_metadata:
                    metadata_index = 2 if audio_path is not None else 1
                    output_options.extend(["-map", "0:v", "-map_metadata", str(metadata_index)])
                if audio_path is not None:
                    output_options.extend(["-map", "1:a"])

                output_options.append(final_filepath)
                ffmpeg_cmd.extend(output_options)

                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, encoding="utf-8")
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")

            print(f"[OK] Video saved successfully to: {final_filepath}")
            return ({"ui": {"text": ["Video saved (Lossless)."]}},)

        except Exception as e:
            print(f"[ERROR] ERROR in SaveVideoWithPath: {str(e)}")
            raise e
