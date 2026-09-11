from concurrent.futures import Future, ThreadPoolExecutor
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from ._crawl_common import scan_tree, to_int


VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".webp",
}


class CRT_ImageLoaderCrawlBatch:
    def __init__(self):
        self.cache = {}
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="crt-image-prefetch",
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
    def _target_dimensions(width, height, megapixels, quantize=8):
        scale = (megapixels * 1_000_000 / (height * width)) ** 0.5
        target_height = max(
            quantize,
            round(height * scale / quantize) * quantize,
        )
        target_width = max(
            quantize,
            round(width * scale / quantize) * quantize,
        )
        return target_width, target_height

    @staticmethod
    def _load_sidecar_txt(path, tag):
        sidecar = path.with_suffix(".txt")
        if not sidecar.is_file():
            print(f"{tag} No sidecar .txt found for '{path.name}'.")
            return ""
        try:
            return sidecar.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as exc:
            print(f"{tag} ERROR reading sidecar '{sidecar}': {exc}")
            return ""

    @staticmethod
    def _decode_rgb(path):
        with Image.open(path) as opened:
            transposed = ImageOps.exif_transpose(opened)
            rgb = transposed.convert("RGB")
            rgb.load()
            return rgb

    @staticmethod
    def _resize(image, width, height):
        if image.size == (width, height):
            return image
        return image.resize(
            (width, height),
            resample=Image.Resampling.LANCZOS,
            reducing_gap=3.0,
        )

    @staticmethod
    def _to_tensor(image):
        # Convert only the final-sized uint8 image to float32. This avoids
        # allocating a full-resolution float32 NumPy array before resizing.
        array = np.array(image, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array).to(dtype=torch.float32)
        tensor.mul_(1.0 / 255.0)
        return tensor.unsqueeze(0)

    @staticmethod
    def _pad_without_resizing(image, target_width, target_height):
        if image.size == (target_width, target_height):
            return image
        canvas = Image.new("RGB", (target_width, target_height))
        left = (target_width - image.width) // 2
        top = (target_height - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas

    def _prepare_batch(self, files, selected_indices, megapixels):
        images = []
        errors = []

        for index in selected_indices:
            path = files[index]
            try:
                image = self._decode_rgb(path)
                if megapixels > 0:
                    width, height = self._target_dimensions(
                        image.width,
                        image.height,
                        megapixels,
                    )
                    image = self._resize(image, width, height)
                images.append(image)
                errors.append(None)
            except Exception as exc:
                images.append(Image.new("RGB", (64, 64)))
                errors.append(str(exc))

        shapes = {(image.height, image.width) for image in images}
        mixed_shapes = len(shapes) > 1

        if mixed_shapes and megapixels == 0:
            # ComfyUI IMAGE batches require a common H/W. Preserve every source
            # pixel and center-pad smaller images instead of resampling them.
            target_width = max(image.width for image in images)
            target_height = max(image.height for image in images)
            images = [
                self._pad_without_resizing(
                    image,
                    target_width,
                    target_height,
                )
                for image in images
            ]
        elif mixed_shapes:
            aspect_ratios = [
                image.width / image.height
                for image in images
            ]
            average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
            target_height = max(
                8,
                round(
                    (
                        megapixels
                        * 1_000_000
                        / average_aspect_ratio
                    )
                    ** 0.5
                    / 8
                )
                * 8,
            )
            target_width = max(
                8,
                round(
                    target_height
                    * average_aspect_ratio
                    / 8
                )
                * 8,
            )

            unified = []
            for image in images:
                current_aspect_ratio = image.width / image.height
                if current_aspect_ratio > average_aspect_ratio:
                    crop_width = max(
                        1,
                        min(
                            round(image.height * average_aspect_ratio),
                            image.width,
                        ),
                    )
                    left = (image.width - crop_width) // 2
                    image = image.crop(
                        (left, 0, left + crop_width, image.height)
                    )
                elif current_aspect_ratio < average_aspect_ratio:
                    crop_height = max(
                        1,
                        min(
                            round(image.width / average_aspect_ratio),
                            image.height,
                        ),
                    )
                    top = (image.height - crop_height) // 2
                    image = image.crop(
                        (0, top, image.width, top + crop_height)
                    )

                unified.append(
                    self._resize(
                        image,
                        target_width,
                        target_height,
                    )
                )
            images = unified

        tensors = [self._to_tensor(image) for image in images]
        return tensors, errors, mixed_shapes

    @staticmethod
    def _batch_key(files, selected_indices, megapixels):
        return (
            tuple(str(files[index]) for index in selected_indices),
            float(megapixels),
        )

    def _consume_prefetch_or_load(
        self,
        key,
        files,
        selected_indices,
        megapixels,
    ):
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
                    "[CRT Image Loader Crawl Batch] "
                    f"Prefetch fallback: {exc}"
                )

        return self._prepare_batch(
            files,
            selected_indices,
            megapixels,
        )

    def _schedule_prefetch(
        self,
        key,
        files,
        selected_indices,
        megapixels,
    ):
        with self._prefetch_lock:
            if self._prefetch_future is not None:
                self._prefetch_future.cancel()

            self._prefetch_key = key
            self._prefetch_future = self._prefetch_executor.submit(
                self._prepare_batch,
                files,
                selected_indices,
                megapixels,
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
                "folder_path": ("STRING", {"default": ""}),
                "batch_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                        "tooltip": (
                            "Number of images to load. Window starts at "
                            "seed × batch_count and wraps around the folder."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": (
                            "Selects the starting image: "
                            "index = (seed × batch_count) % total_images."
                        ),
                    },
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 16.0,
                        "step": 0.05,
                        "tooltip": (
                            "Target resolution in megapixels. Use 0 to keep "
                            "the source resolution (no resize). Mixed-size "
                            "batches are center-padded to the largest image."
                        ),
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
                "remove_extension": ("BOOLEAN", {"default": False}),
                "print_index": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Print each selected file index and name "
                            "to the console."
                        ),
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

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "images",
        "file_names",
        "file_paths",
        "batch_count",
        "total_images",
        "sidecar_txt_prompt",
    )
    FUNCTION = "load_batch"
    CATEGORY = "CRT/Load"

    @classmethod
    def IS_CHANGED(cls, folder_path, batch_count, seed, megapixels, max_depth,
                   remove_extension, print_index, **kwargs):
        # Nodes without input links are cache-skipped while their widget values
        # are unchanged, so images swapped inside the folder would never reload
        # without checking the selected files' external state here.
        try:
            folder = Path(str(folder_path).strip()).expanduser()
            if not folder.is_dir():
                return 0
            folder = folder.resolve()
            _, files_by_folder = scan_tree(folder, to_int(max_depth))
            files = [p for fl in files_by_folder.values() for p in fl]
            files = [p for p in files if p.suffix.lower() in VALID_EXTENSIONS]
            files = sorted(files, key=cls.natural_sort_key)
            total = len(files)
            if total == 0:
                return 0
            start = (int(seed) * int(batch_count)) % total
            state = [folder.stat().st_mtime_ns]
            for index in range(int(batch_count)):
                stat = files[(start + index) % total].stat()
                state.append(stat.st_mtime_ns)
                state.append(stat.st_size)
            return hash(tuple(state))
        except Exception:
            return time.time()

    # -- Main ------------------------------------------------------------------

    def load_batch(
        self,
        folder_path,
        batch_count,
        seed,
        megapixels,
        max_depth,
        remove_extension,
        print_index,
        subfolder_seed=None,
        **kwargs,
    ):
        tag = "[CRT Image Loader Crawl Batch]"
        megapixels = float(megapixels)

        def blank():
            return torch.zeros(1, 64, 64, 3, dtype=torch.float32)

        if not folder_path or not folder_path.strip():
            return (blank(), "Error: folder path is empty", "", 0, 0, "")

        folder = Path(folder_path.strip()).expanduser()
        if not folder.is_dir():
            print(f"{tag} ERROR: Folder '{folder}' not found.")
            return (blank(), "Error: folder not found", "", 0, 0, "")
        folder = folder.resolve()

        # -- File-list cache ---------------------------------------------------
        seed = to_int(seed)
        max_depth = to_int(max_depth)
        sub_seed = to_int(subfolder_seed) if subfolder_seed is not None else None
        cache_key = (str(folder), max_depth, sub_seed)
        current_mtime = folder.stat().st_mtime_ns

        if (
            cache_key not in self.cache
            or self.cache[cache_key]["mtime"] != current_mtime
        ):
            print(f"{tag} Scanning '{folder}' (depth {max_depth})...")
            try:
                folders, files_by_folder = scan_tree(folder, max_depth)
                tree_files = [p for fl in files_by_folder.values() for p in fl]
                if sub_seed is not None:
                    if not folders:
                        raise ValueError("No folders found under the given path")
                    sel_folder = folders[sub_seed % len(folders)]
                    tree_files = files_by_folder.get(os.path.normpath(str(sel_folder)), [])
                files = sorted(
                    (
                        path
                        for path in tree_files
                        if path.suffix.lower() in VALID_EXTENSIONS
                    ),
                    key=self.natural_sort_key,
                )
                self.cache[cache_key] = {
                    "files": files,
                    "mtime": current_mtime,
                }
                self._cancel_prefetch()
                print(f"{tag} Found {len(files)} images.")
            except Exception as exc:
                print(f"{tag} ERROR scanning: {exc}")
                self.cache.pop(cache_key, None)
                self._cancel_prefetch()
                return (blank(), f"Error: {exc}", "", 0, 0, "")

        files = self.cache[cache_key]["files"]
        total = len(files)

        if total == 0:
            print(f"{tag} No images found in '{folder}'.")
            return (blank(), "No images found", "", 0, 0, "")

        # -- Select and load batch ---------------------------------------------
        start = (seed * batch_count) % total
        selected_indices = [
            (start + index) % total
            for index in range(batch_count)
        ]
        current_key = self._batch_key(
            files,
            selected_indices,
            megapixels,
        )
        tensors, errors, mixed_shapes = self._consume_prefetch_or_load(
            current_key,
            files,
            selected_indices,
            megapixels,
        )

        if mixed_shapes:
            if megapixels == 0:
                print(
                    f"{tag} Mixed resolutions detected - "
                    "center-padding without resampling."
                )
            else:
                print(
                    f"{tag} Mixed resolutions detected - "
                    "center-cropping to a uniform size."
                )

        names = []
        paths = []
        prompts = []
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
            # file_paths is the containing directory. The filename is exposed
            # separately through file_names.
            paths.append(str(path.parent))
            prompts.append(self._load_sidecar_txt(path, tag))

        batch = torch.cat(tensors, dim=0)

        # Predict the next sequential seed. During the rest of the workflow,
        # its decode/resize can overlap GPU inference. Only one future batch is
        # retained, so memory use stays bounded.
        next_start = ((seed + 1) * batch_count) % total
        next_indices = [
            (next_start + index) % total
            for index in range(batch_count)
        ]
        next_key = self._batch_key(
            files,
            next_indices,
            megapixels,
        )
        self._schedule_prefetch(
            next_key,
            files,
            next_indices,
            megapixels,
        )

        return (
            batch,
            "\n".join(names),
            "\n".join(paths),
            len(tensors),
            total,
            "\n".join(prompts),
        )


NODE_CLASS_MAPPINGS = {
    "CRT_ImageLoaderCrawlBatch": CRT_ImageLoaderCrawlBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_ImageLoaderCrawlBatch": "Image Loader Crawl Batch (CRT)",
}
