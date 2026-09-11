import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from ._crawl_common import pick_folder, scan_tree, to_int


class ImageLoaderCrawl:
    def __init__(self):
        # Instance-level cache to store file lists and folder modification times.
        self.cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
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
                "remove_extension": ("BOOLEAN", {"default": False}),
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

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("image_output", "file_name", "file_path", "total_images", "sidecar_txt_prompt")
    FUNCTION = "load_image_incrementally"
    CATEGORY = "CRT/Load"

    @staticmethod
    def _load_sidecar_txt(image_path):
        """Return the contents of the .txt file sharing the image's name,
        or a placeholder message when no sidecar file exists."""
        sidecar = image_path.with_suffix(".txt")
        if not sidecar.is_file():
            return "No corresponding text file found"
        try:
            return sidecar.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:
            print(f"[ERROR] Error reading sidecar '{sidecar}': {str(e)}")
            return "No corresponding text file found"

    def load_image_incrementally(self, folder_path, seed, max_depth, remove_extension, subfolder_seed=None):
        # Create a blank image as fallback
        def create_blank_image():
            blank = np.zeros((512, 512, 3), dtype=np.float32)
            return torch.from_numpy(blank)[None,]

        if not folder_path or not folder_path.strip():
            return (create_blank_image(), "Error: Folder path is empty", "", 0, "")

        folder = Path(folder_path.strip())
        if not folder.is_dir():
            print(f"[ERROR] Error: Folder '{folder}' not found.")
            return (create_blank_image(), "Error: Folder not found", "", 0, "")

        seed = to_int(seed)
        max_depth = to_int(max_depth)
        sub_seed = to_int(subfolder_seed) if subfolder_seed is not None else None

        # --- Smart Caching Logic ---
        cache_key = (str(folder.resolve()), max_depth)
        current_mtime = folder.stat().st_mtime

        try:
            if cache_key not in self.cache or self.cache[cache_key]['mtime'] != current_mtime:
                print(f"[INFO] Folder changed or not cached. Scanning '{folder}' (depth {max_depth})...")
                folders, files_by_folder = scan_tree(folder, max_depth)
                self.cache[cache_key] = {'folders': folders, 'files': files_by_folder, 'mtime': current_mtime}
                print(f"[OK] Cached folder tree from '{folder}'")
            folders = self.cache[cache_key]['folders']
            files_by_folder = self.cache[cache_key]['files']
        except Exception as e:
            print(f"[ERROR] Error accessing folder '{folder}': {str(e)}")
            if cache_key in self.cache:
                del self.cache[cache_key]
            return (create_blank_image(), "Error accessing folder", "", 0, "")

        selected_folder, inner_seed = pick_folder(folders, sub_seed, seed)
        if selected_folder is None:
            print("[ERROR] No folders found under the given path.")
            return (create_blank_image(), "No folders found", "", 0, "")

        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.ti', '.gi', '.webp'}
        files = [p for p in files_by_folder.get(os.path.normpath(str(selected_folder)), []) if p.suffix.lower() in valid_extensions]
        files.sort()

        if not files:
            print(f"[ERROR] Warning: No valid image files found in '{selected_folder}'.")
            return (create_blank_image(), "No images found", "", 0, "")

        num_files = len(files)
        selected_index = inner_seed % num_files
        selected_file = files[selected_index]

        try:
            with Image.open(selected_file) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array)[None,]

            base_name = selected_file.stem if remove_extension else selected_file.name
            sidecar_txt = self._load_sidecar_txt(selected_file)
            print(f"[OK] Seed {seed} -> Folder '{selected_folder.name}' Image {selected_index + 1}/{num_files}: '{base_name}'")

            return (img_tensor, base_name, str(selected_file.parent.resolve()), num_files, sidecar_txt)
        # Self-healing: If a file is in the cache but was deleted just before loading, this will catch it.
        except FileNotFoundError:
            print(
                f"[ERROR] Error: File '{selected_file}' was in cache but not found on disk. Invalidating cache for next run."
            )
            # Forcing a rescan on the next execution by removing the invalid cache entry.
            if cache_key in self.cache:
                del self.cache[cache_key]
            return (create_blank_image(), "Error: Cached file not found", "", 0, "")
        except Exception as e:
            print(f"[ERROR] Error loading image '{selected_file}': {str(e)}")
            return (create_blank_image(), "Error loading image", "", 0, "")


# Node mappings
NODE_CLASS_MAPPINGS = {"ImageLoaderCrawl": ImageLoaderCrawl}

NODE_DISPLAY_NAME_MAPPINGS = {"ImageLoaderCrawl": "Image Loader Crawl (Smart)"}
