import os
from pathlib import Path

from ._crawl_common import pick_folder, scan_tree, to_int


class TextLoaderCrawl:
    def __init__(self):
        # Instance-level cache of the folder tree, keyed by (resolved root, max_depth)
        # and invalidated when the root directory's mtime changes.
        self.cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to the folder containing text files"}),
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
                    "STRING",
                    {"default": ".txt", "tooltip": "File extension to filter (e.g., .txt, .md)"},
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
                "max_words": (
                    "INT",
                    {"default": 0, "min": 0, "tooltip": "Maximum number of words in output (0 for no limit)"},
                ),
                "remove_extension": ("BOOLEAN", {"default": False, "tooltip": "Output filename without extension"}),
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

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text_output", "file_name", "file_path")
    FUNCTION = "load_text_file"
    CATEGORY = "CRT/Load"

    def limit_words(self, text, max_words):
        """Limit the text to a specified number of words."""
        if max_words <= 0:
            return text
        words = text.split()
        return ' '.join(words[:max_words])

    def load_text_file(self, folder_path, seed, file_extension, max_depth, max_words, remove_extension, subfolder_seed=None):
        # Define a safe, empty return value for error cases
        safe_return = ("", "", "")

        if not folder_path or not folder_path.strip():
            print("[ERROR] Error: Folder path is empty.")
            return safe_return

        folder = Path(folder_path.strip())
        if not folder.is_dir():
            print(f"[ERROR] Error: Folder '{folder}' not found or is not a directory.")
            return safe_return

        ext = file_extension.strip().lower()
        if ext and not ext.startswith('.'):
            ext = f".{ext}"

        seed = to_int(seed)
        max_depth = to_int(max_depth)
        sub_seed = to_int(subfolder_seed) if subfolder_seed is not None else None

        try:
            cache_key = (str(folder.resolve()), max_depth)
            current_mtime = folder.stat().st_mtime

            if cache_key not in self.cache or self.cache[cache_key]['mtime'] != current_mtime:
                print(f"[INFO] Folder changed or not cached. Scanning '{folder}' (depth {max_depth})...")
                folders, files_by_folder = scan_tree(folder, max_depth)
                self.cache[cache_key] = {'folders': folders, 'files': files_by_folder, 'mtime': current_mtime}
            folders = self.cache[cache_key]['folders']
            files_by_folder = self.cache[cache_key]['files']

            selected_folder, inner_seed = pick_folder(folders, sub_seed, seed)
            if selected_folder is None:
                print("[ERROR] No folders found under the given path.")
                return safe_return

            files = files_by_folder.get(os.path.normpath(str(selected_folder)), [])
            if ext:
                files = [f for f in files if f.suffix.lower() == ext]
            files.sort()

            if not files:
                print(f"[ERROR] Warning: No files with extension '{ext}' found in '{selected_folder}'.")
                return safe_return

            num_files = len(files)
            selected_index = inner_seed % num_files
            selected_file = files[selected_index]

            print(f"[OK] Seed {seed} -> Folder '{selected_folder.name}' File {selected_index + 1}/{num_files}: '{selected_file.name}'")

            with open(selected_file, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()

            limited_content = self.limit_words(content, max_words)
            file_name = selected_file.stem if remove_extension else selected_file.name
            return (limited_content, file_name, str(selected_file.parent.resolve()))

        except Exception as e:
            print(f"[ERROR] An unexpected error occurred in TextLoaderCrawl: {str(e)}")
            return safe_return