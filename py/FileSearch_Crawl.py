import os
from pathlib import Path

from ._crawl_common import pick_folder, scan_tree, to_int


class FileSearchCrawl:
    def __init__(self):
        # Instance-level cache of the folder tree, keyed by (resolved root, max_depth)
        # and invalidated when the root directory's mtime changes.
        self.cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "tooltip": "Root folder to crawl for files."}),
                "search": ("STRING", {"default": ".txt", "tooltip": "Search (ext or keyword). Starts with a dot to match an extension (e.g. .txt), otherwise matches any file whose name contains this keyword. Empty = no filter."}),
                "max_depth": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1,
                        "max": 100,
                        "tooltip": "Subfolder crawl depth. 0 = only the root folder, 1 = one level deep, -1 = infinite.",
                    },
                ),
                "file_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Selects the file. With no subfolder seed connected, crawls each folder's first match, then each folder's second match, etc.",
                    },
                ),
                "folder_sort": (
                    ["alphabetical", "last created", "last modified"],
                    {"default": "alphabetical", "tooltip": "How to order folders before seed selection."},
                ),
                "file_sort": (
                    ["alphabetical", "last created", "last modified"],
                    {"default": "alphabetical", "tooltip": "How to order files before seed selection."},
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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = (
        "file_name",
        "folder_name",
        "folder_path",
        "file_path",
        "extension",
        "all_files",
        "match_count",
        "selected_index",
    )
    FUNCTION = "search_files"
    CATEGORY = "CRT/Load"
    DESCRIPTION = "Crawls a folder tree using a seed for folder selection (or column-major with the file seed) and one for file selection, and outputs name, folder, path, extension, all matching files, match count, and selected index."

    def _sort_key(self, mode):
        if mode == "last created":
            return lambda p: p.stat().st_ctime
        if mode == "last modified":
            return lambda p: p.stat().st_mtime
        return lambda p: p.name.lower()

    def search_files(self, folder_path, search, max_depth, file_seed, folder_sort, file_sort, subfolder_seed=None):
        safe_return = ("", "", "", "", "", "", 0, 0)

        if not folder_path or not folder_path.strip():
            print("[ERROR] FileSearch Crawl (CRT): folder path is empty.")
            return safe_return

        root = Path(folder_path.strip())
        if not root.is_dir():
            print(f"[ERROR] FileSearch Crawl (CRT): folder '{root}' not found or is not a directory.")
            return safe_return

        search = search.strip().lower()
        max_depth = to_int(max_depth)
        file_seed = to_int(file_seed)
        sub_seed = to_int(subfolder_seed) if subfolder_seed is not None else None

        key = (str(root.resolve()), max_depth)
        current_mtime = root.stat().st_mtime
        if key not in self.cache or self.cache[key]["mtime"] != current_mtime:
            folders, files_by_folder = scan_tree(root, max_depth)
            self.cache[key] = {"mtime": current_mtime, "folders": folders, "files": files_by_folder}
        else:
            folders = self.cache[key]["folders"]
            files_by_folder = self.cache[key]["files"]

        folders = sorted(folders, key=self._sort_key(folder_sort))
        folder, inner_seed = pick_folder(folders, sub_seed, file_seed)
        if folder is None:
            print(f"[ERROR] FileSearch Crawl (CRT): no folders found under '{root}'.")
            return safe_return

        matching = list(files_by_folder.get(os.path.normpath(str(folder)), []))
        if search:
            if search.startswith('.'):
                matching = [f for f in matching if f.suffix.lower() == search]
            else:
                matching = [f for f in matching if search in f.name.lower()]
        matching.sort(key=self._sort_key(file_sort))

        match_count = len(matching)
        if match_count == 0:
            print(f"[ERROR] FileSearch Crawl (CRT): no files matching '{search}' found in '{folder}'.")
            return safe_return

        selected_index = inner_seed % match_count
        selected = matching[selected_index]
        all_files = "\n".join(str(p) for p in matching)

        return (
            selected.stem,
            folder.name,
            str(folder),
            str(selected),
            selected.suffix,
            all_files,
            match_count,
            selected_index,
        )


NODE_CLASS_MAPPINGS = {"FileSearchCrawl": FileSearchCrawl}
NODE_DISPLAY_NAME_MAPPINGS = {"FileSearchCrawl": "FileSearch Crawl (CRT)"}