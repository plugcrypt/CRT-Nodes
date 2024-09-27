"""
@author: CRT
@title: CRT-Nodes
@version: 1.00.0
@project: "https://github.com/plugcrypt/CRT-Nodes",
@description: Single Blocks Arguments for LoRA Training
https://discord.gg/8wYS9MBQqp
"""

from .toggle_lora_unet_blocks_L1 import ToggleLoraUnetBlocksNodeL1
from .toggle_lora_unet_blocks_L2 import ToggleLoraUnetBlocksNodeL2
from .remove_trailing_comma_node import RemoveTrailingCommaNode

NODE_CLASS_MAPPINGS = {
    "Toggle Lora Unet Blocks L1": ToggleLoraUnetBlocksNodeL1,
    "Toggle Lora Unet Blocks L2": ToggleLoraUnetBlocksNodeL2,
    "Remove Trailing Comma": RemoveTrailingCommaNode,
}

NODE_DISPLAY_NAMES_MAPPINGS = {
    "Toggle Lora Unet Blocks L1": "Toggle Lora Unet Blocks L1",
    "Toggle Lora Unet Blocks L2": "Toggle Lora Unet Blocks L2",
    "Remove Trailing Comma": "Remove Trailing Comma",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAMES_MAPPINGS']
