from setuptools import setup, find_packages

setup(
    name='CRT-Nodes',
    version='0.1',
    packages=find_packages(),  # Automatically find packages in your_custom_nodes
    install_requires=[],  # Add any dependencies here
    entry_points={
        'comfyui.nodes': [
            'Toggle Lora Unet Blocks L1 = your_custom_nodes.toggle_lora_unet_blocks_L1:ToggleLoraUnetBlocksNodeL1',
            'Toggle Lora Unet Blocks L2 = your_custom_nodes.toggle_lora_unet_blocks_L2:ToggleLoraUnetBlocksNodeL2',
            'Remove Trailing Comma = your_custom_nodes.remove_trailing_comma_node:RemoveTrailingCommaNode',
        ],
    },
)
