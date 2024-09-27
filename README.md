# CRT-Nodes

CRT-Nodes is a collection of custom nodes for ComfyUI. 
This set includes toggle nodes for Lora Unet blocks and a node to remove trailing comma from string end. 

## Features

1. **Toggle Lora Unet Blocks L1**: 
   - This node allows you to toggle Lora Unet blocks at layer 1. 
   - It takes a string input and outputs a concatenated string of the active blocks.

2. **Toggle Lora Unet Blocks L2**: 
   - Similar to the L1 node but designed for layer 2. 
   - It also takes a string input and outputs a concatenated string of the active blocks.

3. **Remove Trailing Comma**: 
   - This node takes a string input and removes the last trailing comma, providing a clean string output for the Flux LoRA training "block_args".

![workflow (63)](https://github.com/user-attachments/assets/53e01a25-4f57-4dd0-bb7a-581907dd652a)


Example:

Toggle Lora Unet Blocks L1 Node
specify which blocks you want to toggle on.
The output will provide a concatenated string of the active blocks.

Toggle Lora Unet Blocks L2 Node
Similar to the L1 node, connect your input and specify blocks to toggle.

Remove Trailing Comma Node
Connect the input string that may have a trailing comma, and the output will be a cleaned string without the trailing comma.

Contributing
Feel free to submit issues and pull requests. Contributions are welcome!

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
Thanks to the ComfyUI community for their support and contributions.

https://civitai.green/user/pgc
https://discord.gg/8wYS9MBQqp