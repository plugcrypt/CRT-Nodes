class ToggleLoraUnetBlocksNodeL2:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "Block_0_L2": ("BOOLEAN", {"default": False}),
                "Block_1_L2": ("BOOLEAN", {"default": False}),
                "Block_2_L2": ("BOOLEAN", {"default": False}),
                "Block_3_L2": ("BOOLEAN", {"default": False}),
                "Block_4_L2": ("BOOLEAN", {"default": False}),
                "Block_5_L2": ("BOOLEAN", {"default": False}),
                "Block_6_L2": ("BOOLEAN", {"default": False}),
                "Block_7_L2": ("BOOLEAN", {"default": False}),
                "Block_8_L2": ("BOOLEAN", {"default": False}),
                "Block_9_L2": ("BOOLEAN", {"default": False}),
                "Block_10_L2": ("BOOLEAN", {"default": False}),
                "Block_11_L2": ("BOOLEAN", {"default": False}),
                "Block_12_L2": ("BOOLEAN", {"default": False}),
                "Block_13_L2": ("BOOLEAN", {"default": False}),
                "Block_14_L2": ("BOOLEAN", {"default": False}),
                "Block_15_L2": ("BOOLEAN", {"default": False}),
                "Block_16_L2": ("BOOLEAN", {"default": False}),
                "Block_17_L2": ("BOOLEAN", {"default": False}),
                "Block_18_L2": ("BOOLEAN", {"default": False}),
                "Block_19_L2": ("BOOLEAN", {"default": False}),
                "Block_20_L2": ("BOOLEAN", {"default": False}),
                "Block_21_L2": ("BOOLEAN", {"default": False}),
                "Block_22_L2": ("BOOLEAN", {"default": False}),
                "Block_23_L2": ("BOOLEAN", {"default": False}),
                "Block_24_L2": ("BOOLEAN", {"default": False}),
                "Block_25_L2": ("BOOLEAN", {"default": False}),
                "Block_26_L2": ("BOOLEAN", {"default": False}),
                "Block_27_L2": ("BOOLEAN", {"default": False}),
                "Block_28_L2": ("BOOLEAN", {"default": False}),
                "Block_29_L2": ("BOOLEAN", {"default": False}),
                "Block_30_L2": ("BOOLEAN", {"default": False}),
                "Block_31_L2": ("BOOLEAN", {"default": False}),
                "Block_32_L2": ("BOOLEAN", {"default": False}),
                "Block_33_L2": ("BOOLEAN", {"default": False}),
                "Block_34_L2": ("BOOLEAN", {"default": False}),
                "Block_35_L2": ("BOOLEAN", {"default": False}),
                "Block_36_L2": ("BOOLEAN", {"default": False}),
                "Block_37_L2": ("BOOLEAN", {"default": False}),
                "input_string": ("STRING", {"default": ""}),  # String input to join
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "combine_strings"
    CATEGORY = "CRT"

    def combine_strings(self, Block_0_L2, Block_1_L2, Block_2_L2, Block_3_L2, Block_4_L2, Block_5_L2, Block_6_L2,
                        Block_7_L2, Block_8_L2, Block_9_L2, Block_10_L2, Block_11_L2, Block_12_L2, Block_13_L2,
                        Block_14_L2, Block_15_L2, Block_16_L2, Block_17_L2, Block_18_L2, Block_19_L2,
                        Block_20_L2, Block_21_L2, Block_22_L2, Block_23_L2, Block_24_L2, Block_25_L2,
                        Block_26_L2, Block_27_L2, Block_28_L2, Block_29_L2, Block_30_L2, Block_31_L2,
                        Block_32_L2, Block_33_L2, Block_34_L2, Block_35_L2, Block_36_L2, Block_37_L2,
                        input_string):  # Added input_string parameter
        # Define the strings for active blocks
        active_blocks = []
        for i in range(38):  # Loop through all blocks
            if locals()[f"Block_{i}_L2"]:  # Check if the block is active
                active_blocks.append(f"lora_unet_single_blocks_{i}_linear2,")

        # Combine active blocks and input string
        result = "".join(active_blocks) + ("" + input_string if input_string else "")
        
        # Return the result as a tuple
        return (result,)
