class ToggleLoraUnetBlocksNodeL1:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "Block_0_L1": ("BOOLEAN", {"default": False}),
                "Block_1_L1": ("BOOLEAN", {"default": False}),
                "Block_2_L1": ("BOOLEAN", {"default": False}),
                "Block_3_L1": ("BOOLEAN", {"default": False}),
                "Block_4_L1": ("BOOLEAN", {"default": False}),
                "Block_5_L1": ("BOOLEAN", {"default": False}),
                "Block_6_L1": ("BOOLEAN", {"default": False}),
                "Block_7_L1": ("BOOLEAN", {"default": False}),
                "Block_8_L1": ("BOOLEAN", {"default": False}),
                "Block_9_L1": ("BOOLEAN", {"default": False}),
                "Block_10_L1": ("BOOLEAN", {"default": False}),
                "Block_11_L1": ("BOOLEAN", {"default": False}),
                "Block_12_L1": ("BOOLEAN", {"default": False}),
                "Block_13_L1": ("BOOLEAN", {"default": False}),
                "Block_14_L1": ("BOOLEAN", {"default": False}),
                "Block_15_L1": ("BOOLEAN", {"default": False}),
                "Block_16_L1": ("BOOLEAN", {"default": False}),
                "Block_17_L1": ("BOOLEAN", {"default": False}),
                "Block_18_L1": ("BOOLEAN", {"default": False}),
                "Block_19_L1": ("BOOLEAN", {"default": False}),
                "Block_20_L1": ("BOOLEAN", {"default": False}),
                "Block_21_L1": ("BOOLEAN", {"default": False}),
                "Block_22_L1": ("BOOLEAN", {"default": False}),
                "Block_23_L1": ("BOOLEAN", {"default": False}),
                "Block_24_L1": ("BOOLEAN", {"default": False}),
                "Block_25_L1": ("BOOLEAN", {"default": False}),
                "Block_26_L1": ("BOOLEAN", {"default": False}),
                "Block_27_L1": ("BOOLEAN", {"default": False}),
                "Block_28_L1": ("BOOLEAN", {"default": False}),
                "Block_29_L1": ("BOOLEAN", {"default": False}),
                "Block_30_L1": ("BOOLEAN", {"default": False}),
                "Block_31_L1": ("BOOLEAN", {"default": False}),
                "Block_32_L1": ("BOOLEAN", {"default": False}),
                "Block_33_L1": ("BOOLEAN", {"default": False}),
                "Block_34_L1": ("BOOLEAN", {"default": False}),
                "Block_35_L1": ("BOOLEAN", {"default": False}),
                "Block_36_L1": ("BOOLEAN", {"default": False}),
                "Block_37_L1": ("BOOLEAN", {"default": False}),
                "input_string": ("STRING", {"default": ""}),  # String input to join
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "combine_strings"
    CATEGORY = "CRT"

    def combine_strings(self, Block_0_L1, Block_1_L1, Block_2_L1, Block_3_L1, Block_4_L1, Block_5_L1, Block_6_L1,
                        Block_7_L1, Block_8_L1, Block_9_L1, Block_10_L1, Block_11_L1, Block_12_L1, Block_13_L1,
                        Block_14_L1, Block_15_L1, Block_16_L1, Block_17_L1, Block_18_L1, Block_19_L1,
                        Block_20_L1, Block_21_L1, Block_22_L1, Block_23_L1, Block_24_L1, Block_25_L1,
                        Block_26_L1, Block_27_L1, Block_28_L1, Block_29_L1, Block_30_L1, Block_31_L1,
                        Block_32_L1, Block_33_L1, Block_34_L1, Block_35_L1, Block_36_L1, Block_37_L1,
                        input_string):  # Added input_string parameter
        # Define the strings for active blocks
        active_blocks = []
        for i in range(38):  # Loop through all blocks
            if locals()[f"Block_{i}_L1"]:  # Check if the block is active
                active_blocks.append(f"lora_unet_single_blocks_{i}_linear1,")

        # Combine active blocks and input string
        result = "".join(active_blocks) + ("" + input_string if input_string else "")
        
        # Return the result as a tuple
        return (result,)
