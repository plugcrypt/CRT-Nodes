class RemoveTrailingCommaNode:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "input_string": ("STRING", {"default": ""}),  # Input string to remove trailing comma
            }
        }

    RETURN_TYPES = ("STRING",)  # Output type
    FUNCTION = "remove_trailing_comma"
    CATEGORY = "CRT"

    def remove_trailing_comma(self, input_string):
        # Trim the trailing comma if it exists
        if input_string.endswith(","):
            trimmed_string = input_string[:-1]  # Remove the last character
        else:
            trimmed_string = input_string  # No change if no trailing comma
        
        return (trimmed_string,)  # Return as a tuple
