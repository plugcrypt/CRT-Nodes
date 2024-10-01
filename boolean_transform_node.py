class BooleanTransformNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {
                    "multiline": False, 
                    "default": "0.0"
                })
            }
        }

    # Returning both INT and BOOLEAN outputs
    RETURN_TYPES = ("INT", "BOOLEAN")
    FUNCTION = "transform_to_boolean"
    CATEGORY = "CRT"  # Category for UI organization

    def transform_to_boolean(self, input_string):
        # If input_string is a list, get the first element
        if isinstance(input_string, list) and input_string:
            input_string = input_string[0]
        
        # Ensure the input is a string
        if not isinstance(input_string, str):
            input_string = str(input_string)
        
        try:
            # Convert the string to a float
            num = float(input_string)
        except ValueError:
            # If conversion fails, treat it as 0
            num = 0.0

        # Boolean value (True if non-zero, False if zero)
        boolean_value = num != 0
        # Integer equivalent (1 if non-zero, 0 if zero)
        int_value = 1 if boolean_value else 0

        # Return both int and boolean values (int_value and boolean_value)
        return (int_value, boolean_value)
