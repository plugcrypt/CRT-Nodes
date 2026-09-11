class CRT_IntToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "to_string"
    CATEGORY = "CRT/Utils/Logic & Values"

    def to_string(self, value):
        return (str(int(value)),)


NODE_CLASS_MAPPINGS = {"CRT_IntToString": CRT_IntToString}

NODE_DISPLAY_NAME_MAPPINGS = {"CRT_IntToString": "Int to String (CRT)"}