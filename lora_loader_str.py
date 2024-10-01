import folder_paths
import comfy.utils

class LoraLoaderStr:
    """Node to load LoRA models into a given model and clip."""

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        # Get the list of LoRA files available in the specified directory
        file_list = folder_paths.get_filename_list("loras")
        file_list.insert(0, "None")  # Add "None" option at the top of the list
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "switch": (["Off", "On"],),  # Switch to enable/disable LoRA loading
                "lora_name": (file_list,),  # Selection box for available LoRA files
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "include_strength": (["Yes", "No"],)  # Toggle to include strength in output
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")  # Added STRING as a return type
    FUNCTION = "load_lora"
    CATEGORY = "CRT"

    def load_lora(self, model, clip, switch, lora_name, strength_model, strength_clip, include_strength):
        """Loads the specified LoRA model into the provided model and clip."""
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, "No LoRA Loaded")  # Return a message when no LoRA is loaded

        if switch == "Off" or lora_name == "None":
            return (model, clip, "No LoRA Loaded")  # Return a message when no LoRA is loaded

        # Get the full path of the selected LoRA
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None

        # Check if the LoRA has already been loaded
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                del self.loaded_lora  # Clear loaded LoRA if different

        if lora is None:
            try:
                # Load the LoRA if it's not already loaded
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)
            except Exception as e:
                print(f"Error loading LoRA from {lora_path}: {e}")
                return (model, clip, "Error loading LoRA")  # Return error message

        # Apply the LoRA to the model and clip
        try:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

            # Prepare the output string
            output_string = f"LoRA: {lora_name}"
            if include_strength == "Yes":
                output_string += f" (Strength Model: {strength_model}, Strength Clip: {strength_clip})"

            return (model_lora, clip_lora, output_string)  # Return the LoRA name and strength values if toggled
        except Exception as e:
            print(f"Error applying LoRA: {e}")
            return (model, clip, "Error applying LoRA")  # Return error message
