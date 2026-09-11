import torch


class CRTImageUnbatch:
    MAX_IMAGES = 64

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",) * MAX_IMAGES
    RETURN_NAMES = tuple(f"image_{i + 1}" for i in range(MAX_IMAGES))

    FUNCTION = "unbatch"
    CATEGORY = "CRT/Image"

    def unbatch(self, images: torch.Tensor):
        batch_size = images.shape[0]
        last = images[-1:]
        outputs = []
        for i in range(self.MAX_IMAGES):
            if i < batch_size:
                outputs.append(images[i:i + 1])
            else:
                outputs.append(last)
        return tuple(outputs)


NODE_CLASS_MAPPINGS = {"CRTImageUnbatch": CRTImageUnbatch}
NODE_DISPLAY_NAME_MAPPINGS = {"CRTImageUnbatch": "Image Unbatch (CRT)"}