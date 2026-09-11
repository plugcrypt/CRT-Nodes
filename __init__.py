"""
@author: CRT
@title: CRT-Nodes
@version: 2.19.0
@project: "https://github.com/PGCRT/CRT-Nodes",
@description: Set of nodes for ComfyUI
https://discord.gg/8wYS9MBQqp
"""

import folder_paths
import os
import sys

sys.modules["crt_nodes"] = sys.modules[__name__]
__package__ = "crt_nodes"

if sys.platform == "win32":
    # The Proactor event loop logs a raw ConnectionResetError whenever a client
    # drops its TCP connection while the server is still flushing writes (e.g.
    # browser websocket or batch-automation HTTP client at prompt completion).
    # Wrap the transport callback so disconnect noise stays out of the console;
    # every other exception keeps propagating unchanged.
    try:
        from asyncio import proactor_events

        _crt_original_call_connection_lost = (
            proactor_events._ProactorBasePipeTransport._call_connection_lost
        )

        if not getattr(_crt_original_call_connection_lost, "_crt_quiet_disconnect", False):
            def _crt_quiet_call_connection_lost(self, exc):
                try:
                    return _crt_original_call_connection_lost(self, exc)
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    pass

            _crt_quiet_call_connection_lost._crt_quiet_disconnect = True
            proactor_events._ProactorBasePipeTransport._call_connection_lost = (
                _crt_quiet_call_connection_lost
            )
    except Exception:
        pass

if True:

    from .py.Boolean_Transform_Node import BooleanTransformNode
    from .py.Video_Duration_Calculator import VideoDurationCalculator
    from .py.Post_Process_Node import CRTPostProcessNode
    from .py.Flux_Lora_Blocks_Patcher import FluxLoraBlocksPatcher
    from .py.Fancy_Note_Node import FancyNoteNode
    from .py.Text_Loader_Crawl import TextLoaderCrawl
    from .py.Image_Loader_Crawl import ImageLoaderCrawl
    from .py.Image_Loader_Crawl_Batch import CRT_ImageLoaderCrawlBatch
    from .py.Audio_Loader_Crawl import AudioLoaderCrawl
    from .py.Audio_Loader_Crawl_Batch import CRT_AudioLoaderCrawlBatch
    from .py.Mask_Empty_Float_Node import MaskEmptyFloatNode
    from .py.Mask_Pass_Or_Placeholder import MaskPassOrPlaceholder
    from .py.Mask_Temporal_Enhancer import MaskTemporalEnhancer
    from .py.Latent_Injection_Sampler import LatentNoiseInjectionSampler
    from .py.Face_Enhancement_Pipeline_With_Injection import (
        UltralyticsEnhancer as FaceEnhancementWithInjection,
    )
    from .py.Pony_Upscale_Sampler_With_Injection import PonyUpscaleSamplerWithInjection
    from .py.SEGS_Enhancer_Multi import (
        FaceEnhancementWithInjectionSEGS,
    )
    from .py.Mask_Censor import MaskCensor
    from .py.Sampler_Scheduler_Selector import SamplerSchedulerSelector
    from .py.Sampler_Scheduler_Crawler import SamplerSchedulerCrawler
    from .py.Resolution import Resolution
    from .py.Solid_Color import SolidColor
    from .py.Simple_Knob import SimpleKnobNode
    from .py.Simple_Toggle import SimpleToggleNode
    from .py.CLIP_Text_Encode_Unload import CRTCLIPTextEncode
    from .py.Load_Image_Resize import LoadImageResize
    from .py.Autoprompt_Processor import AutopromptProcessor
    from .py.Chroma_Key_Overlay import CRTChromaKeyOverlay
    from .py.Get_First_Last_Frame import CRTFirstLastFrameSelector
    from .py.Even_Batch_Picker import CRTEvenBatchPicker
    from .py.Image_Unbatch import CRTImageUnbatch
    from .py.Seamless_Loop_Blender import SeamlessLoopBlender
    from .py.Crop_By_Percent import CRTPctCropCalculator
    from .py.Audio_Previewer import AudioPreviewer
    from .py.Audio_Compressor import AudioCompressor
    from .py.Eq_Node import ParametricEQNode
    from .py.Load_Last_Image import LoadLastImage
    from .py.Load_Last_Video import CRTLoadLastVideo
    from .py.Save_Image_With_Path import SaveImageWithPath
    from .py.Save_Text_With_Path import SaveTextWithPath
    from .py.Save_Audio_With_Path import SaveAudioWithPath
    from .py.Video_Loader_Crawl import VideoLoaderCrawl
    from .py.Save_Video_With_Path import SaveVideoWithPath
    from .py.Save_Latent_With_Path import SaveLatentWithPath
    from .py.Load_Last_Latent import LoadLastLatent
    from .py.Save_Latents_Conditioning import SaveLatentsConditioning
    from .py.Load_Latents_Conditioning import LoadLatentsConditioning
    from .py.Enable_Latent import EnableLatent
    from .py.Boolean_Invert import BooleanInvert
    from .py.Strength_To_Steps_Node import StrengthToStepsNode
    from .py.Clarity_FX import ClarityFX
    from .py.Colourfulness_FX import ColourfulnessFX
    from .py.Film_Grain_FX import FilmGrainFX
    from .py.Technicolor2_FX import Technicolor2FX
    from .py.Advanced_Bloom_FX import AdvancedBloomFX
    from .py.Lens_FX import LensFX
    from .py.Contour_FX import ContourFX
    from .py.Color_Isolation_FX import ColorIsolationFX
    from .py.Lens_Distort_FX import LensDistortFX
    from .py.Smart_De_Noise_FX import SmartDeNoiseFX
    from .py.Arcane_Bloom_FX import ArcaneBloomFX
    from .py.Fancy_Timer_Node import FancyTimerNode
    from .py.Wan_Compare_Sampler import WAN2_2LoraCompareSampler
    from .py.Add_Settings_And_Prompt import CRT_AddSettingsAndPrompt
    from .py.Wan_Batch_Sampler import CRT_WAN_BatchSampler
    from .py.Dynamic_Prompt_Scheduler import CRT_DynamicPromptScheduler
    from .py.File_Batch_Prompt_Scheduler import CRT_FileBatchPromptScheduler
    from .py.File_Batch_Prompt_Scheduler_KREA2 import (
        CRT_FileBatchPromptSchedulerKREA2,
    )
    from .py.Text_Loader_Crawl_Batch import TextLoaderCrawlBatch
    from .py.Text_Add_Rows import TextAddRows
    from .py.Text_Rows_Crawl import TextRowsCrawl
    from .py.Extract_QA import ExtractQA
    from .py.Merge_QA import MergeQA
    from .py.Audio_Data_To_Frame_Count import AudioOrManualFrameCount
    from .py.Quantize_And_Crop import CRT_QuantizeAndCropImage
    from .py.String_Batcher import CRT_StringBatcher
    from .py.String_Splitter import CRT_StringSplitter
    from .py.Image_Dimensions_From_MP import ImageDimensionsFromMegaPixels
    from .py.Image_Dimensions_From_MP_Alt import ImageDimensionsFromMegaPixelsAlt
    from .py.Wan_Video_Lora_Select_Multi_Improved import WanVideoLoraSelectMultiImproved
    from .py.Ksampler_Batch import CRT_KSamplerBatch
    from .py.Ksampler_Batch_Advanced import CRT_KSamplerBatchAdvanced
    from .py.String_Line_Counter import CRT_StringLineCounter
    from .py.Text_Box_Line_Spot import CRT_LineSpot
    from .py.Textbox import CRT_Textbox
    from .py.Join_Strings import CRT_JoinStrings
    from .py.Remove_Lines import CRT_RemoveLines
    from .py.Int_Value import CRT_IntValue
    from .py.Int_To_String import CRT_IntToString
    from .py.FileSearch_Crawl import FileSearchCrawl
    from .py.Minimax_Length import CRT_MinimaxLength
    from .py.Mono_To_Stereo_Converter import MonoToStereoConverter
    from .py.Any_Trigger import AnyTrigger
    from .py.Depth_Anything_Tensorrt_Format import DepthAnythingTensorrtFormat
    from .py.Audio_Frame_Adjuster import AudioFrameAdjuster
    from .py.Batch_Brightness_Curve import BatchBrightnessCurve
    from .py.Image_Scale_Range_From_MP import ImageScaleRangeFromMp
    from .py.Load_Image_Base64 import LoadImageBase64
    from .py.Reference_Latent_Batch import ReferenceLatentBatch
    from .py.Save_Jpeg_Websocket import SaveJpegWebsocket
    from .py.Tile_Checker import ImageTileChecker
    from .py.Scale_Latent_To_Megapixels import ScaleLatentToMegapixels
    from .py.Resolution_By_Side import ResolutionBySide
    from .py.LTX23_Unified_Sampler import (
        CRT_LTX23USConfig,
        CRT_LTX23USModelsPipe,
        CRT_LTX23UnifiedSampler,
    )
    from .py.MiniMaxH3_Unified_Sampler import (
        CRT_MiniMaxH3USConfig,
        CRT_MiniMaxH3USModelsPipe,
        CRT_MiniMaxH3USModelsPipeCNET,
        CRT_MiniMaxH3UnifiedSampler,
    )
    from .py.H3_Fun_ControlNet import CRT_MiniMaxH3FunControlApply
    from .py.Isolate import (
        CRT_IsolateInput,
        CRT_IsolateOutput,
    )
    from .py.Isolate_CLIPSeg import CRT_IsolateInputCLIPSeg
    from .py.ERNIE_Image_Aesthetic_Scorer import ErnieImageAestheticScore
    from .py.Unsloth_Studio_Bridge import UnslothLLM
    from .py.Unsloth_Thinking_Display import CRT_UnslothThinkingDisplay
    from .py.AutoDL_Nodes import (
        NODE_CLASS_MAPPINGS as CRT_AUTODL_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as CRT_AUTODL_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .py.VAE_Decode_Last_Frame import CRTVAEDecodeLastFrame
    from .py.DepthAnything3_CRT import CRT_DepthAnything3
    from .py.MarigoldV2_CRT import CRT_MarigoldV2Depth
    from .py.YuE2_Music_Generator import (
        YuEMusicGenerator,
    )

    # Add GGUF unet folder path if not already registered
    try:
        import folder_paths
        comfy_dir = os.path.dirname(folder_paths.__file__)
        models_dir = os.path.join(comfy_dir, "models")
        unet_gguf_path = os.path.join(models_dir, "unet_gguf")
        if os.path.isdir(unet_gguf_path):
            folder_paths.add_model_folder_path("unet_gguf", unet_gguf_path)
    except Exception:
        pass

    CRT_LTX23AutoDownload = None
    LTX23AutoDownloadAPI = None

    SaveImageBase64 = None
    MagicLoraLoader = None
    SaveMergedLora = None
    _crt_pll_setup_routes = None
    try:
        from .py.Save_Image_Base64 import SaveImageBase64
    except Exception as e:
        print(f"[CRT-Nodes] Warning: Save Image Base64 node unavailable: {e}")
    try:
        from .py.Magic_Lora_Loader import (
            MagicLoraLoader,
            SaveMergedLora,
            setup_routes as _crt_pll_setup_routes,
        )
    except Exception as e:
        print(f"[CRT-Nodes] Warning: Magic LoRA Loader node unavailable: {e}")

    CRT_AudioTranscriptBatch = None
    try:
        from .py.Audio_Transcript_Batch import CRT_AudioTranscriptBatch
    except Exception as e:
        print(f"[CRT-Nodes] Warning: Audio Transcript Batch node unavailable: {e}")

    try:
        comfy_dir = os.path.dirname(folder_paths.__file__)
        models_dir = os.path.join(comfy_dir, "models")
        bbox_path = os.path.join(models_dir, "ultralytics", "bbox")
        segm_path = os.path.join(models_dir, "ultralytics", "segm")

        if os.path.isdir(bbox_path):
            folder_paths.add_model_folder_path("ultralytics_bbox", bbox_path)
        if os.path.isdir(segm_path):
            folder_paths.add_model_folder_path("ultralytics_segm", segm_path)
    except Exception as e:
        print(f"[CRT-Nodes] Warning: Could not register ultralytics paths. Error: {e}")

    if _crt_pll_setup_routes is not None:
        try:
            _crt_pll_setup_routes(os.path.dirname(os.path.abspath(__file__)))
        except Exception as e:
            print(f"[CRT-Nodes] Warning: Could not setup CRT PLL routes. Error: {e}")

else:
    pass

NODE_CLASS_MAPPINGS = {
    "Boolean Transform": BooleanTransformNode,
    "Video Duration Calculator": VideoDurationCalculator,
    "CRT Post-Process Suite": CRTPostProcessNode,
    "FluxLoraBlocksPatcher": FluxLoraBlocksPatcher,
    "FancyNoteNode": FancyNoteNode,
    "TextLoaderCrawl": TextLoaderCrawl,
    "ImageLoaderCrawl": ImageLoaderCrawl,
    "CRT_ImageLoaderCrawlBatch": CRT_ImageLoaderCrawlBatch,
    "AudioLoaderCrawl": AudioLoaderCrawl,
    "CRT_AudioLoaderCrawlBatch": CRT_AudioLoaderCrawlBatch,
    "MaskEmptyFloatNode": MaskEmptyFloatNode,
    "MaskPassOrPlaceholder": MaskPassOrPlaceholder,
    "MaskTemporalEnhancer": MaskTemporalEnhancer,
    "LatentNoiseInjectionSampler": LatentNoiseInjectionSampler,
    "PonyUpscaleSamplerWithInjection": PonyUpscaleSamplerWithInjection,
    "FaceEnhancementWithInjection": FaceEnhancementWithInjection,
    "FaceEnhancementWithInjectionSEGS": FaceEnhancementWithInjectionSEGS,
    "MaskCensor": MaskCensor,
    "SamplerSchedulerSelector": SamplerSchedulerSelector,
    "SamplerSchedulerCrawler": SamplerSchedulerCrawler,
    "Resolution": Resolution,
    "SolidColor": SolidColor,
    "SimpleKnobNode": SimpleKnobNode,
    "SimpleToggleNode": SimpleToggleNode,
    "CRTCLIPTextEncode": CRTCLIPTextEncode,
    "LoadImageResize": LoadImageResize,
    "AutopromptProcessor": AutopromptProcessor,
    "CRTChromaKeyOverlay": CRTChromaKeyOverlay,
    "CRTFirstLastFrameSelector": CRTFirstLastFrameSelector,
    "CRTEvenBatchPicker": CRTEvenBatchPicker,
    "CRTImageUnbatch": CRTImageUnbatch,
    "SeamlessLoopBlender": SeamlessLoopBlender,
    "CRTPctCropCalculator": CRTPctCropCalculator,
    "AudioPreviewer": AudioPreviewer,
    "AudioCompressor": AudioCompressor,
    "ParametricEQNode": ParametricEQNode,
    "LoadLastImage": LoadLastImage,
    "CRTLoadLastVideo": CRTLoadLastVideo,
    "SaveImageWithPath": SaveImageWithPath,
    "SaveTextWithPath": SaveTextWithPath,
    "SaveAudioWithPath": SaveAudioWithPath,
    "VideoLoaderCrawl": VideoLoaderCrawl,
    "SaveVideoWithPath": SaveVideoWithPath,
    "SaveLatentWithPath": SaveLatentWithPath,
    "LoadLastLatent": LoadLastLatent,
    "SaveLatentsConditioning": SaveLatentsConditioning,
    "LoadLatentsConditioning": LoadLatentsConditioning,
    "EnableLatent": EnableLatent,
    "BooleanInvert": BooleanInvert,
    "Strength To Steps": StrengthToStepsNode,
    "ClarityFX": ClarityFX,
    "ColourfulnessFX": ColourfulnessFX,
    "FilmGrainFX": FilmGrainFX,
    "Technicolor2FX": Technicolor2FX,
    "AdvancedBloomFX": AdvancedBloomFX,
    "LensFX": LensFX,
    "ContourFX": ContourFX,
    "ColorIsolationFX": ColorIsolationFX,
    "LensDistortFX": LensDistortFX,
    "SmartDeNoiseFX": SmartDeNoiseFX,
    "ArcaneBloomFX": ArcaneBloomFX,
    "FancyTimerNode": FancyTimerNode,
    "WAN2.2 LoRA Compare Sampler": WAN2_2LoraCompareSampler,
    "CRT_AddSettingsAndPrompt": CRT_AddSettingsAndPrompt,
    "CRT_WAN_BatchSampler": CRT_WAN_BatchSampler,
    "CRT_DynamicPromptScheduler": CRT_DynamicPromptScheduler,
    "CRT_FileBatchPromptScheduler": CRT_FileBatchPromptScheduler,
    "CRT_FileBatchPromptSchedulerKREA2": CRT_FileBatchPromptSchedulerKREA2,
    "TextLoaderCrawlBatch": TextLoaderCrawlBatch,
    "AudioOrManualFrameCount": AudioOrManualFrameCount,
    "CRT_QuantizeAndCropImage": CRT_QuantizeAndCropImage,
    "CRT_StringBatcher": CRT_StringBatcher,
    "CRT_StringSplitter": CRT_StringSplitter,
    "ImageDimensionsFromMegaPixels": ImageDimensionsFromMegaPixels,
    "ImageDimensionsFromMegaPixelsAlt": ImageDimensionsFromMegaPixelsAlt,
    "WanVideoLoraSelectMultiImproved": WanVideoLoraSelectMultiImproved,
    "CRT_KSamplerBatch": CRT_KSamplerBatch,
    "CRT_KSamplerBatchAdvanced": CRT_KSamplerBatchAdvanced,
    "CRT_StringLineCounter": CRT_StringLineCounter,
    "Text Box line spot": CRT_LineSpot,
    "CRT_Textbox": CRT_Textbox,
    "CRT_JoinStrings": CRT_JoinStrings,
    "CRT_RemoveLines": CRT_RemoveLines,
    "TextAddRows": TextAddRows,
    "TextRowsCrawl": TextRowsCrawl,
    "ExtractQA": ExtractQA,
    "MergeQA": MergeQA,
    "CRT_IntValue": CRT_IntValue,
    "CRT_IntToString": CRT_IntToString,
    "FileSearchCrawl": FileSearchCrawl,
    "CRT_MinimaxLength": CRT_MinimaxLength,
    "MonoToStereoConverter": MonoToStereoConverter,
    "AnyTrigger": AnyTrigger,
    "DepthAnythingTensorrtFormat": DepthAnythingTensorrtFormat,
    "AudioFrameAdjuster": AudioFrameAdjuster,
    "BatchBrightnessCurve": BatchBrightnessCurve,
    "ImageScaleRangeFromMp": ImageScaleRangeFromMp,
    "LoadImageBase64": LoadImageBase64,
    "ReferenceLatentBatch": ReferenceLatentBatch,
    "SaveJpegWebsocket": SaveJpegWebsocket,
    "ImageTileChecker": ImageTileChecker,
    "ScaleLatentToMegapixels": ScaleLatentToMegapixels,
    "ResolutionBySide": ResolutionBySide,
    "CRT_LTX23USModelsPipe": CRT_LTX23USModelsPipe,
    "CRT_LTX23USConfig": CRT_LTX23USConfig,
    "CRT_LTX23UnifiedSampler": CRT_LTX23UnifiedSampler,
    "CRT_MiniMaxH3USModelsPipe": CRT_MiniMaxH3USModelsPipe,
    "CRT_MiniMaxH3USConfig": CRT_MiniMaxH3USConfig,
    "CRT_MiniMaxH3UnifiedSampler": CRT_MiniMaxH3UnifiedSampler,
    "CRT_MiniMaxH3USModelsPipeCNET": CRT_MiniMaxH3USModelsPipeCNET,
    "CRT_MiniMaxH3FunControlApply": CRT_MiniMaxH3FunControlApply,
    "CRT_IsolateInput": CRT_IsolateInput,
    "CRT_IsolateOutput": CRT_IsolateOutput,
    "CRT_IsolateInputCLIPSeg": CRT_IsolateInputCLIPSeg,
    "ErnieImageAestheticScore": ErnieImageAestheticScore,
    "UnslothLLM": UnslothLLM,
    "CRT_UnslothThinkingDisplay": CRT_UnslothThinkingDisplay,
    "CRTVAEDecodeLastFrame": CRTVAEDecodeLastFrame,
    "CRT_DepthAnything3": CRT_DepthAnything3,
    "CRT_MarigoldV2Depth": CRT_MarigoldV2Depth,
    "YuEMusicGenerator": YuEMusicGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Boolean Transform": "String to Boolean (CRT)",
    "Video Duration Calculator": "Video Duration Calculator (CRT)",
    "CRT Post-Process Suite": "Post-Process Suite (CRT)",
    "FluxLoraBlocksPatcher": "Flux LoRA Blocks Patcher (CRT)",
    "FancyNoteNode": "Fancy Note (CRT)",
    "TextLoaderCrawl": "Text Loader Crawl (CRT)",
    "ImageLoaderCrawl": "Image Loader Crawl (CRT)",
    "CRT_ImageLoaderCrawlBatch": "Image Loader Crawl Batch (CRT)",
    "AudioLoaderCrawl": "Audio Loader Crawl (CRT)",
    "CRT_AudioLoaderCrawlBatch": "Audio Loader Crawl Batch (CRT)",
    "MaskEmptyFloatNode": "Mask Empty Float (CRT)",
    "MaskPassOrPlaceholder": "Mask Pass or Placeholder (CRT)",
    "MaskTemporalEnhancer": "Mask Temporal Enhancer (CRT)",
    "LatentNoiseInjectionSampler": "Latent Noise Injection Sampler (CRT)",
    "PonyUpscaleSamplerWithInjection": "Image Upscale Sampler (CRT)",
    "FaceEnhancementWithInjection": "Ultralytics Enhancer (CRT)",
    "FaceEnhancementWithInjectionSEGS": "SEGS Enhancer Multi (CRT)",
    "MaskCensor": "Mask Censor (CRT)",
    "SamplerSchedulerSelector": "Sampler & Scheduler Selector (CRT)",
    "SamplerSchedulerCrawler": "Sampler & Scheduler Crawler (CRT)",
    "Resolution": "Resolution (CRT)",
    "SolidColor": "Solid Color (CRT)",
    "SimpleKnobNode": "K",
    "SimpleToggleNode": "T",
    "CRTCLIPTextEncode": "CLIP Text Encode + Unload (CRT)",
    "LoadImageResize": "Load Image Resize (CRT)",
    "AutopromptProcessor": "AutopromptProcessor (CRT)",
    "CRTChromaKeyOverlay": "Chroma Key Overlay (CRT)",
    "CRTFirstLastFrameSelector": "Get First & Last Frame (CRT)",
    "CRTEvenBatchPicker": "Even Batch Picker (CRT)",
    "CRTImageUnbatch": "Image Unbatch (CRT)",
    "SeamlessLoopBlender": "Seamless Loop Blender (CRT)",
    "CRTPctCropCalculator": "Percentage Crop Calculator (CRT)",
    "AudioPreviewer": "Preview Audio (CRT)",
    "AudioCompressor": "Tube Compressor (CRT)",
    "ParametricEQNode": "Parametric EQ (CRT)",
    "LoadLastImage": "Load Last Image (CRT)",
    "CRTLoadLastVideo": "Load Last Video (CRT)",
    "SaveImageWithPath": "Save Image With Path (CRT)",
    "SaveTextWithPath": "Save Text With Path (CRT)",
    "SaveAudioWithPath": "Save Audio With Path (CRT)",
    "VideoLoaderCrawl": "Video Loader Crawl (CRT)",
    "SaveVideoWithPath": "Save Video With Path (CRT)",
    "SaveLatentWithPath": "Save Latent With Path (CRT)",
    "LoadLastLatent": "Load Last Latent (CRT)",
    "SaveLatentsConditioning": "Save Latents Conditioning (CRT)",
    "LoadLatentsConditioning": "Load Latents Conditioning (CRT)",
    "EnableLatent": "Enable Latent (CRT)",
    "BooleanInvert": "Boolean Invert (CRT)",
    "Strength To Steps": "Strength to Steps (CRT)",
    "ClarityFX": "Clarity FX (CRT)",
    "ColourfulnessFX": "Colourfulness FX (CRT)",
    "FilmGrainFX": "Film Grain FX (CRT)",
    "Technicolor2FX": "Technicolor 2 FX (CRT)",
    "AdvancedBloomFX": "Advanced Bloom FX (CRT)",
    "LensFX": "Lens FX (CRT)",
    "ContourFX": "Contour FX (CRT)",
    "ColorIsolationFX": "Color Isolation FX (CRT)",
    "LensDistortFX": "Lens Distort FX (CRT)",
    "SmartDeNoiseFX": "Smart DeNoise FX (CRT)",
    "ArcaneBloomFX": "Arcane Bloom FX (CRT)",
    "FancyTimerNode": "Fancy Timer (CRT)",
    "WAN2.2 LoRA Compare Sampler": "WAN 2.2 LoRA Compare Sampler (CRT)",
    "CRT_AddSettingsAndPrompt": "Add Settings and Prompt (CRT)",
    "CRT_WAN_BatchSampler": "WAN 2.2 Batch Sampler (CRT)",
    "CRT_DynamicPromptScheduler": "Dynamic Prompt Scheduler (CRT)",
    "CRT_FileBatchPromptScheduler": "File Batch Prompt Scheduler (CRT)",
    "CRT_FileBatchPromptSchedulerKREA2": (
        "File Batch Prompt Scheduler KREA2 (CRT)"
    ),
    "TextLoaderCrawlBatch": "Text Loader Crawl Batch (CRT)",
    "AudioOrManualFrameCount": "Frame Count (Audio or Manual) (CRT)",
    "CRT_QuantizeAndCropImage": "Quantize and Crop Image (CRT)",
    "CRT_StringBatcher": "String Batcher (CRT)",
    "CRT_StringSplitter": "String Splitter (CRT)",
    "ImageDimensionsFromMegaPixels": "Image Dimensions From Megapixels (CRT)",
    "ImageDimensionsFromMegaPixelsAlt": "Image Dimensions From MP alt (CRT)",
    "WanVideoLoraSelectMultiImproved": "Wan Video Multi-LoRA Select (CRT)",
    "CRT_KSamplerBatch": "KSampler Batch (CRT)",
    "CRT_KSamplerBatchAdvanced": "KSampler Batch Advanced (CRT)",
    "CRT_StringLineCounter": "String Line Counter (CRT)",
    "Text Box line spot": "TextBox line spot (CRT)",
    "CRT_Textbox": "Textbox (CRT)",
    "CRT_JoinStrings": "Join Strings (CRT)",
    "CRT_RemoveLines": "Remove Lines (CRT)",
    "TextAddRows": "Text Add Rows (CRT)",
    "TextRowsCrawl": "Text Rows Crawl (CRT)",
    "ExtractQA": "Extract Q/A (CRT)",
    "MergeQA": "Merge Q/A (CRT)",
    "CRT_IntValue": "Int Value (CRT)",
    "CRT_IntToString": "Int to String (CRT)",
    "FileSearchCrawl": "FileSearch Crawl (CRT)",
    "CRT_MinimaxLength": "Minimax Length (CRT)",
    "MonoToStereoConverter": "Mono to Stereo Converter (CRT)",
    "AnyTrigger": "Any Trigger (CRT)",
    "DepthAnythingTensorrtFormat": "Depth Anything Tensorrt Format (CRT)",
    "AudioFrameAdjuster": "Audio Frame Adjuster (CRT)",
    "BatchBrightnessCurve": "Batch Brightness Curve (U-Shape) (CRT)",
    "ImageScaleRangeFromMp": "Image Scale Range From MP (CRT)",
    "LoadImageBase64": "Load Image Base64 (CRT)",
    "ReferenceLatentBatch": "Reference Latent Batch (CRT)",
    "SaveJpegWebsocket": "Save JPEG Websocket (CRT)",
    "ImageTileChecker": "Image Tile Checker (CRT)",
    "ScaleLatentToMegapixels": "Scale Latent To Megapixels (CRT)",
    "ResolutionBySide": "Resolution By Side (CRT)",
    "CRT_LTX23USModelsPipe": "LTX US Models Pipe (CRT)",
    "CRT_LTX23USConfig": "LTX US Config (CRT)",
    "CRT_LTX23UnifiedSampler": "LTX Unified Sampler (CRT)",
    "CRT_MiniMaxH3USModelsPipe": "MiniMax H3 US Models Pipe (CRT)",
    "CRT_MiniMaxH3USConfig": "MiniMax H3 US Config (CRT)",
    "CRT_MiniMaxH3UnifiedSampler": "MiniMax H3 Unified Sampler (CRT)",
    "CRT_MiniMaxH3USModelsPipeCNET": "MiniMax H3 US Models Pipe CNET (CRT)",
    "CRT_MiniMaxH3FunControlApply": "MiniMax H3 Fun Control Apply (CRT)",
    "CRT_IsolateInput": "Isolate Input SAM3.1 (CRT)",
    "CRT_IsolateOutput": "Isolate Output (CRT)",
    "CRT_IsolateInputCLIPSeg": "Isolate Input CLIPSeg (CRT)",
    "ErnieImageAestheticScore": "ERNIE Image Aesthetic Score (CRT)",
    "UnslothLLM": "Unsloth Studio Bridge (CRT)",
    "CRT_UnslothThinkingDisplay": "Unsloth Studio Bridge Thinking Display (CRT)",
    "CRTVAEDecodeLastFrame": "VAE Decode Last Frame (CRT)",
    "CRT_DepthAnything3": "Fast Depth Anything v3 (CRT)",
    "CRT_MarigoldV2Depth": "Heavy Depth Marigold v2 (CRT)",
    "YuEMusicGenerator": "YuE Music Generator (CRT)",
}

NODE_CLASS_MAPPINGS.update(CRT_AUTODL_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(CRT_AUTODL_NODE_DISPLAY_NAME_MAPPINGS)

if SaveImageBase64 is not None:
    NODE_CLASS_MAPPINGS["SaveImageBase64"] = SaveImageBase64
    NODE_DISPLAY_NAME_MAPPINGS["SaveImageBase64"] = "Save Image Base64 (CRT)"

if MagicLoraLoader is not None:
    NODE_CLASS_MAPPINGS["Magic LoRA Loader"] = MagicLoraLoader
    NODE_DISPLAY_NAME_MAPPINGS["Magic LoRA Loader"] = "Magic LoRA Loader (CRT)"

if SaveMergedLora is not None:
    NODE_CLASS_MAPPINGS["Magic Save Merged LoRA"] = SaveMergedLora
    NODE_DISPLAY_NAME_MAPPINGS["Magic Save Merged LoRA"] = (
        "Magic Save Merged LoRA (CRT)"
    )

if CRT_AudioTranscriptBatch is not None:
    NODE_CLASS_MAPPINGS["CRT_AudioTranscriptBatch"] = CRT_AudioTranscriptBatch
    NODE_DISPLAY_NAME_MAPPINGS["CRT_AudioTranscriptBatch"] = (
        "Audio Transcript Batch (CRT)"
    )

# Filter out None values from mappings
NODE_CLASS_MAPPINGS = {k: v for k, v in NODE_CLASS_MAPPINGS.items() if v is not None}
NODE_DISPLAY_NAME_MAPPINGS = {
    k: v for k, v in NODE_DISPLAY_NAME_MAPPINGS.items() if v is not None
}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
