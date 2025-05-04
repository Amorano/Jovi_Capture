""" Jovi_Capture - Core """

from typing import Dict

import numpy as np

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.node import \
    CozyImageNode

from cozy_comfyui import \
    IMAGE_SIZE_MIN, \
    deep_merge

# ==============================================================================
# === CLASS ===
# ==============================================================================

class StreamNodeHeader(CozyImageNode):
    NOT_IDEMPOTENT = True

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float('nan')

    @classmethod
    def VALIDATE_INPUTS(cls, input_types) -> bool:
        return True

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, str]:
        d = super().INPUT_TYPES()

        return deep_merge(d, {
            "optional": {
                Lexicon.FLIP: ("BOOLEAN", {
                    "default": False}),
                Lexicon.REVERSE: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reverse image left-to-right"}),
                Lexicon.FPS: ("INT", {
                    "default": 30, "min": 1, "max": 60}),
                Lexicon.BATCH: ("INT", {
                    "default": 1, "min": 1,
                    "tooltip": "Number of frames wanted at the Framerate in FPS"}),
            }
        })

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.empty = np.zeros((IMAGE_SIZE_MIN, IMAGE_SIZE_MIN, 4), dtype=np.uint8)

class VideoStreamNodeHeader(StreamNodeHeader):
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, str]:
        d = super().INPUT_TYPES()

        return deep_merge(d, {
            "optional": {
                Lexicon.PAUSE: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Hold (pause) frame capture"}),
                Lexicon.TIMEOUT: ("INT", {
                    "default": 8, "min": 5, "max": 30, "step": 1})
            }
        })

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.device = None
