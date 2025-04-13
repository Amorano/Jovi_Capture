""" Jovi_Capture - Core """

from typing import Dict

import numpy as np

from cozy_comfyui.node import \
    CozyImageNode

from cozy_comfyui import \
    IMAGE_SIZE_MIN, \
    deep_merge

# ==============================================================================
# === CLASS ===
# ==============================================================================

class StreamNodeHeader(CozyImageNode):
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, str]:
        d = super().INPUT_TYPES()

        return deep_merge(d, {
            "optional": {
                "FLIP": ("BOOLEAN", {"default": False, "tooltip": "Flip image top-to-bottom"}),
                "REVERSE": ("BOOLEAN", {"default": False, "tooltip": "reverse image left-to-right"}),
                "FPS": ("INT", {"default": 30, "min": 1, "max": 60,
                                "tooltip": "Framerate to attempt when capturing"}),
                "BATCH": ("INT", {"default": 1, "min": 1,
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
                "PAUSE": ("BOOLEAN", {"default": False,
                                      "tooltip": "If the stream should hold (pause) it's frame capture"}),
                "TIMEOUT": ("INT", {"default": 5, "min": 5, "max": 30, "step": 1,
                                    "tooltip": "How long to wait before failing if no frames are being captured"})
            }
        })

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.device = None
