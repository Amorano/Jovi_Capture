"""
Jovi_Capture - http://www.github.com/amorano/Jovi_Capture
Core
"""

__version__ = "1.0.0"

import os

import torch

from cozy_comfyui.node import CozyImageNode
from cozy_comfyui import \
    MIN_IMAGE_SIZE, \
    deep_merge

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

JOV_SCAN_DEVICES = os.getenv("JOV_SCAN_DEVICES", "False").lower() in ['1', 'true', 'on']

# ==============================================================================
# === CLASS ===
# ==============================================================================

class StreamNodeHeader(CozyImageNode):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()

        return deep_merge(d, {
            "optional": {
                # "WH": ("VEC2INT", {"default": [640, 480], "mij": 160, "tooltip": "width and height"}),
                "FPS": ("INT", {"default": 30, "min": 1, "max": 60,
                                "tooltip": "Framerate to attempt when capturing"}),
                "BATCH": ("INT", {"default": 1, "min": 1,
                                  "tooltip": "Number of frames wanted at the Framerate in FPS"}),
                "PAUSE": ("BOOLEAN", {"default": False,
                                      "tooltip": "If the stream should hold (pause) it's frame capture"}),
                "TIMEOUT": ("INT", {"default": 5, "min": 5, "max": 30, "step": 1,
                                    "tooltip": "How long to wait before failing if no frames are being captured"})
            }
        })

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.device = None
        self.empty = [
            torch.zeros((1, MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 4), dtype=torch.uint8, device="cpu"),
            torch.zeros((1, MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8, device="cpu"),
            torch.zeros((1, MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 1), dtype=torch.uint8, device="cpu")
        ]
