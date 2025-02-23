"""
Jovi_Capture - http://www.github.com/amorano/Jovi_Capture
REMOTE -- Capture remove URL
"""

import time
from typing import Dict, Tuple

import cv2
import torch

from comfy.utils import ProgressBar

from cozy_comfyui import \
    EnumConvertType, \
    logger, \
    deep_merge, parse_param

from cozy_comfyui.image.convert import cv_to_tensor_full

from . import StreamNodeHeader

# ==============================================================================
# === NODE ===
# ==============================================================================

class RemoteSteamReader(StreamNodeHeader):
    NAME = "REMOTE"
    DESCRIPTION = """
Capture frames from a URL. Supports batch processing, allowing multiple frames to be captured simultaneously. The node provides options for configuring the source, resolution, frame rate, zoom, orientation, and interpolation method. Additionally, it supports capturing frames from multiple monitors or windows simultaneously.
"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, str]:
        d = super().INPUT_TYPES()

        return deep_merge({
            "optional": {
                "URL": ("STRING", {"default": "", "dynamicPrompts": False})
            }
        }, d)

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__url = ""
        self.__device = None # MediaStreamURL

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        wait = parse_param(kw, "WAIT", EnumConvertType.BOOLEAN, False)[0]
        if wait:
            return self.__last
        images = []
        batch_size, rate = parse_param(kw, "BATCH", EnumConvertType.VEC2INT, [(1, 30)], 1)[0]
        pbar = ProgressBar(batch_size)
        rate = 1. / rate

        url = parse_param(kw, "URL", EnumConvertType.STRING, "")[0]
        url = url.split('-')[0].strip()
        try:
            _ = int(url)
            url = str(url)
        except: url = ""

        # timeout and try again?
        if self.__capturing > 0 and time.perf_counter() - self.__capturing > 3000:
            logger.error(f'timed out {self.__url}')
            self.__capturing = 0
            self.__url = ""

        if self.__device is not None:
            self.__capturing = 0

            if wait:
                self.__device.pause()
            else:
                self.__device.play()

            fps = parse_param(kw, "FPS", EnumConvertType.INT, 30)[0]
            self.__device.fps = fps
            self.__device.zoom = parse_param(kw, "ZOOM", EnumConvertType.FLOAT, 0, 0, 1)[0]

            for idx in range(batch_size):
                img = self.__device.frame
                if img is None:
                    images.append(self.__empty)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
                    images.append(cv_to_tensor_full(img))
                pbar.update_absolute(idx)
                if batch_size > 1:
                    time.sleep(rate)

        if len(images) == 0:
            images.append(self.__empty)
        self.__last = [torch.stack(i) for i in zip(*images)]
        return self.__last
