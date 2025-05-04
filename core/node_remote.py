""" Capture remote URL """

import time
from typing import Dict

from comfy.utils import ProgressBar

from cozy_comfyui import \
    EnumConvertType, \
    logger, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui import \
    RGBAMaskType

from cozy_comfyui.image.convert \
    import cv_to_tensor_full

from cozy_comfyui.image.misc import \
    image_stack

from . import VideoStreamNodeHeader
from .stream import MediaStreamBase

# ==============================================================================
# === NODE ===
# ==============================================================================

class RemoteSteamReader(VideoStreamNodeHeader):
    NAME = "REMOTE"
    DESCRIPTION = """
Capture frames from a URL. Supports batch processing, allowing multiple frames to be captured simultaneously. The node provides options for configuring the source, resolution, frame rate, zoom, orientation, and interpolation method. Additionally, it supports capturing frames from multiple monitors or windows simultaneously.
"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, str]:
        d = super().INPUT_TYPES()

        return deep_merge({
            "optional": {
                "URL": ("STRING", {"default": "", "dynamicPrompts": False,
                                   "tooltip":"A remote URL to stream"})
            }
        }, d)

    def run(self, **kw) -> RGBAMaskType:
        # need to see if we have a device...
        # 63.142.190.238:6106/mjpg/video.mjpg
        if self.device is None:
            self.device = MediaStreamBase()

        images = []
        self.device.url = parse_param(kw, "URL", EnumConvertType.STRING, "")[0]
        self.device.fps = parse_param(kw, "FPS", EnumConvertType.INT, 30)[0]
        batch_size = parse_param(kw, "BATCH", EnumConvertType.INT, 1, 1)[0]
        if parse_param(kw, "PAUSE", EnumConvertType.BOOLEAN, False)[0]:
            self.device.pause()
        else:
            self.device.play()
        self.device.timeout = parse_param(kw, "TIMEOUT", EnumConvertType.INT, 8, 1, 30)[0]
        flip = parse_param(kw, "FLIP", EnumConvertType.BOOLEAN, False)
        reverse = parse_param(kw, "REVERSE", EnumConvertType.BOOLEAN, False)

        rate = 1. / self.device.fps
        pbar = ProgressBar(batch_size)
        batch_size = [batch_size] * batch_size
        params = list(zip_longest_fill(batch_size, flip, reverse))
        for idx, (batch_size, flip, reverse) in enumerate(params):
            start_time = time.perf_counter()
            self.device.flip = flip
            self.device.reverse = reverse
            while True:
                if not (img := self.device.frame) is None and img.sum() > 0:
                    break
                if time.perf_counter() - start_time > self.device.timeout:
                    logger.error("could not capture device")
                    img = self.empty
                    break

            images.append(cv_to_tensor_full(img))
            if batch_size > 1:
                time.sleep(rate)
            pbar.update_absolute(idx)

        if len(images) == 0:
            images.append(self.__empty)
        self.__last = image_stack(images)
        return self.__last
