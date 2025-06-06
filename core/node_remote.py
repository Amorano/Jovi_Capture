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

from cozy_comfyui.lexicon import \
    Lexicon

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

        d = deep_merge({
            "optional": {
                Lexicon.URL: ("STRING", {
                    "default": "", "dynamicPrompts": False,
                    "tooltip":"A remote URL to stream"})
            }
        }, d)
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        # need to see if we have a device...
        # 63.142.190.238:6106/mjpg/video.mjpg
        if self.device is None:
            self.device = MediaStreamBase()

        images = []
        self.device.url = parse_param(kw, Lexicon.URL, EnumConvertType.STRING, "")[0]
        if parse_param(kw, Lexicon.PAUSE, EnumConvertType.BOOLEAN, False)[0]:
            self.device.pause()
        else:
            self.device.play()

        self.device.timeout = parse_param(kw, Lexicon.TIMEOUT, EnumConvertType.INT, 8, 1, 30)[0]
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        reverse = parse_param(kw, Lexicon.REVERSE, EnumConvertType.BOOLEAN, False)
        self.device.fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 30)[0]
        batch_size = parse_param(kw, Lexicon.BATCH, EnumConvertType.INT, 1, 1)[0]


        rate = 1. / self.device.fps
        pbar = ProgressBar(batch_size)
        batch_size = [batch_size] * batch_size
        params = list(zip_longest_fill(flip, reverse, batch_size))
        for idx, (flip, reverse, batch_size) in enumerate(params):
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
