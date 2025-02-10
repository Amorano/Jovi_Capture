"""
Jovi_Capture - http://www.github.com/amorano/Jovi_Capture
Capture -- WEBCAM, REMOTE URLS
"""

import time
from typing import Tuple

import cv2
import torch
import numpy as np

from loguru import logger

from comfy.utils import ProgressBar

from . import \
    JOV_SCAN_DEVICES, \
    EnumConvertType, StreamNodeHeader, \
    deep_merge, parse_param

from .support.stream import MediaStreamBase
from .support.image import cv2tensor_full

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def cameraList() -> list:
    camera_list = {}
    if not JOV_SCAN_DEVICES:
        return camera_list
    failed = 0
    idx = 0
    while failed < 2:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            camera_list[idx] = {
                'w': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'h': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': int(cap.get(cv2.CAP_PROP_FPS))
            }
            cap.release()
        else:
            failed += 1
        idx += 1
    return camera_list

# ==============================================================================
# === SUPPORT - CLASS ===
# ==============================================================================

class MediaStreamCamera(MediaStreamBase):
    """A system device like a web camera."""
    def __init__(self, fps:float=30) -> None:
        self.__focus = 0
        self.__exposure = 1
        self.__zoom = 0
        self.__flip: bool = False
        super().__init__(fps=fps)

    def frame(self):
        frame = super().frame
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
            if self.__flip:
                frame = cv2.flip(frame, 1)
        except:
            pass
        return frame

    @property
    def flip(self) -> bool:
        return self.__flip

    @flip.setter
    def flip(self, flip: bool) -> None:
        self.__flip = flip

    @property
    def zoom(self) -> float:
        return self.__zoom

    @zoom.setter
    def zoom(self, val: float) -> None:
        if self.source is None:
            return
        self.__zoom = np.clip(val, 0, 1)
        val = 100 + 300 * self.__zoom
        self.source.set(cv2.CAP_PROP_ZOOM, val)

    @property
    def exposure(self) -> float:
        return self.__exposure

    @exposure.setter
    def exposure(self, val: float) -> None:
        if self.source is None:
            return
        # -10 to -1 range
        self.__exposure = np.clip(val, 0, 1)
        val = -10 + 9 * self.__exposure
        self.source.set(cv2.CAP_PROP_EXPOSURE, val)

    @property
    def focus(self) -> float:
        return self.__focus

    @focus.setter
    def focus(self, val: float) -> None:
        if self.source is None:
            return
        self.__focus = np.clip(val, 0, 1)
        val = 255 * self.__focus
        self.source.set(cv2.CAP_PROP_FOCUS, val)

# ==============================================================================
# === NODE ===
# ==============================================================================

class CameraStreamReader(StreamNodeHeader):
    NAME = "STREAM WEB CAMERA"
    CAMERAS = None
    DESCRIPTION = """
Capture frames from a web camera. Supports batch processing, allowing multiple frames to be captured simultaneously. The node provides options for configuring the source, resolution, frame rate, zoom, orientation, and interpolation method. Additionally, it supports capturing frames from multiple monitors or windows simultaneously.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()

        if cls.CAMERAS is None:
            cls.CAMERAS = [f"{i} - {v['w']}x{v['h']}" for i, v in enumerate(cameraList().values())]
        camera_default = cls.CAMERAS[0] if len(cls.CAMERAS) else "NONE"

        return deep_merge({
            "optional": {
                "CAMERA": (cls.CAMERAS, {"default": camera_default, "tooltip": "The camera from the auto-scanned list"}),
                "FLIP": ("BOOLEAN", {"default": False, "tooltip": "Camera flip image left-to-right"}),
                "ZOOM": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "Camera zoom"}),
                "FOCUS": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "Camera focus"}),
                "EXPOSURE": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1, "tooltip": "Camera exsposure"}),
            }
        }, d)

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.device = MediaStreamCamera()

    def run(self, **kw) -> Tuple[torch.Tensor, ...]:
        images = []
        self.device.fps = parse_param(kw, "FPS", EnumConvertType.INT, 30)[0]
        batch_size = parse_param(kw, "BATCH", EnumConvertType.INT, 1, 1)[0]
        if parse_param(kw, "PAUSE", EnumConvertType.BOOLEAN, False)[0]:
            self.device.pause()
        else:
            self.device.play()
        #self.device.timeout = parse_param(kw, "TIMEOUT", EnumConvertType.INT, 5, 1, 30)[0]
        url = parse_param(kw, "CAMERA", EnumConvertType.STRING, "")[0]
        self.device.url = int(url.split('-')[0].strip())
        # is in milliseconds
        self.device.flip = parse_param(kw, "FLIP", EnumConvertType.BOOLEAN, False)[0]
        self.device.zoom = parse_param(kw, "ZOOM", EnumConvertType.INT, 0, 0, 100)[0] / 100.
        self.device.focus = parse_param(kw, "FOCUS", EnumConvertType.INT, 0, 0, 100)[0] / 100.
        self.device.exposure = parse_param(kw, "EXPOSURE", EnumConvertType.INT, 0, 0, 100)[0] / 100.

        rate = 1. / self.device.fps
        pbar = ProgressBar(batch_size)
        for idx in range(batch_size):
            if (img := self.device.frame()) is None:
                images.append(self.empty)
            else:
                images.append(cv2tensor_full(img))
            if batch_size > 1:
                time.sleep(rate)
            pbar.update_absolute(idx)

        if len(images) == 0:
            logger.error("no images captured")
            return self.empty

        return [torch.stack(i) for i in zip(*images)]
