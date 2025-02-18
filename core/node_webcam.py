"""
Jovi_Capture - http://www.github.com/amorano/Jovi_Capture
Capture -- WEBCAM, REMOTE URLS
"""

import time
from typing import List, Tuple

import cv2
import torch
import numpy as np
from loguru import logger

from comfy.utils import ProgressBar

from cozy_comfyui import \
    EnumConvertType, \
    deep_merge, parse_param

from cozy_comfyui.image.convert import cv_to_tensor_full

from . import \
    JOV_SCAN_DEVICES, \
    StreamNodeHeader

from .stream import MediaStreamBase

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def cameraList() -> List[str]:
    camera_list = []
    if not JOV_SCAN_DEVICES:
        return camera_list
    failed = 0
    idx = 0
    while failed < 2:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            f = int(cap.get(cv2.CAP_PROP_FPS))
            camera_list.append(f"{idx} - {w}x{h}x{f}")
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

    @property
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
            cameras = cameraList()
            cls.CAMERAS = cameras if len(cameras) else ["NONE"]
        camera_default = cls.CAMERAS[0]

        return deep_merge({
            "optional": {
                "CAMERA": (cls.CAMERAS, {"default": camera_default, "tooltip": "The camera from the auto-scanned list"}),
                "FLIP": ("BOOLEAN", {"default": False, "tooltip": "Camera flip image left-to-right"}),
                "ZOOM": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "Camera zoom"}),
                "FOCUS": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "Camera focus"}),
                "EXPOSURE": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1, "tooltip": "Camera exposure"})
            }
        }, d)

    def run(self, **kw) -> Tuple[torch.Tensor, ...]:
        # need to see if we have a device...
        url = parse_param(kw, "CAMERA", EnumConvertType.STRING, "")[0]
        url = "0"
        try:
            url = int(url.split('-')[0].strip())
        except Exception:
            logger.warning(f"bad camera url {url}")
            return self.empty

        if self.device is None:
            self.device = MediaStreamCamera()

        self.device.timeout = parse_param(kw, "TIMEOUT", EnumConvertType.INT, 5, 1, 30)[0]
        self.device.url = url

        #wh = parse_param(kw, "WH", EnumConvertType.VEC2INT, [640, 480], 160)[0]
        #self.device.width = wh[0]
        #self.device.height = wh[1]

        images = []
        self.device.fps = parse_param(kw, "FPS", EnumConvertType.INT, 30)[0]
        batch_size = parse_param(kw, "BATCH", EnumConvertType.INT, 1, 1)[0]
        if parse_param(kw, "PAUSE", EnumConvertType.BOOLEAN, False)[0]:
            self.device.pause()
        else:
            self.device.play()

        self.device.flip = parse_param(kw, "FLIP", EnumConvertType.BOOLEAN, False)[0]
        self.device.zoom = parse_param(kw, "ZOOM", EnumConvertType.INT, 0, 0, 100)[0] / 100.
        self.device.focus = parse_param(kw, "FOCUS", EnumConvertType.INT, 0, 0, 100)[0] / 100.
        self.device.exposure = parse_param(kw, "EXPOSURE", EnumConvertType.INT, 0, 0, 100)[0] / 100.

        rate = 1. / self.device.fps
        pbar = ProgressBar(batch_size)
        for idx in range(batch_size):
            start_time = time.perf_counter()
            while True:
                if not (img := self.device.frame) is None and img.sum() > 0:
                    break
                if time.perf_counter() - start_time > self.device.timeout:
                    logger.error("could not capture device")
                    return self.empty

            images.append(cv_to_tensor_full(img))
            if batch_size > 1:
                time.sleep(rate)
            pbar.update_absolute(idx)

        return [torch.stack(i) for i in zip(*images)]
