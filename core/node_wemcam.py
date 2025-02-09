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

from .support.stream import MediaStreamURL
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

class MediaStreamCamera(MediaStreamURL):
    """A system device like a web camera."""
    def __init__(self, fps:float=30) -> None:
        self.__focus = 0
        self.__exposure = 1
        self.__zoom = 0
        super().__init__(fps=fps)

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
                "ZOOM": ("FLOAT", {"min": 0, "max": 100, "step": 1, "default": 50, "tooltip": "Camera's own zoom level"}),
            }
        }, d)

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__device = None

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        # get device?

        self.__device = MediaStreamCamera(fps)

        wait = parse_param(kw, "WAIT", EnumConvertType.BOOLEAN, False)[0]
        if wait:
            return self.last

        images = []
        batch_size = parse_param(kw, "BATCH", EnumConvertType.INT, 1, 1)[0]
        fps = parse_param(kw, "FPS", EnumConvertType.INT, 30)[0]
        pbar = ProgressBar(batch_size)
        rate = 1. / fps

        camera = parse_param(kw, "CAMERA", EnumConvertType.STRING, "")[0]
        camera = camera.split('-')[0].strip()
        try:
            _ = int(camera)
            camera = str(camera)
        except:
            camera = ""

        if self.__device is not None:

            if wait:
                self.__device.pause()
            else:
                self.__device.play()

            self.__device.fps = fps
            self.__device.zoom = parse_param(kw, "ZOOM", EnumConvertType.FLOAT, 0, 0, 100)[0] / 100.
            # is in milliseconds
            timeout = parse_param(kw, "TIMEOUT", EnumConvertType.INT, 5, 1, 30)[0] * 1000

            for idx in range(batch_size):
                start_time = time.perf_counter()
                while (time.perf_counter() - start_time) < timeout:
                    if (img := self.__device.frame):
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
                        images.append(cv2tensor_full(img))
                        break

                if img is None:
                    logger.error("image failed to capture from device")
                    break

                pbar.update_absolute(idx)
                if batch_size > 1:
                    time.sleep(rate)

        if len(images) == 0:
            logger.error("no images captured")
            return []

        self.last = [torch.stack(i) for i in zip(*images)]
        return self.last
