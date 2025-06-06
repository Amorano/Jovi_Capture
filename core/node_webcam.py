""" Capture -- WEBCAM """

import os
import time
from typing import Any, Dict, List

import cv2
import numpy as np
from aiohttp import web

from comfy.utils import ProgressBar
from server import PromptServer

from cozy_comfyui import \
    EnumConvertType, \
    logger, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui import \
    RGBAMaskType

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.image.convert import \
    cv_to_tensor_full

from cozy_comfyui.image.misc import \
    image_stack

from . import VideoStreamNodeHeader
from .. import PACKAGE
from .stream import MediaStreamBase

# ==============================================================================
# === CONSTANT ===
# ==============================================================================

JOV_SCAN_DEVICES = os.getenv("JOV_SCAN_DEVICES", "False").lower() in ['1', 'true', 'on']

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def camera_list() -> List[str]:
    idx = 0
    failed = 0
    camera_list = []
    while failed < 1:
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

    if len(camera_list) == 0:
        camera_list = ["NONE"]
    return camera_list

# ==============================================================================
# === API ROUTE ===
# ==============================================================================

@PromptServer.instance.routes.get(f"/{PACKAGE.lower()}/camera")
async def route_cameraList(req) -> Any:
    force = req.query_string == "force=true"
    if force and not JOV_SCAN_DEVICES:
        return web.json_response(["NONE"])
    CameraStreamReader.CAMERAS = camera_list()
    return web.json_response(CameraStreamReader.CAMERAS, content_type='application/json')

# ==============================================================================
# === CLASS ===
# ==============================================================================

class MediaStreamCamera(MediaStreamBase):
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

class CameraStreamReader(VideoStreamNodeHeader):
    NAME = "CAMERA"
    DESCRIPTION = """
Capture frames from a web camera. Supports batch processing, allowing multiple frames to be captured simultaneously. The node provides options for configuring the source, resolution, frame rate, zoom, orientation, and interpolation method. Additionally, it supports capturing frames from multiple monitors or windows simultaneously.
"""
    CAMERAS = None

    @classmethod
    # Dict[str, Dict[str, Tuple[str, Dict[str, Any]]]]:
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        d = super().INPUT_TYPES()

        if cls.CAMERAS is None:
            cls.CAMERAS = camera_list() if JOV_SCAN_DEVICES else ["0 -NONE"]

        d = deep_merge({
            "optional": {
                Lexicon.CAMERA: (cls.CAMERAS, {
                    "default": cls.CAMERAS[0],
                    "tooltip": "Camera from auto-scanned list"}),
                Lexicon.ZOOM: ("INT", {
                    "default": 0, "min": 0, "max": 100, "step": 1}),
                Lexicon.FOCUS: ("INT", {
                    "default": 0, "min": 0, "max": 100, "step": 1}),
                Lexicon.EXPOSURE: ("INT", {
                    "default": 50, "min": 0, "max": 100, "step": 1})
            }
        }, d)
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        # need to see if we have a device...
        url = parse_param(kw, Lexicon.CAMERA, EnumConvertType.STRING, "")[0]
        try:
            url = int(url.split('-')[0].strip())
        except Exception:
            logger.warning(f"bad camera url {url}")
            img = cv_to_tensor_full(self.empty)
            return image_stack(img)

        if self.device is None:
            self.device = MediaStreamCamera()

        self.device.timeout = parse_param(kw, Lexicon.TIMEOUT, EnumConvertType.INT, 8, 1, 30)[0]
        self.device.url = url

        images = []
        self.device.fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 30)[0]
        batch_size = parse_param(kw, Lexicon.BATCH, EnumConvertType.INT, 1, 1)[0]
        if parse_param(kw, Lexicon.PAUSE, EnumConvertType.BOOLEAN, False)[0]:
            self.device.pause()
        else:
            self.device.play()
        self.device.zoom = parse_param(kw, Lexicon.ZOOM, EnumConvertType.INT, 0, 0, 100)[0] / 100.
        self.device.focus = parse_param(kw, Lexicon.FOCUS, EnumConvertType.INT, 0, 0, 100)[0] / 100.
        self.device.exposure = parse_param(kw, Lexicon.EXPOSURE, EnumConvertType.INT, 0, 0, 100)[0] / 100.
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        reverse = parse_param(kw, Lexicon.REVERSE, EnumConvertType.BOOLEAN, False)

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

        return image_stack(images)
