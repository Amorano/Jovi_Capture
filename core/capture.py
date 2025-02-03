"""
Jovi_Capture - http://www.github.com/amorano/Jovi_Capture
Capture -- WEBCAM, REMOTE URLS
"""

import time
import threading
from enum import Enum
from typing import Any, Tuple

import cv2
import torch
import numpy as np

from loguru import logger

from comfy.utils import ProgressBar

from . import \
    MIN_IMAGE_SIZE, \
    EnumConvertType, JOVImageNode, \
    deep_merge, parse_param, cv2tensor_full

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def camera_list() -> list:
    camera_list = {}
    global JOV_SCAN_DEVICES

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

class MediaStreamBase:

    TIMEOUT = 5.

    def __init__(self, fps:float=30) -> None:
        self.__quit = False
        self.__paused = False
        self.__captured = False
        self.__fps = fps
        self.__timeout = None
        self.__frame = None
        self.__thread = threading.Thread(target=self.__run, daemon=True)
        self.__thread.start()

    def __run(self) -> None:
        while not self.__quit:

            delta = 1. / self.__fps
            waste = time.perf_counter() + delta

            if not self.__paused:
                if not self.__captured:
                    pause = self.__paused
                    self.__paused = True

                    if not self.capture():
                        self.__quit = True
                        break

                    self.__paused = pause
                    self.__captured = True
                    logger.info(f"CAPTURED")

                if self.__timeout is None and self.TIMEOUT > 0:
                    self.__timeout = time.perf_counter() + self.TIMEOUT

                # call the run capture frame command on subclasses
                newframe = self.callback()
                if newframe is not None:
                    self.__frame = newframe
                    self.__timeout = None

            if self.__timeout is not None and time.perf_counter() > self.__timeout:
                self.__timeout = None
                self.__quit = True
                logger.warning(f"TIMEOUT")

            waste = max(waste - time.perf_counter(), 0)
            time.sleep(waste)

        logger.info(f"STOPPED")
        self.end()

    def __del__(self) -> None:
        self.end()

    def __repr__(self) -> str:
        return self.__class__.__name__

    def callback(self) -> Tuple[bool, Any]:
        return None

    def capture(self) -> None:
        self.__captured = True
        return self.__captured

    def end(self) -> None:
        self.release()
        self.__quit = True

    def release(self) -> None:
        self.__captured = False

    def play(self) -> None:
        self.__paused = False

    def pause(self) -> None:
        self.__paused = True

    @property
    def captured(self) -> bool:
        return self.__captured

    @property
    def frame(self) -> Any:
        return self.__frame

    @property
    def fps(self) -> float:
        return self.__fps

    @fps.setter
    def fps(self, val: float) -> None:
        self.__fps = max(1, val)

class MediaStreamURL(MediaStreamBase):
    """A media point (could be a camera index)."""
    def __init__(self, url:int|str, fps:float=30) -> None:
        self.__url = url
        try: self.__url = int(url)
        except: pass
        self.__source = None
        self.__last = None
        super().__init__(fps)

    def callback(self) -> Tuple[bool, Any]:
        ret = False
        try:
            ret, result = self.__source.read()
        except:
            pass

        if ret:
            self.__last = result
            return result

        count = int(self.source.get(cv2.CAP_PROP_FRAME_COUNT))
        pos = int(self.source.get(cv2.CAP_PROP_POS_FRAMES))
        if pos >= count:
            self.source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, result = self.__source.read()

        # maybe its a single frame -- if we ever got one.
        if not ret and self.__last is not None:
            return self.__last

        return result

    @property
    def url(self) -> str:
        return self.__url

    @property
    def source(self) -> cv2.VideoCapture:
        return self.__source

    def capture(self) -> bool:
        if self.captured:
            return True
        self.__source = cv2.VideoCapture(self.__url, cv2.CAP_ANY)
        if self.captured:
            time.sleep(0.3)
            return True
        return False

    @property
    def captured(self) -> bool:
        if self.__source is None:
            return False
        return self.__source.isOpened()

    def release(self) -> None:
        if self.__source is not None:
            self.__source.release()
        super().release()

class MediaStreamDevice(MediaStreamURL):
    """A system device like a web camera."""
    def __init__(self, url:int|str, fps:float=30) -> None:
        self.__focus = 0
        self.__exposure = 1
        self.__zoom = 0
        super().__init__(url, fps=fps)

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

class StreamReader(JOVImageNode):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()

        return deep_merge(d, {
            "optional": {
                "FPS": ("INT", {"min": 1, "max": 60, "default": 30, "tooltip": "Framerate to attempt when capturing"}),
                "PAUSE": ("BOOLEAN", {"default": False, "tooltip": "If the stream should hold (pause) it's frame capture"}),
                "BATCH": ("INT", {"default": 1, "tooltip": "Number of frames wanted at the Framerate in FPS"}),
                "ZOOM": ("FLOAT", {"min": 0, "max": 1, "step": 0.005, "default": 0., "tooltip": "FILLIN"}),
            }
        })

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.capturing = 0
        a = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 4), dtype=torch.uint8, device="cpu")
        e = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8, device="cpu")
        m = torch.ones((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 1), dtype=torch.uint8, device="cpu")
        self.empty = (a, e, m,)
        self.last = [(a, e, m,)]

class WebcameraReader(StreamReader):
    NAME = "WEB CAMERA"
    SORT = 20
    CAMERAS = None
    DESCRIPTION = """
Capture frames from a web camera. Supports batch processing, allowing multiple frames to be captured simultaneously. The node provides options for configuring the source, resolution, frame rate, zoom, orientation, and interpolation method. Additionally, it supports capturing frames from multiple monitors or windows simultaneously.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()

        if cls.CAMERAS is None:
            cls.CAMERAS = [f"{i} - {v['w']}x{v['h']}" for i, v in enumerate(camera_list().values())]
        camera_default = cls.CAMERAS[0] if len(cls.CAMERAS) else "NONE"

        return deep_merge(d, {
            "optional": {
                "CAMERA": (cls.CAMERAS, {"default": camera_default, "tooltip": "FILLIN"}),
                "ZOOM": ("FLOAT", {"min": 0, "max": 1, "step": 0.005, "default": 0., "tooltip": "FILLIN"}),
            }
        })

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__device = None

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        wait = parse_param(kw, "WAIT", EnumConvertType.BOOLEAN, False)[0]
        if wait:
            return self.__last
        images = []
        batch_size, rate = parse_param(kw, "BATCH", EnumConvertType.VEC2INT, [(1, 30)], 1)[0]
        pbar = ProgressBar(batch_size)
        rate = 1. / rate

        url = parse_param(kw, "URL", EnumConvertType.STRING, "")[0]
        url = parse_param(kw, "CAMERA", EnumConvertType.STRING, "")[0]
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
                    images.append(cv2tensor_full(img))
                pbar.update_absolute(idx)
                if batch_size > 1:
                    time.sleep(rate)

        if len(images) == 0:
            images.append(self.__empty)
        self.__last = [torch.stack(i) for i in zip(*images)]
        return self.__last

class StreamReader(StreamReader):
    NAME = "STREAM READER"
    SORT = 50
    DESCRIPTION = """
Capture frames from a URL. Supports batch processing, allowing multiple frames to be captured simultaneously. The node provides options for configuring the source, resolution, frame rate, zoom, orientation, and interpolation method. Additionally, it supports capturing frames from multiple monitors or windows simultaneously.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()

        return deep_merge(d, {
            "optional": {
                "URL": ("STRING", {"default": "", "dynamicPrompts": False})
            }
        })

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__url = ""

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        wait = parse_param(kw, "WAIT", EnumConvertType.BOOLEAN, False)[0]
        if wait:
            return self.__last
        images = []
        batch_size, rate = parse_param(kw, "BATCH", EnumConvertType.VEC2INT, [(1, 30)], 1)[0]
        pbar = ProgressBar(batch_size)
        rate = 1. / rate

        url = parse_param(kw, "URL", EnumConvertType.STRING, "")[0]
        url = parse_param(kw, "CAMERA", EnumConvertType.STRING, "")[0]
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
                    images.append(cv2tensor_full(img))
                pbar.update_absolute(idx)
                if batch_size > 1:
                    time.sleep(rate)

        if len(images) == 0:
            images.append(self.__empty)
        self.__last = [torch.stack(i) for i in zip(*images)]
        return self.__last
