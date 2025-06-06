""" Capture Monitors """

import time
from typing import Dict

import cv2
import mss
import numpy as np
from PIL import ImageGrab

from comfy.utils import ProgressBar

from cozy_comfyui import \
    RGBAMaskType, EnumConvertType, \
    logger, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.image import \
    ImageType

from cozy_comfyui.image.convert import \
    cv_to_tensor_full

from cozy_comfyui.image.misc import \
    image_stack

from . import StreamNodeHeader

# ==============================================================================
# === CONSTANT ===
# ==============================================================================

JOV_DOCKERENV = False
try:
    with open('/proc/1/cgroup', 'rt') as f:
        content = f.read()
        JOV_DOCKERENV = any(x in content for x in ['docker', 'kubepods', 'containerd'])
except FileNotFoundError:
    pass

if JOV_DOCKERENV:
    logger.info("RUNNING IN A DOCKER")

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def monitor_capture_all(width:int=None, height:int=None) -> ImageType:
    if JOV_DOCKERENV:
        return None

    img = ImageGrab.grab(all_screens=True)
    img = np.array(img, dtype='uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if height is not None and width is not None:
        return cv2.resize(img, (width, height))
    return img

# ==============================================================================
# === NODE ===
# ==============================================================================

class MonitorStreamReader(StreamNodeHeader):
    NAME = "MONITOR"
    DESCRIPTION = """
Capture frames from a desktop monitor. Supports batch processing, allowing multiple frames to be captured simultaneously. The node provides options for configuring the source, resolution, frame rate, zoom, orientation, and interpolation method. Additionally, it supports capturing frames from multiple monitors or windows simultaneously.
"""
    MONITOR = None

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, str]:
        if not JOV_DOCKERENV:
            cls.MONITOR = []
            with mss.mss() as screen:
                for i, m in enumerate(screen.monitors):
                    cls.MONITOR.append(f"{i}-{m['width']}x{m['height']}")

        if cls.MONITOR is None or len(cls.MONITOR) == 0:
            cls.MONITOR = ["NONE"]

        d = super().INPUT_TYPES()
        d = deep_merge({
            "optional": {
                Lexicon.MONITOR: (cls.MONITOR, {
                    "default": cls.MONITOR[0],
                    "choice": "list of system monitor devices",
                    "tooltip": "list of system monitor devices"}),
                Lexicon.XY: ("VEC2", {
                    "default": (0, 0), "mij": 0, "int": True,
                    "label": ["TOP", "LEFT"],
                    "tooltip": "Top, Left position"}),
                Lexicon.WH: ("VEC2", {
                    "default": (0, 0), "mij": 0, "int": True,
                    "label": ["WIDTH", "HEIGHT"]})
            }
        }, d)
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:

        if JOV_DOCKERENV:
            img = cv_to_tensor_full(self.empty)
            return image_stack(img)

        # only allow monitor to capture single one per "batch"
        images = []

        # allow these to "flex" length so as to animate
        monitor = parse_param(kw, Lexicon.MONITOR, EnumConvertType.STRING, "NONE")
        xy = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2INT, (0,0), 0)
        wh = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (0,0), 0)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        reverse = parse_param(kw, Lexicon.REVERSE, EnumConvertType.BOOLEAN, False)
        fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 30)
        batch_size = parse_param(kw, "batch", EnumConvertType.INT, 1, 1)[0]

        pbar = ProgressBar(batch_size)
        size = [batch_size] * batch_size
        params = list(zip_longest_fill(monitor, xy, wh, flip, reverse, fps, size))
        with mss.mss() as screen:
            for idx, (monitor, xy, wh, flip, reverse, fps, size) in enumerate(params):

                try:
                    monitor = int(monitor.split('-')[0].strip())
                except Exception:
                    logger.warning(f"bad monitor {monitor}")
                    img = self.empty
                else:
                    capture = screen.monitors[monitor]
                    width = capture['width']
                    height = capture['height']

                    # clip the position to be "in-bounds"
                    left = min(width-1, xy[0])
                    top = min(height-1, xy[1])

                    width  = width  if wh[0] == 0 else int(np.clip(wh[0], 1, width))
                    height = height if wh[1] == 0 else int(np.clip(wh[1], 1, height))
                    width = min(width, capture['width']-left)
                    height = min(height, capture['height']-top)
                    region = {
                        'top': top + capture['top'],
                        'left': left + capture['left'],
                        'width': width,
                        'height': height
                    }
                    img = screen.grab(region)
                    img = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                    if flip:
                        img = cv2.flip(img, 0)
                    if reverse:
                        img = cv2.flip(img, 1)

                images.append(cv_to_tensor_full(img))
                pbar.update_absolute(idx)
                if batch_size > 1:
                    rate = 1. / fps
                    time.sleep(rate)

        return image_stack(images)
