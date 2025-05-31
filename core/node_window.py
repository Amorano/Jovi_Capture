""" Capture Dekstop Window """

import re
import json
import time
import platform
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pywinctl as pwc
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

from cozy_comfyui.image import \
    ImageType

from cozy_comfyui.image.convert import \
    cv_to_tensor_full

from cozy_comfyui.image.misc import \
    image_stack

if platform.system() == "Windows":
    import win32gui
    import win32ui
    import win32con
    from ctypes import windll
elif platform.system() == "Darwin":
    from Quartz import *
elif platform.system() == "Linux":
    from Xlib import display, X
    from Xlib.ext import composite

from . import StreamNodeHeader
from .. import ROOT, PACKAGE

# ==============================================================================
# === INITIALIZE ===
# ==============================================================================

try:
    with open(f"{ROOT}/skip.json", "r") as fp:
        IGNORE_LIST = json.load(fp)
    IGNORE_LIST['regex'] = [re.compile(x) for x in IGNORE_LIST['regex']]
except Exception as e:
    logger.error(e)
    IGNORE_LIST = {
        "full": [],
        "regex": []
    }

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def window_list() -> Dict[int, str]:
    """List all draggable, user-usable windows with their handles and titles."""
    windows = pwc.getAllWindows()
    valid_windows = {}
    for win in windows:
        if win.isVisible and not win.isMinimized and \
            win.width>0 and win.height>0 \
            and win.title not in IGNORE_LIST['full'] and \
            not any(pattern.search(win.title) for pattern in IGNORE_LIST['regex']):
                valid_windows[win.title] = win.getHandle()

    return valid_windows

def window_capture(hwnd: int, client_area_only: bool=False, region: Optional[Tuple[int,...]]=None) -> ImageType:
    """
    Capture a window or region within a window.

    Args:
        hwnd: Window handle
        client_area_only: If True, captures only the client area without borders/decorations
        region: Optional (x, y, width, height) tuple specifying region within window to capture

    Returns:
        ImageType: Captured image in RGBA format
    """
    system = platform.system()
    if system == "Windows":
        dc = None
        compatible_dc = None
        bitmap = None
        window_dc = None

        # Get correct window rect based on capture mode
        if client_area_only:
            rect = win32gui.GetClientRect(hwnd)
            left, top = win32gui.ClientToScreen(hwnd, (0, 0))
            max_width = rect[2]
            max_height = rect[3]
        else:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            max_width = right - left
            max_height = bottom - top

        if region is None:
            x, y = 0, 0
            width, height = max_width, max_height
        else:
            rl, rt, rw, rh = region
            x = min(max(0, rl), max_width - 1)
            y = min(max(0, rt), max_height - 1)
            width  = max_width  if rw == 0 else min(x + rw, max_width)
            height = max_height if rh == 0 else min(y + rh, max_height)

        # logger.info(f"Capture region: pos=({x},{y}) size=({width},{height})")

        img = None
        try:
            window_dc = win32gui.GetWindowDC(hwnd)
            dc = win32ui.CreateDCFromHandle(window_dc)
            compatible_dc = dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(dc, width, height)
            compatible_dc.SelectObject(bitmap)

            result = windll.user32.PrintWindow(hwnd, compatible_dc.GetSafeHdc(), 0)
            if result is None:
                return None
            bmpstr = bitmap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype='uint8')
            img = img.reshape((height, width, 4))
            img = img[y:y+height, x:x+width]
            img = img[..., [2, 1, 0, 3]]

        except Exception as e:
            logger.error(e)
        finally:
            if bitmap:
                win32gui.DeleteObject(bitmap.GetHandle())
            if compatible_dc:
                compatible_dc.DeleteDC()
            if dc:
                dc.DeleteDC()
            if window_dc:
                win32gui.ReleaseDC(hwnd, window_dc)

    elif system == "Darwin":
        # Get window info
        window_list = CGWindowListCopyWindowInfo(
            kCGWindowListOptionIncludingWindow,
            hwnd
        )
        window_info = window_list[0]

        # Get bounds
        bounds = window_info[kCGWindowBounds]
        if client_area_only:
            # Adjust bounds to exclude title bar and borders
            # Note: This is an approximation, as macOS doesn't have a direct equivalent
            bounds.origin.y += 22  # Typical title bar height
            bounds.size.height -= 22

        if region:
            rx, ry, rw, rh = region
            bounds.origin.x += rx
            bounds.origin.y += ry

            bounds.size.width = min(rw, bounds.size.width - rx) if rw > 0 else bounds.size.width - rx
            bounds.size.height = min(rh, bounds.size.height - ry) if rh > 0 else bounds.size.height - ry

        bounds.size.width = max(1, bounds.size.width)
        bounds.size.height = max(1, bounds.size.height)

        # Create image of window contents
        image = CGWindowListCreateImage(
            bounds,
            kCGWindowListOptionIncludingWindow,
            hwnd,
            kCGWindowImageBoundsIgnoreFraming | kCGWindowImageShouldBeOpaque
        )

        dataProvider = CGImageGetDataProvider(image)
        data = dataProvider.copy()

        width = CGImageGetWidth(image)
        height = CGImageGetHeight(image)
        img = np.frombuffer(data, dtype=np.uint8)
        img = img.reshape((height, width, 4))

    elif system == "Linux":
        d = display.Display()
        window = d.create_resource_object('window', hwnd)

        if client_area_only:
            # Get window properties to find decorations
            prop = window.get_full_property(
                d.intern_atom('_NET_FRAME_EXTENTS'),
                X.AnyPropertyType
            )
            if prop:
                # left, right, top, bottom
                frame_extents = prop.value
                x = frame_extents[0]
                y = frame_extents[2]
                geom = window.get_geometry()
                width = geom.width - (frame_extents[0] + frame_extents[1])
                height = geom.height - (frame_extents[2] + frame_extents[3])
            else:
                geom = window.get_geometry()
                x, y = 0, 0
                width, height = geom.width, geom.height
        else:
            geom = window.get_geometry()
            x, y = 0, 0
            width, height = geom.width, geom.height

        if region:
            rx, ry, rw, rh = region
            x += rx
            y += ry
            width = min(rw, width - rx) if rw > 0 else width - rx
            height = min(rh, height - ry) if rh > 0 else height - ry

        width = max(1, width)
        height = max(1, height)

        try:
            composite.composite_redirect_window(d, window, True)
            pixmap = window.create_pixmap(width, height, window.get_attributes().depth)
            gc = pixmap.create_gc()

            # Copy the specified region
            window.composite_name_window_pixmap()
            gc.copy_area(window, pixmap, x, y, 0, 0, width, height)

            image = pixmap.get_image(0, 0, width, height, X.ZPixmap, 0xffffffff)
            img = np.frombuffer(image.data, dtype=np.uint8)
            img = img.reshape((height, width, 4))

        finally:
            gc.free()
            pixmap.free()

    return img

# ==============================================================================
# === API ROUTE ===
# ==============================================================================

@PromptServer.instance.routes.get(f"/{PACKAGE.lower()}/window")
async def route_windowList(req) -> Any:
    WindowStreamReader.WINDOWS = window_list()
    return web.json_response(WindowStreamReader.WINDOWS, content_type='application/json')

# ==============================================================================
# === NODE ===
# ==============================================================================

class WindowStreamReader(StreamNodeHeader):
    NAME = "WINDOW"
    DESCRIPTION = """
Capture frames from a dekstop window. Supports batch processing, allowing multiple frames to be captured simultaneously. The node provides options for configuring the source, resolution, frame rate, zoom, orientation, and interpolation method. Additionally, it supports capturing frames from multiple monitors or windows simultaneously.
"""
    WINDOWS = None

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, str]:
        d = super().INPUT_TYPES()

        if cls.WINDOWS is None:
            cls.WINDOWS = window_list()

        default = ""
        keys = []
        if len(cls.WINDOWS):
            keys = list(cls.WINDOWS.keys())
            default = keys[0]

        d = deep_merge({
            "optional": {
                Lexicon.WINDOW: (keys, {
                    "default": default,
                    "tooltip": "Window to capture"}),
                Lexicon.XY: ("VEC2", {
                    "default": (0, 0), "mij": 0, "int": True,
                    "label": ["TOP", "LEFT"],
                    "tooltip": "Top, Left position"}),
                Lexicon.WH: ("VEC2", {
                    "default": (0, 0), "mij": 0, "int": True,
                    "label": ["WIDTH", "HEIGHT"]}),
                Lexicon.CLIENT: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only capture the client area -- no scrollbars or menus"}),
            }
        }, d)
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        images = []
        batch_size = parse_param(kw, Lexicon.BATCH, EnumConvertType.INT, 1, 1)[0]
        window = parse_param(kw, Lexicon.WINDOW, EnumConvertType.STRING, "")
        fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 30)
        xy = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2INT, (0,0), 0)
        wh = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (0,0), 0)
        client = parse_param(kw, Lexicon.CLIENT, EnumConvertType.BOOLEAN, False)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        reverse = parse_param(kw, Lexicon.REVERSE, EnumConvertType.BOOLEAN, False)
        pbar = ProgressBar(batch_size)
        size = [batch_size] * batch_size
        params = list(zip_longest_fill(window, fps, xy, wh, client, flip, reverse, size))
        for idx, (window, fps, xy, wh, client, flip, reverse, size) in enumerate(params):
            try:
                window = self.WINDOWS[window]
            except Exception as e:
                logger.error(e)
                img = self.empty
            else:
                region = (xy[0], xy[1], wh[0], wh[1])
                if (img := window_capture(window, client, region)) is None:
                    img = self.empty
                else:
                    if flip:
                        img = cv2.flip(img, 0)
                    if reverse:
                        img = cv2.flip(img, 1)

            images.append(cv_to_tensor_full(img))
            if batch_size > 1:
                rate = 1. / fps
                time.sleep(rate)
            pbar.update_absolute(idx)

        return image_stack(images)
