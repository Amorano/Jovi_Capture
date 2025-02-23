"""Capture Dekstop Window"""

import re
import json
import time
import platform
from typing import Any, Dict, Optional, Tuple

import cv2
import torch
import numpy as np
import pywinctl as pwc
from aiohttp import web

from loguru import logger

from comfy.utils import ProgressBar
from server import PromptServer

from cozy_comfyui import \
    EnumConvertType, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui import RGBAMaskType
from cozy_comfyui.image import ImageType
from cozy_comfyui.image.convert import cv_to_tensor_full

if platform.system() == "Windows":
    import win32gui
    import win32ui
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
        # Get correct window rect based on capture mode
        if client_area_only:
            rect = win32gui.GetClientRect(hwnd)
            left, top = win32gui.ClientToScreen(hwnd, (0, 0))
            right = left + rect[2]
            bottom = top + rect[3]
        else:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)

        width = right - left
        height = bottom - top

        # Adjust for region if specified
        if region:
            rx, ry, rw, rh = region
            left += rx
            top += ry
            width = min(rw, width - rx)
            height = min(rh, height - ry)

        window_dc = win32gui.GetWindowDC(hwnd)
        dc = win32ui.CreateDCFromHandle(window_dc)
        compatible_dc = dc.CreateCompatibleDC()

        try:
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(dc, width, height)
            compatible_dc.SelectObject(bitmap)

            # Set the correct source coordinates for BitBlt
            if client_area_only or region:
                compatible_dc.BitBlt((0, 0), (width, height), dc, (rx if region else 0, ry if region else 0), win32con.SRCCOPY)
            else:
                windll.user32.PrintWindow(hwnd, compatible_dc.GetSafeHdc(), 2)

            bmpstr = bitmap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype='uint8')
            img = img.reshape((height, width, 4))

        finally:
            dc.DeleteDC()
            compatible_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, window_dc)
            win32gui.DeleteObject(bitmap.GetHandle())

        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

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
            bounds.size.width = min(rw, bounds.size.width - rx)
            bounds.size.height = min(rh, bounds.size.height - ry)

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
        return img.reshape((height, width, 4))

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
            width = min(rw, width - rx)
            height = min(rh, height - ry)

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

        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

# ==============================================================================
# === API ROUTE ===
# ==============================================================================

@PromptServer.instance.routes.get(f"/{PACKAGE.lower()}/window")
async def route_windowList(req) -> Any:
    WindowStreamReader.WINDOWS = window_list()
    return web.json_response(WindowStreamReader.WINDOWS)

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

        return deep_merge({
            "optional": {
                "WINDOW": (keys, {"default": default, "tooltip": "Window to capture"}),
                "XY": ("VEC2INT", {"default": (0, 0), "mij": 0, "label": ["TOP", "LEFT"], "tooltip": "Top, Left position"}),
                "WH": ("VEC2INT", {"default": (0, 0), "mij": 0, "label": ["WIDTH", "HEIGHT"], "tooltip": "Width and Height"}),
                "CLIENT": ("BOOLEAN", {"default": False, "tooltip": "Only capture the client area -- no scrollbars or menus"}),
            }
        }, d)

    def run(self, **kw) -> RGBAMaskType:
        images = []
        batch_size = parse_param(kw, "BATCH", EnumConvertType.INT, 1, 1)[0]
        window = parse_param(kw, "WINDOW", EnumConvertType.STRING, "")
        fps = parse_param(kw, "FPS", EnumConvertType.INT, 30)
        xy = parse_param(kw, "XY", EnumConvertType.VEC2INT, [(0,0)], 0)
        wh = parse_param(kw, "WH", EnumConvertType.VEC2INT, [(0,0)], 0)
        client = parse_param(kw, "CLIENT", EnumConvertType.BOOLEAN, False)
        pbar = ProgressBar(batch_size)
        size = [batch_size] * batch_size
        params = list(zip_longest_fill(window, fps, xy, wh, client, size))
        for idx, (window, fps, xy, wh, client, size) in enumerate(params):
            window = self.WINDOWS[window]
            region = None
            if (img := window_capture(window, client, region)) is None:
                img = self.empty

            images.append(cv_to_tensor_full(img))
            if batch_size > 1:
                rate = 1. / fps
                time.sleep(rate)
            pbar.update_absolute(idx)

        return [torch.stack(i) for i in zip(*images)]
