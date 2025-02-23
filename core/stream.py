"""
Jovi_Capture - http://www.github.com/amorano/Jovi_Capture
Core
"""

import time
import threading
from typing import Any

import cv2

from cozy_comfyui import logger

# ==============================================================================
# === CLASS ===
# ==============================================================================

class MediaStreamBase:

    def __init__(self, fps:float=30, timeout:float=5) -> None:
        self.__fps = fps
        self.__timeout = min(30, max(0.5, timeout))
        self.__running = True
        self.__quit = False
        self.__frame = None
        self.__source = None
        self.__url = None
        self.__height = 0
        self.__width = 0
        self.__thread = threading.Thread(target=self.__run, daemon=True)
        self.__thread.start()

    def __run(self) -> None:
        while not self.__quit:
            start_time = time.perf_counter()
            if not self.__source or not self.__source.isOpened():
                logger.error("waiting on device")
                time.sleep(0.5)
                if time.perf_counter() - start_time > self.__timeout:
                    logger.error("could not capture device")
                    self.__quit = True
                continue

            delta = 1. / self.__fps
            if self.__running:
                while True:
                    ret, frame = self.__source.read()
                    if ret and frame is not None and frame.sum() > 0:
                        break
                    if time.perf_counter() - start_time > self.__timeout:
                        logger.error("could not capture frame")
                        self.__quit = True
                        break
                self.__frame = frame
                """
                if not ret:
                    count = int(self.__source.get(cv2.CAP_PROP_FRAME_COUNT))
                    pos = int(self.__source.get(cv2.CAP_PROP_POS_FRAMES))
                    if pos >= count:
                        self.__source.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, self.__frame = self.__source.read()
                """

            elapsed = time.perf_counter() - start_time
            time.sleep(max(delta - elapsed, 0))
        self.__end()

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __end(self) -> None:
        self.__quit = True
        if self.__thread:
            self.__thread.join(timeout=self.__timeout)
        if self.__source is not None:
            self.__source.release()
            self.__source = None

    def play(self) -> None:
        self.__running = True

    def pause(self) -> None:
        self.__running = False

    @property
    def url(self) -> str:
        return self.__url

    @url.setter
    def url(self, url:str) -> None:
        if url == self.__url:
            return

        if self.__source is not None:
            self.__source.release()
            self.__source = None

        new_source = cv2.VideoCapture(url, cv2.CAP_ANY)
        if new_source.isOpened():
            self.__source = new_source
            self.__url = url
            logger.info(f"Captured camera device: {self.__url}")
            return

        logger.error(f"Failed to open camera source: {url}")

    @property
    def source(self) -> cv2.VideoCapture:
        return self.__source

    @property
    def frame(self) -> Any:
        return self.__frame

    @property
    def fps(self) -> float:
        return self.__fps

    @fps.setter
    def fps(self, val: float) -> None:
        self.__fps = min(60, max(1, val))

    @property
    def running(self) -> bool:
        return self.__running

    @property
    def timeout(self) -> float:
        return self.__timeout

    @timeout.setter
    def timeout(self, timeout: float) -> None:
        self.__timeout = min(30, max(1, timeout))

    @property
    def width(self) -> int:
        return self.__width

    @width.setter
    def width(self, width:int) -> None:
        if self.__source is not None:
            self.__width = max(1, width)
            self.__source.set(cv2.CAP_PROP_FRAME_WIDTH, self.__width)

    @property
    def height(self) -> int:
        return self.__height

    @height.setter
    def height(self, height:int) -> None:
        if self.__source is not None:
            self.__height = max(1, height)
            self.__source.set(cv2.CAP_PROP_FRAME_HEIGHT, self.__height)
