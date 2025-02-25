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
        self.__reverse: bool = False
        self.__flip: bool = False
        self.__height = 0
        self.__width = 0
        self.__thread = threading.Thread(target=self.__run, daemon=True)
        self.__thread.start()

    def __run(self) -> None:
        start_time = time.perf_counter()
        while not self.__quit:
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
                        if self.__flip:
                            frame = cv2.flip(frame, 0)
                        if self.__reverse:
                            frame = cv2.flip(frame, 1)
                        self.__frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
                        break
                    elif time.perf_counter() - start_time > self.__timeout:
                        logger.error("could not capture frame")
                        self.__quit = True
                        break

                """
                if not ret:
                    count = int(self.__source.get(cv2.CAP_PROP_FRAME_COUNT))
                    pos = int(self.__source.get(cv2.CAP_PROP_POS_FRAMES))
                    if pos >= count:
                        self.__source.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, self.__frame = self.__source.read()
                https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4
                """

            elapsed = time.perf_counter() - start_time
            time.sleep(max(delta - elapsed, 0))
            start_time = time.perf_counter()
        self.__end()

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __end(self) -> None:
        if self.__thread:
            try:
                self.__thread.join(timeout=self.__timeout)
            except RuntimeError as e:
                pass
        self.__quit = True
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
            logger.info(f"captured url: {self.__url}")
        else:
            logger.error(f"failed to open source: {url}")

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

    @property
    def flip(self) -> bool:
        return self.__flip

    @flip.setter
    def flip(self, flip: bool) -> None:
        self.__flip = flip

    @property
    def reverse(self) -> bool:
        return self.__reverse

    @flip.setter
    def reverse(self, reverse: bool) -> None:
        self.__reverse = reverse