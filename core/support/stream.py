"""
Jovi_Capture - http://www.github.com/amorano/Jovi_Capture
Core
"""

import time
import threading

from typing import Any, Tuple

import cv2
from loguru import logger

# ==============================================================================
# === CLASS ===
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
    def __init__(self, fps:float=30) -> None:
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

    def capture(self) -> bool:
        if self.captured:
            return True
        self.__source = cv2.VideoCapture(self.__url, cv2.CAP_ANY)
        if self.captured:
            time.sleep(0.3)
            return True
        return False

    @property
    def url(self) -> str:
        return self.__url

    @url.setter
    def url(self, url:str) -> None:
        self.__url = url

    @property
    def source(self) -> cv2.VideoCapture:
        return self.__source

    @property
    def captured(self) -> bool:
        if self.__source is None:
            return False
        return self.__source.isOpened()

    def release(self) -> None:
        if self.__source is not None:
            self.__source.release()
        super().release()
