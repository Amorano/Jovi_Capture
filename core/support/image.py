"""
Jovi_Capture - http://www.github.com/amorano/Jovi_Capture
Core
"""

from typing import Tuple

import torch
import numpy as np

from .. import \
    MIN_IMAGE_SIZE, TYPE_IMAGE, TYPE_PIXEL, TYPE_iRGBA, \
    JOVBaseNode, \
    deep_merge

# ==============================================================================
# === IMAGE SUPPORT ===
# ==============================================================================

def cv2tensor_full(image: TYPE_IMAGE, matte:TYPE_PIXEL=(0,0,0,255)) \
    -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    rgba = image_convert(image, 4)
    rgb = image_matte(rgba, matte)[...,:3]
    mask = image_mask(image)
    rgba = torch.from_numpy(rgba.astype(np.float32) / 255.0)
    rgb = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    mask = torch.from_numpy(mask.astype(np.float32) / 255.0)
    return rgba, rgb, mask

def image_convert(image: TYPE_IMAGE, channels: int, width: int=None, height: int=None,
                  matte: Tuple[int, ...]=(0, 0, 0, 255)) -> TYPE_IMAGE:
    """Force image format to a specific number of channels.
    Args:
        image (TYPE_IMAGE): Input image.
        channels (int): Desired number of channels (1, 3, or 4).
        width (int): Desired width. `None` means leave unchanged.
        height (int): Desired height. `None` means leave unchanged.
        matte (tuple): RGBA color to use as background color for transparent areas.
    Returns:
        TYPE_IMAGE: Image with the specified number of channels.
    """
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    if (cc := image.shape[2]) != channels:
        if   cc == 1 and channels == 3:
            image = np.repeat(image, 3, axis=2)
        elif cc == 1 and channels == 4:
            rgb = np.repeat(image, 3, axis=2)
            alpha = np.full(image.shape[:2] + (1,), matte[3], dtype=image.dtype)
            image = np.concatenate([rgb, alpha], axis=2)
        elif cc == 3 and channels == 1:
            image = np.mean(image, axis=2, keepdims=True).astype(image.dtype)
        elif cc == 3 and channels == 4:
            alpha = np.full(image.shape[:2] + (1,), matte[3], dtype=image.dtype)
            image = np.concatenate([image, alpha], axis=2)
        elif cc == 4 and channels == 1:
            rgb = image[..., :3]
            alpha = image[..., 3:4] / 255.0
            image = (np.mean(rgb, axis=2, keepdims=True) * alpha).astype(image.dtype)
        elif cc == 4 and channels == 3:
            image = image[..., :3]

    # Resize if width or height is specified
    h, w = image.shape[:2]
    new_width = width if width is not None else w
    new_height = height if height is not None else h
    if (new_width, new_height) != (w, h):
        # Create a new image with the matte color
        new_image = np.full((new_height, new_width, channels), matte[:channels], dtype=image.dtype)
        paste_x = (new_width - w) // 2
        paste_y = (new_height - h) // 2
        new_image[paste_y:paste_y+h, paste_x:paste_x+w] = image[:h, :w]
        image = new_image

    return image

def image_mask(image: TYPE_IMAGE, color: TYPE_PIXEL = 255) -> TYPE_IMAGE:
    """Create a mask from the image, preserving transparency.

    Args:
        image (TYPE_IMAGE): Input image, assumed to be 2D or 3D (with or without alpha channel).
        color (TYPE_PIXEL): Value to fill the mask (default is 255).

    Returns:
        TYPE_IMAGE: Mask of the image, either the alpha channel or a full mask of the given color.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        return image[..., 3]

    h, w = image.shape[:2]
    return np.ones((h, w), dtype=np.uint8) * color

def image_mask_add(image:TYPE_IMAGE, mask:TYPE_IMAGE=None, alpha:float=255) -> TYPE_IMAGE:
    """Put custom mask into an image. If there is no mask, alpha is applied.
    Images are expanded to 4 channels.
    Existing 4 channel images with no mask input just return themselves.
    """
    image = image_convert(image, 4)
    mask = image_mask(image, alpha) if mask is None else image_convert(mask, 1)
    image[..., 3] = mask if mask.ndim == 2 else mask[:, :, 0]
    return image

def image_matte(image: TYPE_IMAGE, color: TYPE_iRGBA=(0,0,0,255), width: int=None, height: int=None) -> TYPE_IMAGE:
    """
    Puts an RGBA image atop a colored matte expanding or clipping the image if requested.

    Args:
        image (TYPE_IMAGE): The input RGBA image.
        color (TYPE_iRGBA): The color of the matte as a tuple (R, G, B, A).
        width (int, optional): The width of the matte. Defaults to the image width.
        height (int, optional): The height of the matte. Defaults to the image height.

    Returns:
        TYPE_IMAGE: Composited RGBA image on a matte with original alpha channel.
    """

    #if image.ndim != 4 or image.shape[2] != 4:
    #    return image

    # Determine the dimensions of the image and the matte
    image_height, image_width = image.shape[:2]
    width = width or image_width
    height = height or image_height

    # Create a solid matte with the specified color
    matte = np.full((height, width, 4), color, dtype=np.uint8)

    # Extract the alpha channel from the image
    alpha = None
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3] / 255.0

    # Calculate the center position for the image on the matte
    x_offset = (width - image_width) // 2
    y_offset = (height - image_height) // 2

    if alpha is not None:
        # Place the image onto the matte using the alpha channel for blending
        for c in range(0, 3):
            matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] = \
                (1 - alpha) * matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] + \
                alpha * image[:, :, c]

        # Set the alpha channel of the matte to the maximum of the matte's and the image's alpha
        matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, 3] = \
            np.maximum(matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, 3], image[:, :, 3])
    else:
        image = image[y_offset:y_offset + image_height, x_offset:x_offset + image_width, :]
    return matte

# ==============================================================================
# === CLASS ===
# ==============================================================================

class JOVImageNode(JOVBaseNode):
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("RGBA", "RGB", "MASK")
    OUTPUT_TOOLTIPS = (
        "Full channel [RGBA] image. If there is an alpha, the image will be masked out with it when using this output.",
        "Three channel [RGB] image. There will be no alpha.",
        "Single channel mask output."
    )

class StreamNodeHeader(JOVImageNode):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()

        return deep_merge(d, {
            "optional": {
                "FPS": ("INT", {"min": 1, "max": 60, "default": 30, "tooltip": "Framerate to attempt when capturing"}),
                "PAUSE": ("BOOLEAN", {"default": False, "tooltip": "If the stream should hold (pause) it's frame capture"}),
                "BATCH": ("INT", {"default": 1, "tooltip": "Number of frames wanted at the Framerate in FPS"}),
                "TIMEOUT": ("INT", {"default": 5, "min": 1, "max": 30, "tooltip": "How long to wait before failing if no frames are being captured"})
            }
        })

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        a = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 4), dtype=torch.uint8, device="cpu")
        e = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8, device="cpu")
        m = torch.ones((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 1), dtype=torch.uint8, device="cpu")
        self.empty = (a, e, m,)
        self.last = [(a, e, m,)]
