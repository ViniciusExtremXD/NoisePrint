"""
Utility filters and preprocessing helpers for the Noiseprint prototype.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def _validate_image_array(img: np.ndarray) -> Tuple[int, ...]:
    """
    Validate that the provided array is a supported image format.

    Returns
    -------
    Tuple[int, ...]
        The input image shape if validation succeeds.

    Raises
    ------
    TypeError
        If the array is not a NumPy array.
    ValueError
        If the array has an invalid number of dimensions.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("Expected `img` to be a numpy.ndarray.")
    if img.ndim not in (2, 3):
        raise ValueError(
            f"Expected image with 2 or 3 dimensions, received shape {img.shape!r}."
        )
    return img.shape


def to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale while preserving the spatial dimensions.

    Parameters
    ----------
    img
        Input image in BGR, RGB, or grayscale format.

    Returns
    -------
    np.ndarray
        Grayscale image with dtype ``float32`` in range ``[0, 255]``.
    """
    shape = _validate_image_array(img)
    if len(shape) == 2:
        LOGGER.debug("Input already grayscale with shape %s.", shape)
        return img.astype(np.float32)

    if shape[2] != 3:
        raise ValueError(
            f"Expected 3-channel image, but received shape {shape!r}."
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    LOGGER.debug("Converted image to grayscale with shape %s.", gray.shape)
    return gray.astype(np.float32)


def highpass_residual(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Compute a simple high-pass residual using a Gaussian blur.

    Parameters
    ----------
    img
        Input image in grayscale or BGR format.
    ksize
        Kernel size for the Gaussian blur. Must be a positive odd integer.

    Returns
    -------
    np.ndarray
        Residual map emphasizing high-frequency details (float32).
    """
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("`ksize` must be a positive odd integer.")

    gray = to_gray(img)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    residual = gray - blurred
    LOGGER.debug(
        "Computed high-pass residual with kernel size %s and shape %s.",
        ksize,
        residual.shape,
    )
    return residual.astype(np.float32)


def normalize_map(m: np.ndarray) -> np.ndarray:
    """
    Normalize a residual or heatmap to the [0, 1] range.

    Parameters
    ----------
    m
        Input map to be normalized.

    Returns
    -------
    np.ndarray
        Normalized map in ``float32`` with values between 0 and 1.
    """
    _validate_image_array(m)
    arr = m.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if np.isclose(max_val - min_val, 0.0):
        LOGGER.debug("Map has near-constant values; returning zeros.")
        return np.zeros_like(arr, dtype=np.float32)

    normalized = (arr - min_val) / (max_val - min_val)
    LOGGER.debug(
        "Normalized map with min %.4f, max %.4f, resulting dtype %s.",
        min_val,
        max_val,
        normalized.dtype,
    )
    return normalized

