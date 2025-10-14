"""
Inference utilities for the Noiseprint prototype.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn

from .filters import highpass_residual, normalize_map

LOGGER = logging.getLogger(__name__)
_MAX_SIDE = 2048


class NoiseprintCNN(nn.Module):
    """
    Lightweight placeholder CNN meant to mimic a noiseprint extractor.

    The architecture is intentionally shallow; it is **not** a faithful
    reproduction of the original Noiseprint model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the placeholder noiseprint map.
        """
        features = self.features(x)
        return self.head(features)


def _maybe_resize(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Optionally resize the image so the largest side is <= `_MAX_SIDE`.

    Returns the possibly resized image and the original spatial shape.
    """
    original_shape = (img.shape[0], img.shape[1])
    h, w = original_shape
    max_side = max(original_shape)
    if max_side <= _MAX_SIDE:
        return img, original_shape

    scale = _MAX_SIDE / float(max_side)
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    LOGGER.info(
        "Downscaled image from %s to %s to satisfy max side constraint.",
        original_shape,
        resized.shape[:2],
    )
    return resized, original_shape


def load_model(weights_path: str = "weights/noiseprint.pth") -> Optional[torch.nn.Module]:
    """
    Load Noiseprint weights when available.

    Parameters
    ----------
    weights_path
        Path pointing to ``noiseprint.pth`` weights.

    Returns
    -------
    Optional[torch.nn.Module]
        Instantiated model set to evaluation mode, or ``None`` if weights
        are missing or could not be loaded.
    """
    path = Path(weights_path)
    if not path.exists():
        LOGGER.info("Weights not found at %s. Falling back to handcrafted residual.", path)
        return None

    model = NoiseprintCNN()
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - broad to keep CLI resilient
        LOGGER.error("Failed to load weights from %s: %s", path, exc)
        return None

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    try:
        model.load_state_dict(checkpoint)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("State dict from %s is incompatible: %s", path, exc)
        return None

    model.eval()
    LOGGER.info("Loaded Noiseprint weights from %s.", path)
    return model


def compute_noiseprint(
    img_bgr: np.ndarray,
    model: Optional[torch.nn.Module] = None,
) -> np.ndarray:
    """
    Compute a noiseprint map for the provided BGR image.

    Parameters
    ----------
    img_bgr
        Input image in BGR format (``uint8`` or ``float``).
    model
        Optional neural network. When not provided, a handcrafted residual
        is used as fallback.

    Returns
    -------
    np.ndarray
        Normalized noiseprint map in ``float32`` with shape ``HxW``.
    """
    if not isinstance(img_bgr, np.ndarray):
        raise TypeError("`img_bgr` must be a numpy.ndarray.")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected BGR image with 3 channels, received shape {img_bgr.shape!r}."
        )

    working_img, original_shape = _maybe_resize(img_bgr)

    if model is None:
        LOGGER.debug("Using high-pass residual fallback.")
        residual = highpass_residual(working_img)
        normalized = normalize_map(residual)
    else:
        LOGGER.debug("Running CNN-based noiseprint extraction.")
        tensor = (
            torch.from_numpy(working_img.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        model_device = next(model.parameters()).device  # type: ignore[call-overload]
        tensor = tensor.to(model_device)
        with torch.no_grad():
            output = model(tensor)
        if output.ndim == 4 and output.shape[1] == 1:
            output = output.squeeze(0).squeeze(0)
        elif output.ndim == 3:
            output = output.squeeze(0)
        noise_map = output.detach().cpu().numpy().astype(np.float32)
        normalized = normalize_map(noise_map)

    if working_img.shape[:2] != original_shape:
        normalized = cv2.resize(
            normalized,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return normalized.astype(np.float32)

