"""
Visualization utilities for Noiseprint residuals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from matplotlib import cm

from .filters import normalize_map

LOGGER = logging.getLogger(__name__)


def _prepare_output_path(out_path: str) -> Path:
    """
    Ensure the directory for the output path exists and return the path.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _to_colormap(residual: np.ndarray, cmap: str) -> np.ndarray:
    """
    Convert a residual map to an RGB heatmap array using matplotlib.
    """
    if residual.ndim != 2:
        raise ValueError("Residual map must be a 2D array.")
    normalized = normalize_map(residual)
    colormap = cm.get_cmap(cmap)
    heatmap = colormap(normalized)[:, :, :3]
    heatmap_rgb = (heatmap * 255).astype(np.uint8)
    return heatmap_rgb


def save_heatmap(residual: np.ndarray, out_path: str, cmap: str = "jet") -> None:
    """
    Save a residual as a colored heatmap image.
    """
    path = _prepare_output_path(out_path)
    heatmap_rgb = _to_colormap(residual, cmap)
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), heatmap_bgr):
        raise IOError(f"Failed to save heatmap to {path}")
    LOGGER.info("Saved heatmap to %s.", path)


def save_overlay(
    img_bgr: np.ndarray,
    residual: np.ndarray,
    out_path: str,
    alpha: float = 0.6,
    cmap: str = "jet",
) -> None:
    """
    Overlay a residual heatmap on top of the original image.
    """
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("`img_bgr` must be a 3-channel BGR image.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("`alpha` must be within the [0, 1] interval.")

    path = _prepare_output_path(out_path)
    heatmap_rgb = _to_colormap(residual, cmap)
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)

    if heatmap_bgr.shape[:2] != img_bgr.shape[:2]:
        heatmap_bgr = cv2.resize(
            heatmap_bgr,
            (img_bgr.shape[1], img_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    img_float = img_bgr.astype(np.float32)
    heatmap_float = heatmap_bgr.astype(np.float32)
    overlay = cv2.addWeighted(img_float, 1.0 - alpha, heatmap_float, alpha, 0)
    overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)

    if not cv2.imwrite(str(path), overlay_uint8):
        raise IOError(f"Failed to save overlay to {path}")
    LOGGER.info("Saved overlay to %s.", path)

