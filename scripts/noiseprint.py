"""
Minimal Noiseprint extraction CLI for the video tutorial prototype.
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Sequence, Tuple, TypeVar

import cv2
import numpy as np
import torch
from matplotlib import cm
from torch import nn

LOGGER = logging.getLogger("noiseprint.cli")
MAX_SIDE = 2048
T = TypeVar("T", bound=np.ndarray)


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def validate_image(img: np.ndarray) -> None:
    if not isinstance(img, np.ndarray):
        raise TypeError("Expected numpy.ndarray for image input.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Images must be BGR arrays with three channels.")


def to_gray(img: np.ndarray) -> np.ndarray:
    validate_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def highpass_residual(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer.")
    gray = to_gray(img)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    return (gray - blurred).astype(np.float32)


def normalize_map(m: np.ndarray) -> np.ndarray:
    if not isinstance(m, np.ndarray) or m.ndim not in (2, 3):
        raise ValueError("Maps must be numpy arrays with 2 or 3 dimensions.")
    arr = m.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if np.isclose(max_val - min_val, 0.0):
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_val) / (max_val - min_val)


def gather_inputs(pattern: str) -> List[Path]:
    candidate = Path(pattern)
    if candidate.exists():
        return [candidate]
    matches = [Path(p) for p in glob.glob(pattern)]
    return sorted(p for p in matches if p.is_file())


def resize_if_needed(img: np.ndarray, max_side: Optional[int]) -> Tuple[np.ndarray, float]:
    if not max_side or max_side <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img, 1.0
    scale = max_side / float(longest)
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def to_colormap(residual: np.ndarray, cmap: str = "jet") -> np.ndarray:
    if residual.ndim != 2:
        raise ValueError("Residual must be a 2D array.")
    normalized = normalize_map(residual)
    heatmap = cm.get_cmap(cmap)(normalized)[..., :3]
    return (heatmap * 255).astype(np.uint8)


def save_heatmap(residual: np.ndarray, out_path: Path, cmap: str = "jet") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_rgb = to_colormap(residual, cmap=cmap)
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(out_path), heatmap_bgr):
        raise IOError(f"Failed to save heatmap to {out_path}")


def save_overlay(
    image_bgr: np.ndarray,
    residual: np.ndarray,
    out_path: Path,
    alpha: float = 0.6,
    cmap: str = "jet",
) -> None:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be within [0, 1].")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_rgb = to_colormap(residual, cmap=cmap)
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    if heatmap_bgr.shape[:2] != image_bgr.shape[:2]:
        heatmap_bgr = cv2.resize(
            heatmap_bgr,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    image_float = image_bgr.astype(np.float32)
    heatmap_float = heatmap_bgr.astype(np.float32)
    overlay = cv2.addWeighted(image_float, 1.0 - alpha, heatmap_float, alpha, 0)
    overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(out_path), overlay_uint8):
        raise IOError(f"Failed to save overlay to {out_path}")


# --------------------------------------------------------------------------- #
# Model placeholder
# --------------------------------------------------------------------------- #


class NoiseprintCNN(nn.Module):
    """
    Shallow convolutional model acting as a placeholder for the real Noiseprint network.
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
        features = self.features(x)
        return self.head(features)


def load_model(weights_path: Path) -> Optional[nn.Module]:
    if not weights_path.exists():
        LOGGER.info("Weights not found at %s. Using fallback residual.", weights_path)
        return None
    model = NoiseprintCNN()
    try:
        checkpoint = torch.load(weights_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to load weights from %s: %s", weights_path, exc)
        return None
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    try:
        model.load_state_dict(checkpoint)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Weights at %s are incompatible: %s", weights_path, exc)
        return None
    model.eval()
    LOGGER.info("Loaded Noiseprint weights from %s.", weights_path)
    return model


def compute_noiseprint(image_bgr: np.ndarray, model: Optional[nn.Module]) -> np.ndarray:
    validate_image(image_bgr)
    working, _ = resize_if_needed(image_bgr, MAX_SIDE)
    if model is None:
        residual = highpass_residual(working)
        return normalize_map(residual)

    tensor = (
        torch.from_numpy(working.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    device = next(model.parameters()).device  # type: ignore[arg-type]
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
    if output.ndim == 4:
        output = output.squeeze(0)
    if output.ndim == 3 and output.shape[0] == 1:
        output = output.squeeze(0)
    noise_map = output.detach().cpu().numpy().astype(np.float32)
    return normalize_map(noise_map)


# --------------------------------------------------------------------------- #
# CLI workflow
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Noiseprint-like residuals.")
    parser.add_argument(
        "--input",
        required=True,
        help="File path or glob pattern (e.g. data/input/*.jpg).",
    )
    parser.add_argument(
        "--weights",
        default="weights/noiseprint.pth",
        help="Optional path to trained weights.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory for generated results.",
    )
    parser.add_argument(
        "--resize-max",
        type=int,
        default=None,
        help="Resize the longest side before processing.",
    )
    parser.add_argument(
        "--save-heatmap",
        action="store_true",
        help="Persist heatmap images.",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Persist overlay images.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read image at {path}")
    return img


def process_file(
    path: Path,
    output_dir: Path,
    model: Optional[nn.Module],
    save_heatmap_flag: bool,
    save_overlay_flag: bool,
    resize_max: Optional[int],
) -> None:
    start = perf_counter()
    image = read_image(path)
    resized, scale = resize_if_needed(image, resize_max)
    residual = compute_noiseprint(resized, model=model)

    base_name = path.stem
    if save_heatmap_flag:
        heatmap_path = output_dir / f"{base_name}_noiseprint.png"
        save_heatmap(residual, heatmap_path)
    if save_overlay_flag:
        overlay_input = resized if scale != 1.0 else image
        overlay_path = output_dir / f"{base_name}_overlay.png"
        save_overlay(overlay_input, residual, overlay_path)

    elapsed = perf_counter() - start
    LOGGER.info("Processed %s in %.2fs.", path.name, elapsed)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    input_paths = gather_inputs(args.input)
    if not input_paths:
        LOGGER.error("No files matched the pattern: %s", args.input)
        return 1

    weights_path = Path(args.weights)
    model = load_model(weights_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = perf_counter()
    success = True
    for path in input_paths:
        try:
            process_file(
                path=path,
                output_dir=output_dir,
                model=model,
                save_heatmap_flag=args.save_heatmap,
                save_overlay_flag=args.save_overlay,
                resize_max=args.resize_max,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed processing %s: %s", path, exc)
            success = False
    total_elapsed = perf_counter() - total_start
    LOGGER.info("Completed processing in %.2fs.", total_elapsed)
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())

