"""
Command-line interface to extract noiseprint residuals.
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from noiseprint import compute_noiseprint, load_model, save_heatmap, save_overlay  # noqa: E402

if TYPE_CHECKING:
    from torch import nn as nn

LOGGER = logging.getLogger("noiseprint.cli")


def configure_logging(level: str = "INFO") -> None:
    """
    Configure root logging with a simple format.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def gather_inputs(pattern: str) -> List[Path]:
    """
    Gather input files matching the provided pattern.
    """
    candidate = Path(pattern)
    if candidate.exists():
        return [candidate]

    matches = [Path(p) for p in glob.glob(pattern)]
    return sorted(match for match in matches if match.is_file())


def read_image(path: Path) -> np.ndarray:
    """
    Read an image from disk raising a descriptive error when it fails.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read image at {path}")
    return img


def resize_if_needed(img: np.ndarray, max_side: Optional[int]) -> Tuple[np.ndarray, float]:
    """
    Resize the image so that its largest side is at most `max_side`.

    Returns the resized image and the applied scale factor.
    """
    if not max_side or max_side <= 0:
        return img, 1.0

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img, 1.0

    scale = max_side / float(longest)
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    LOGGER.info(
        "Resized %s from (%d, %d) to (%d, %d).",
        "image",
        h,
        w,
        resized.shape[0],
        resized.shape[1],
    )
    return resized, scale


def process_file(
    path: Path,
    output_dir: Path,
    model: Optional["nn.Module"],
    save_heatmap_flag: bool,
    save_overlay_flag: bool,
    resize_max: Optional[int],
) -> None:
    """
    Process a single image path end-to-end.
    """
    start = perf_counter()
    img = read_image(path)
    resized_img, scale = resize_if_needed(img, resize_max)

    residual = compute_noiseprint(resized_img, model=model)

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = path.stem
    if save_heatmap_flag:
        heatmap_path = output_dir / f"{base_name}_noiseprint.png"
        save_heatmap(residual, str(heatmap_path))
    if save_overlay_flag:
        overlay_input = resized_img if scale != 1.0 else img
        overlay_path = output_dir / f"{base_name}_overlay.png"
        save_overlay(overlay_input, residual, str(overlay_path))

    elapsed = perf_counter() - start
    LOGGER.info("Processed %s in %.2fs.", path.name, elapsed)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Extract Noiseprint residual maps.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input file path or glob pattern (e.g., 'data/input/*.jpg').",
    )
    parser.add_argument(
        "--weights",
        default=str(ROOT / "weights" / "noiseprint.pth"),
        help="Optional path to a trained Noiseprint model.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "output"),
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--resize-max",
        type=int,
        default=None,
        help="Resize longest image side to this value before processing.",
    )
    parser.add_argument(
        "--save-heatmap",
        action="store_true",
        help="Persist the residual heatmap as PNG.",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Persist the residual overlaid on the source image.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry-point for the CLI.
    """
    args = parse_args(argv)
    configure_logging(args.log_level)

    input_paths = gather_inputs(args.input)

    if not input_paths:
        LOGGER.error("No files matched the pattern: %s", args.input)
        return 1

    model = load_model(args.weights)
    output_dir = Path(args.output_dir)

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
        except Exception as exc:  # noqa: BLE001 - keep batch processing resilient
            LOGGER.exception("Failed processing %s: %s", path, exc)
            success = False

    total_elapsed = perf_counter() - total_start
    LOGGER.info("Completed processing in %.2fs.", total_elapsed)
    return 0 if success else 2


if __name__ == "__main__":
    sys.exit(main())
