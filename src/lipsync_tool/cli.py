"""Command line interface for the lip-sync generation pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import LipSyncPipeline, SadTalkerResources, ensure_dependencies

DEFAULT_PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "opencv-python",
    "ffmpeg-python",
    "librosa",
    "tqdm",
    "scipy",
    "audioread",
    "face-alignment",
    "lpips",
    "gfpgan",
    "gdown",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the input portrait image")
    parser.add_argument("audio", type=Path, help="Path to the input speech audio")
    parser.add_argument(
        "output",
        type=Path,
        help="Destination path for the generated lip-synced video",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/sadtalker"),
        help="Directory to cache the SadTalker repository and checkpoints",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frames per second for the generated video",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        choices=[256, 512],
        help="Base generation resolution before optional 1080p upscaling",
    )
    parser.add_argument(
        "--preprocess",
        default="full",
        choices=["full", "crop", "extreme_crop"],
        help="Preprocessing strategy passed to SadTalker",
    )
    parser.add_argument(
        "--expression-scale",
        type=float,
        default=1.0,
        help="Expression scale factor for SadTalker (higher yields bigger motion)",
    )
    parser.add_argument(
        "--no-still",
        dest="still_mode",
        action="store_false",
        default=True,
        help="Disable SadTalker's still mode for additional head motion",
    )
    parser.add_argument(
        "--enhancer",
        default=None,
        help="Optional SadTalker enhancer flag (e.g., gfpgan)",
    )
    parser.add_argument(
        "--no-upscale",
        dest="upscale",
        action="store_false",
        help="Skip the final 1080p upscaling stage",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate files generated during processing",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Automatically install Python dependencies if they are missing",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for the pipeline",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    if args.install_deps:
        ensure_dependencies(DEFAULT_PACKAGES)

    resources = SadTalkerResources(root_dir=args.cache_dir)
    pipeline = LipSyncPipeline(
        resources=resources,
        fps=args.fps,
        upscale_to_1080p=args.upscale,
        keep_intermediate=args.keep_temp,
        resolution=args.resolution,
    )

    pipeline.run(
        image_path=args.image,
        audio_path=args.audio,
        output_path=args.output,
        preprocess=args.preprocess,
        expression_scale=args.expression_scale,
        still_mode=args.still_mode,
        enhancer=args.enhancer,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
