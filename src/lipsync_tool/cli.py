"""Command line interface for the lip-sync generation pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import LipSyncPipeline, Wav2LipResources, ensure_dependencies

DEFAULT_PACKAGES = [
    "torch",
    "numpy",
    "opencv-python",
    "ffmpeg-python",
    "librosa",
    "tqdm",
    "numba",
    "scipy",
    "audioread",
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
        default=Path(".cache/wav2lip"),
        help="Directory to cache the Wav2Lip repository and checkpoints",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frames per second for the generated video",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for the Wav2Lip inference",
    )
    parser.add_argument(
        "--nosmooth",
        action="store_true",
        help="Disable Wav2Lip's smoothing post-processing",
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

    resources = Wav2LipResources(root_dir=args.cache_dir)
    pipeline = LipSyncPipeline(
        resources=resources,
        fps=args.fps,
        upscale_to_1080p=args.upscale,
        keep_intermediate=args.keep_temp,
    )

    pipeline.run(
        image_path=args.image,
        audio_path=args.audio,
        output_path=args.output,
        batch_size=args.batch_size,
        nosmooth=args.nosmooth,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
