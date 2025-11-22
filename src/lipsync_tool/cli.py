"""Command line interface for the lip-sync generation pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import (
    EngineName,
    LipSyncPipeline,
    SadTalkerResources,
    SadTalkerSettings,
    Wav2LipResources,
    Wav2LipSettings,
    ensure_dependencies,
)

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
    "imageio",
    "requests",
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
        "--engine",
        default="sadtalker",
        choices=["sadtalker", "wav2lip"],
        help="Select which backend to use for lip-sync generation",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory to cache repositories and checkpoints (defaults per engine)",
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
    _add_sadtalker_group(parser)
    _add_wav2lip_group(parser)
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

    engine: EngineName = args.engine
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = Path(".cache/sadtalker" if engine == "sadtalker" else ".cache/wav2lip")

    if engine == "sadtalker":
        pipeline = LipSyncPipeline(
            engine="sadtalker",
            fps=args.fps,
            upscale_to_1080p=args.upscale,
            keep_intermediate=args.keep_temp,
            resolution=args.resolution,
            sadtalker_resources=SadTalkerResources(root_dir=cache_dir),
            sadtalker_settings=SadTalkerSettings(
                preprocess=args.preprocess,
                expression_scale=args.expression_scale,
                still_mode=args.still_mode,
                enhancer=args.enhancer,
            ),
        )
    else:
        wav2lip_settings = Wav2LipSettings(
            pads=tuple(args.wav2lip_pads),
            static=args.wav2lip_static,
            nosmooth=args.wav2lip_nosmooth,
            wav2lip_batch_size=args.wav2lip_batch_size,
            face_det_batch_size=args.wav2lip_face_det_batch_size,
            resize_factor=args.wav2lip_resize_factor,
            crop=tuple(args.wav2lip_crop) if args.wav2lip_crop else None,
        )
        pipeline = LipSyncPipeline(
            engine="wav2lip",
            fps=args.fps,
            upscale_to_1080p=args.upscale,
            keep_intermediate=args.keep_temp,
            resolution=args.resolution,
            wav2lip_resources=Wav2LipResources(root_dir=cache_dir),
            wav2lip_settings=wav2lip_settings,
        )

    pipeline.run(
        image_path=args.image,
        audio_path=args.audio,
        output_path=args.output,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


def _add_sadtalker_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("SadTalker options")
    group.add_argument(
        "--preprocess",
        default="full",
        choices=["full", "crop", "extreme_crop"],
        help="Preprocessing strategy passed to SadTalker",
    )
    group.add_argument(
        "--expression-scale",
        type=float,
        default=1.0,
        help="Expression scale factor for SadTalker (higher yields bigger motion)",
    )
    group.add_argument(
        "--no-still",
        dest="still_mode",
        action="store_false",
        default=True,
        help="Disable SadTalker's still mode for additional head motion",
    )
    group.add_argument(
        "--enhancer",
        default=None,
        help="Optional SadTalker enhancer flag (e.g., gfpgan)",
    )


def _add_wav2lip_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Wav2Lip options")
    group.add_argument(
        "--wav2lip-static",
        dest="wav2lip_static",
        action="store_true",
        default=True,
        help="Force static mode (single image) for Wav2Lip",
    )
    group.add_argument(
        "--wav2lip-no-static",
        dest="wav2lip_static",
        action="store_false",
        help="Allow dynamic face inputs for Wav2Lip",
    )
    group.add_argument(
        "--wav2lip-nosmooth",
        action="store_true",
        help="Disable smoothing in Wav2Lip post-processing",
    )
    group.add_argument(
        "--wav2lip-batch-size",
        type=int,
        default=None,
        help="Batch size for the Wav2Lip model",
    )
    group.add_argument(
        "--wav2lip-face-det-batch-size",
        type=int,
        default=None,
        help="Batch size for the face detector",
    )
    group.add_argument(
        "--wav2lip-resize-factor",
        type=float,
        default=1.0,
        help="Resize factor applied before running Wav2Lip",
    )
    group.add_argument(
        "--wav2lip-crop",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Crop rectangle for Wav2Lip (x1 y1 x2 y2)",
    )
    group.add_argument(
        "--wav2lip-pads",
        nargs=4,
        type=int,
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        default=(0, 10, 0, 0),
        help="Padding around the detected face for Wav2Lip",
    )
