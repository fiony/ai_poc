"""High level pipeline for generating lip-synced video clips.

This module orchestrates the preparation of the official Wav2Lip repository,
its pretrained checkpoints, and the execution pipeline required to produce an
HD (1080p) video that syncs a still portrait with an audio clip.

The implementation is intentionally light-weight and focuses on automation so
that a user only needs to provide the face image and audio source.  Heavy model
artifacts are downloaded on demand from their public locations.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import logging


LOGGER = logging.getLogger(__name__)


@dataclass
class Wav2LipResources:
    """Utility class that prepares the Wav2Lip repository and checkpoints."""

    root_dir: Path = Path(".cache/wav2lip")
    repo_url: str = "https://github.com/Rudrabha/Wav2Lip.git"
    checkpoint_url: str = (
        "https://storage.googleapis.com/aihub-models/Wav2Lip/Wav2Lip.pth"
    )
    checkpoint_md5: str = "9bf9ce9331f3e9f0a9f4e6a1e9dd42ce"

    def repo_path(self) -> Path:
        return self.root_dir / "repo"

    def checkpoint_path(self) -> Path:
        return self.root_dir / "Wav2Lip.pth"

    def ensure(self) -> None:
        """Ensure that both repository and checkpoint are present on disk."""

        self.root_dir.mkdir(parents=True, exist_ok=True)
        if not self.repo_path().exists():
            LOGGER.info("Cloning Wav2Lip repository into %s", self.repo_path())
            self._git_clone()
        else:
            LOGGER.debug("Wav2Lip repository already exists at %s", self.repo_path())

        if not self.checkpoint_path().exists():
            LOGGER.info("Downloading Wav2Lip checkpoint to %s", self.checkpoint_path())
            self._download_checkpoint()
        else:
            LOGGER.debug(
                "Wav2Lip checkpoint already exists at %s", self.checkpoint_path()
            )

        self._validate_checkpoint()

    # ------------------------------------------------------------------
    def _git_clone(self) -> None:
        subprocess.run(
            ["git", "clone", "--depth", "1", self.repo_url, str(self.repo_path())],
            check=True,
        )

    def _download_checkpoint(self) -> None:
        import urllib.request

        with urllib.request.urlopen(self.checkpoint_url) as response, open(
            self.checkpoint_path(), "wb"
        ) as target:
            shutil.copyfileobj(response, target)

    def _validate_checkpoint(self) -> None:
        md5 = hashlib.md5()
        with open(self.checkpoint_path(), "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                md5.update(chunk)
        digest = md5.hexdigest()
        if digest != self.checkpoint_md5:
            raise ValueError(
                "Checkpoint MD5 mismatch: expected %s got %s"
                % (self.checkpoint_md5, digest)
            )


@dataclass
class LipSyncPipeline:
    """Generate a lip-synced video using a still image and an audio clip."""

    resources: Wav2LipResources
    fps: int = 25
    upscale_to_1080p: bool = True
    keep_intermediate: bool = False

    def run(
        self,
        image_path: Path | str,
        audio_path: Path | str,
        output_path: Path | str,
        *,
        batch_size: int = 128,
        pads: Sequence[int] = (0, 10, 0, 0),
        nosmooth: bool = False,
    ) -> Path:
        """Run the pipeline and return the resulting video path."""

        image_path = Path(image_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.resources.ensure()

        repo = self.resources.repo_path()
        temp_output = output_path.with_suffix(".temp.mp4")

        command = [
            sys.executable,
            str(repo / "inference.py"),
            "--checkpoint_path",
            str(self.resources.checkpoint_path()),
            "--face",
            str(image_path),
            "--audio",
            str(audio_path),
            "--outfile",
            str(temp_output),
            "--fps",
            str(self.fps),
            "--pads",
            *map(str, pads),
            "--batch_size",
            str(batch_size),
        ]
        if nosmooth:
            command.append("--nosmooth")

        LOGGER.info("Executing Wav2Lip inference: %s", " ".join(command))
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(repo))
        subprocess.run(command, cwd=repo, env=env, check=True)

        final_output = output_path
        if self.upscale_to_1080p:
            self._upscale_to_1080p(temp_output, final_output)
        else:
            shutil.move(temp_output, final_output)

        if not self.keep_intermediate and temp_output.exists():
            temp_output.unlink()

        return final_output

    # ------------------------------------------------------------------
    def _upscale_to_1080p(self, src: Path, dst: Path) -> None:
        """Use ffmpeg to upscale the generated video to 1080p."""

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vf",
            "scale=1920:1080:flags=bicubic",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "copy",
            str(dst),
        ]
        LOGGER.info("Upscaling video to 1080p with ffmpeg")
        subprocess.run(command, check=True)


def ensure_dependencies(packages: Iterable[str]) -> None:
    """Install Python dependencies at runtime if they are missing."""

    missing: list[str] = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        LOGGER.info("Installing missing dependencies: %s", ", ".join(missing))
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)


__all__ = ["LipSyncPipeline", "Wav2LipResources", "ensure_dependencies"]
