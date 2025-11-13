"""High level pipeline for generating lip-synced video clips.

This module orchestrates the preparation of the official SadTalker repository,
its pretrained checkpoints, and the execution pipeline required to produce an
HD (1080p) video that syncs a still portrait with an audio clip.

The implementation is intentionally light-weight and focuses on automation so
that a user only needs to provide the face image and audio source. Heavy model
artifacts are downloaded on demand from their public locations using the
SadTalker helper scripts.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import logging


LOGGER = logging.getLogger(__name__)


@dataclass
class SadTalkerResources:
    """Utility class that prepares the SadTalker repository and checkpoints."""

    root_dir: Path = Path(".cache/sadtalker")
    repo_url: str = "https://github.com/OpenTalker/SadTalker.git"
    models_script: str = "scripts/download_models.py"
    checkpoints_subdir: str = "checkpoints"

    def repo_path(self) -> Path:
        return self.root_dir / "repo"

    def checkpoints_path(self) -> Path:
        return self.repo_path() / self.checkpoints_subdir

    def ensure(self, *, resolution: int = 512) -> None:
        """Ensure that both repository and checkpoints are present on disk."""

        self.root_dir.mkdir(parents=True, exist_ok=True)
        if not self.repo_path().exists():
            LOGGER.info("Cloning SadTalker repository into %s", self.repo_path())
            self._git_clone()
        else:
            LOGGER.debug("SadTalker repository already exists at %s", self.repo_path())

        if not any(self.checkpoints_path().rglob("*")):
            LOGGER.info("Downloading SadTalker checkpoints via helper script")
            self._download_checkpoints(resolution=resolution)
        else:
            LOGGER.debug(
                "SadTalker checkpoints already available in %s",
                self.checkpoints_path(),
            )

    # ------------------------------------------------------------------
    def _git_clone(self) -> None:
        subprocess.run(
            ["git", "clone", "--depth", "1", self.repo_url, str(self.repo_path())],
            check=True,
        )

    def _download_checkpoints(self, *, resolution: int) -> None:
        repo = self.repo_path()
        script = repo / self.models_script
        if not script.exists():
            raise FileNotFoundError(
                "SadTalker download script not found at %s" % script
            )
        command = [
            sys.executable,
            str(script),
            "--model_folder",
            str(self.checkpoints_path()),
            "--resolution",
            str(resolution),
        ]
        LOGGER.debug("Executing SadTalker model download: %s", " ".join(command))
        subprocess.run(command, cwd=repo, check=True)


@dataclass
class LipSyncPipeline:
    """Generate a lip-synced video using a still image and an audio clip."""

    resources: SadTalkerResources
    fps: int = 25
    upscale_to_1080p: bool = True
    keep_intermediate: bool = False
    resolution: int = 512

    def run(
        self,
        image_path: Path | str,
        audio_path: Path | str,
        output_path: Path | str,
        *,
        preprocess: str = "full",
        expression_scale: float = 1.0,
        still_mode: bool = True,
        enhancer: str | None = None,
    ) -> Path:
        """Run the pipeline and return the resulting video path."""

        image_path = Path(image_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.resources.ensure(resolution=self.resolution)

        repo = self.resources.repo_path()
        temp_output_dir = output_path.parent / f"{output_path.stem}_sadtalker"
        if not self.keep_intermediate and temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(repo / "inference.py"),
            "--driven_audio",
            str(audio_path),
            "--source_image",
            str(image_path),
            "--checkpoint_dir",
            str(self.resources.checkpoints_path()),
            "--result_dir",
            str(temp_output_dir),
            "--preprocess",
            preprocess,
            "--expression_scale",
            str(expression_scale),
            "--size",
            str(self.resolution),
            "--fps",
            str(self.fps),
        ]
        if still_mode:
            command.append("--still")
        if enhancer:
            command.extend(["--enhancer", enhancer])

        LOGGER.info("Executing SadTalker inference: %s", " ".join(command))
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(repo))
        subprocess.run(command, cwd=repo, env=env, check=True)

        temp_output = self._select_generated_video(temp_output_dir)
        final_output = output_path
        if self.upscale_to_1080p:
            self._upscale_to_1080p(temp_output, final_output)
        else:
            shutil.move(temp_output, final_output)

        if not self.keep_intermediate:
            shutil.rmtree(temp_output_dir, ignore_errors=True)

        return final_output

    # ------------------------------------------------------------------
    def _select_generated_video(self, directory: Path) -> Path:
        candidates = sorted(directory.glob("*.mp4"))
        if not candidates:
            raise FileNotFoundError(
                f"No SadTalker outputs found in {directory}. Expected an mp4 file."
            )
        if len(candidates) > 1:
            LOGGER.warning(
                "Multiple SadTalker outputs detected; selecting %s", candidates[0]
            )
        return candidates[0]

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


__all__ = ["LipSyncPipeline", "SadTalkerResources", "ensure_dependencies"]
