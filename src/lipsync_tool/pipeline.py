"""High level pipeline for generating lip-synced video clips.

This module now supports both SadTalker and Wav2Lip backends. Each engine has a
lightweight resource manager that clones the upstream repository, downloads the
required checkpoints, and exposes a unified interface for the CLI.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence


LOGGER = logging.getLogger(__name__)

EngineName = Literal["sadtalker", "wav2lip"]


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


WAV2LIP_CHECKPOINT_URL = (
    "https://drive.google.com/uc?id=1cwRmZm4EUdS6WZfiP2spXK7pbybY2coh"
)
S3FD_CHECKPOINT_URL = (
    "https://drive.google.com/uc?id=1ER8mWj4lfUYSS-uMndZX45phBORp_ZO7"
)


@dataclass
class Wav2LipResources:
    """Prepare the Wav2Lip repository and required checkpoints."""

    root_dir: Path = Path(".cache/wav2lip")
    repo_url: str = "https://github.com/Rudrabha/Wav2Lip.git"
    checkpoints_subdir: str = "checkpoints"
    wav2lip_filename: str = "wav2lip.pth"
    face_detector_filename: str = "s3fd.pth"

    def repo_path(self) -> Path:
        return self.root_dir / "repo"

    def checkpoints_path(self) -> Path:
        return self.root_dir / self.checkpoints_subdir

    def wav2lip_checkpoint(self) -> Path:
        return self.checkpoints_path() / self.wav2lip_filename

    def face_detector_checkpoint(self) -> Path:
        return self.checkpoints_path() / self.face_detector_filename

    def ensure(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if not self.repo_path().exists():
            LOGGER.info("Cloning Wav2Lip repository into %s", self.repo_path())
            self._git_clone()
        else:
            LOGGER.debug("Wav2Lip repository already exists at %s", self.repo_path())

        self.checkpoints_path().mkdir(parents=True, exist_ok=True)
        if not self.wav2lip_checkpoint().exists():
            LOGGER.info("Downloading Wav2Lip checkpoint")
            self._download(WAV2LIP_CHECKPOINT_URL, self.wav2lip_checkpoint())
        else:
            LOGGER.debug("Wav2Lip checkpoint present at %s", self.wav2lip_checkpoint())

        if not self.face_detector_checkpoint().exists():
            LOGGER.info("Downloading S3FD face detector checkpoint")
            self._download(S3FD_CHECKPOINT_URL, self.face_detector_checkpoint())
        else:
            LOGGER.debug(
                "S3FD face detector checkpoint present at %s",
                self.face_detector_checkpoint(),
            )
        self._sync_detector_into_repo()

    def _git_clone(self) -> None:
        subprocess.run(
            ["git", "clone", "--depth", "1", self.repo_url, str(self.repo_path())],
            check=True,
        )

    def _download(self, url: str, destination: Path) -> None:
        try:
            import gdown
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "gdown is required to download Wav2Lip resources"
            ) from exc

        LOGGER.debug("Downloading %s to %s", url, destination)
        gdown.download(url, str(destination), quiet=False)

    def _sync_detector_into_repo(self) -> None:
        repo_detector = (
            self.repo_path() / "face_detection" / "detection" / "sfd" / self.face_detector_filename
        )
        repo_detector.parent.mkdir(parents=True, exist_ok=True)
        if not repo_detector.exists():
            LOGGER.debug("Copying S3FD checkpoint into repository tree at %s", repo_detector)
            shutil.copy2(self.face_detector_checkpoint(), repo_detector)


@dataclass
class SadTalkerSettings:
    preprocess: str = "full"
    expression_scale: float = 1.0
    still_mode: bool = True
    enhancer: str | None = None


@dataclass
class Wav2LipSettings:
    pads: Sequence[int] = field(default_factory=lambda: (0, 10, 0, 0))
    static: bool = True
    nosmooth: bool = False
    wav2lip_batch_size: int | None = None
    face_det_batch_size: int | None = None
    resize_factor: float = 1.0
    crop: Sequence[int] | None = None


@dataclass
class LipSyncPipeline:
    """Generate a lip-synced video using a still image and an audio clip."""

    engine: EngineName
    fps: int = 25
    upscale_to_1080p: bool = True
    keep_intermediate: bool = False
    resolution: int = 512
    sadtalker_resources: SadTalkerResources = field(default_factory=SadTalkerResources)
    wav2lip_resources: Wav2LipResources = field(default_factory=Wav2LipResources)
    sadtalker_settings: SadTalkerSettings = field(default_factory=SadTalkerSettings)
    wav2lip_settings: Wav2LipSettings = field(default_factory=Wav2LipSettings)

    def run(
        self,
        image_path: Path | str,
        audio_path: Path | str,
        output_path: Path | str,
    ) -> Path:
        """Run the pipeline and return the resulting video path."""

        image_path = Path(image_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        temp_output_dir = output_path.parent / f"{output_path.stem}_{self.engine}"
        if not self.keep_intermediate and temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        if self.engine == "sadtalker":
            temp_output = self._run_sadtalker(image_path, audio_path, temp_output_dir)
        elif self.engine == "wav2lip":
            temp_output = self._run_wav2lip(image_path, audio_path, temp_output_dir)
        else:  # pragma: no cover - safety net for incorrect configuration
            raise ValueError(f"Unsupported engine: {self.engine}")

        final_output = output_path
        if self.upscale_to_1080p:
            self._upscale_to_1080p(temp_output, final_output)
        else:
            shutil.move(temp_output, final_output)

        if not self.keep_intermediate:
            shutil.rmtree(temp_output_dir, ignore_errors=True)

        return final_output

    # ------------------------------------------------------------------
    def _run_sadtalker(self, image: Path, audio: Path, destination: Path) -> Path:
        self.sadtalker_resources.ensure(resolution=self.resolution)
        repo = self.sadtalker_resources.repo_path()
        command = [
            sys.executable,
            str(repo / "inference.py"),
            "--driven_audio",
            str(audio),
            "--source_image",
            str(image),
            "--checkpoint_dir",
            str(self.sadtalker_resources.checkpoints_path()),
            "--result_dir",
            str(destination),
            "--preprocess",
            self.sadtalker_settings.preprocess,
            "--expression_scale",
            str(self.sadtalker_settings.expression_scale),
            "--size",
            str(self.resolution),
            "--fps",
            str(self.fps),
        ]
        if self.sadtalker_settings.still_mode:
            command.append("--still")
        if self.sadtalker_settings.enhancer:
            command.extend(["--enhancer", self.sadtalker_settings.enhancer])

        LOGGER.info("Executing SadTalker inference: %s", " ".join(command))
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(repo))
        subprocess.run(command, cwd=repo, env=env, check=True)

        return self._select_generated_video(destination)

    def _run_wav2lip(self, image: Path, audio: Path, destination: Path) -> Path:
        self.wav2lip_resources.ensure()
        repo = self.wav2lip_resources.repo_path()
        output_file = destination / "result.mp4"

        command = [
            sys.executable,
            str(repo / "inference.py"),
            "--checkpoint_path",
            str(self.wav2lip_resources.wav2lip_checkpoint()),
            "--face",
            str(image),
            "--audio",
            str(audio),
            "--outfile",
            str(output_file),
            "--fps",
            str(self.fps),
            "--pads",
            *(str(v) for v in self.wav2lip_settings.pads),
        ]
        if self.wav2lip_settings.static:
            command.append("--static")
        if self.wav2lip_settings.nosmooth:
            command.append("--nosmooth")
        if self.wav2lip_settings.wav2lip_batch_size:
            command.extend(
                ["--wav2lip_batch_size", str(self.wav2lip_settings.wav2lip_batch_size)]
            )
        if self.wav2lip_settings.face_det_batch_size:
            command.extend(
                [
                    "--face_det_batch_size",
                    str(self.wav2lip_settings.face_det_batch_size),
                ]
            )
        if self.wav2lip_settings.resize_factor and self.wav2lip_settings.resize_factor != 1.0:
            command.extend(["--resize_factor", str(self.wav2lip_settings.resize_factor)])
        if self.wav2lip_settings.crop:
            command.extend(["--crop", *(str(v) for v in self.wav2lip_settings.crop)])

        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(repo))
        env.setdefault(
            "WAV2LIP_CHECKPOINT_PATH", str(self.wav2lip_resources.wav2lip_checkpoint())
        )
        env.setdefault(
            "S3FD_PATH", str(self.wav2lip_resources.face_detector_checkpoint())
        )

        LOGGER.info("Executing Wav2Lip inference: %s", " ".join(command))
        subprocess.run(command, cwd=repo, env=env, check=True)

        if not output_file.exists():
            raise FileNotFoundError(
                f"Expected Wav2Lip output at {output_file}, but the file was not created."
            )
        return output_file

    # ------------------------------------------------------------------
    def _select_generated_video(self, directory: Path) -> Path:
        candidates = sorted(directory.glob("*.mp4"))
        if not candidates:
            raise FileNotFoundError(
                f"No outputs found in {directory}. Expected an mp4 file."
            )
        if len(candidates) > 1:
            LOGGER.warning(
                "Multiple outputs detected; selecting %s", candidates[0]
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


__all__ = [
    "EngineName",
    "LipSyncPipeline",
    "SadTalkerResources",
    "SadTalkerSettings",
    "Wav2LipResources",
    "Wav2LipSettings",
    "ensure_dependencies",
]
