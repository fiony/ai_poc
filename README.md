# AI PoC – Lip-sync Generator

This proof-of-concept packages an automated workflow for building a lip-sync
video from a still portrait and a speech audio track. It wraps the
[SadTalker](https://github.com/OpenTalker/SadTalker) project, handling
repository cloning, checkpoint download and HD upscaling so you can focus on
supplying the inputs.

## Features

- **Automated resource management** – the official SadTalker repository and the
  pretrained checkpoints are fetched on demand through the upstream helper
  script.
- **Command line interface** – run a single command to create a lip-synced clip
  from an image and an audio file.
- **1080p output** – the resulting video is upscaled to full HD using `ffmpeg`
  with high-quality bicubic filtering.
- **Runtime dependency bootstrap** – optional `--install-deps` flag installs the
  required Python packages if they are missing in the current environment.

## Installation

> **Prerequisites**
>
> - Python 3.10+
> - `ffmpeg` command line tool available on your `PATH`
> - Sufficient GPU resources are recommended for real-time execution, although
>   the pipeline can run on CPU with reduced performance.

Clone the repository and install dependencies:

```bash
pip install -e .
```

If you prefer not to install globally, you can run the CLI via the module entry
point after installing dependencies manually or by using the runtime bootstrap
flag as shown below.

## Usage

```bash
lipsync-tool /path/to/portrait.jpg /path/to/speech.wav output.mp4 \
  --install-deps \
  --cache-dir ~/.cache/sadtalker \
  --fps 25 \
  --resolution 512 \
  --preprocess full
```

The first run clones the upstream SadTalker repository and downloads the
pretrained checkpoints into the specified cache directory. Subsequent runs reuse
these assets.

### Options

- `--resolution`: choose between the SadTalker 256 or 512 pipelines.
- `--preprocess`: select the upstream preprocessing strategy (`full`, `crop`,
  or `extreme_crop`).
- `--expression-scale`: increase or decrease facial motion intensity.
- `--no-still`: allow more head movement by disabling SadTalker's still mode.
- `--enhancer`: forward an enhancer flag to SadTalker (e.g., `gfpgan`).
- `--no-upscale`: skip the 1080p upscaling stage.
- `--keep-temp`: retain intermediate files for debugging.
- `--log-level`: choose logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

## How it works

1. The CLI validates inputs and optionally installs Python dependencies.
2. `SadTalkerResources` clones the official SadTalker repository and invokes
   the provided model download helper to populate checkpoints.
3. `LipSyncPipeline` launches the upstream `inference.py` script to produce a
   preliminary MP4 clip inside a temporary directory.
4. The generated clip is upscaled to 1080p using `ffmpeg` with bicubic
   resampling while preserving the audio track.

## Extending the PoC

- Integrate a frontend (Streamlit, Gradio, FastAPI) that calls the CLI module.
- Swap the final upscaling step with a learned super-resolution model such as
  Real-ESRGAN for higher fidelity.
- Add voice activity detection or automatic audio trimming before inference.
- Build a natural-language assistant (LLM) that assembles calls to this pipeline
  based on user prompts for a fully conversational workflow.

## Research on Alternative Foundations

If you need to evaluate other open-source projects before building on top of
this proof-of-concept, consult the curated survey in
[`docs/lipsync_repo_research.md`](docs/lipsync_repo_research.md). The document
summarizes popular lip-sync pipelines (SadTalker, Wav2Lip + Real-ESRGAN, EMO,
PIRenderer/Audio2Head), highlights which requirements they satisfy, and notes
where additional engineering—such as LLM integration or 1080p upscaling—is
required. This research reflects the current landscape: no single repository
ships with an open-source LLM *and* native 1080p output, but SadTalker provides
the most balanced starting point when paired with super-resolution and an LLM
orchestration layer.

## Troubleshooting

- Ensure `ffmpeg` is installed and callable from the shell.
- The SadTalker inference step benefits greatly from CUDA acceleration; CPU
  execution can take several minutes for a short clip.
- If the checkpoint download fails, delete the cache directory and rerun with
  `--install-deps` to retry.

## License

This repository contains original glue code licensed under the MIT license. It
relies on the upstream SadTalker project, which is distributed under its own
license. Consult their repository for details before shipping a product.
