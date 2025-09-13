# Local Image Generation with FLUX and Stable Diffusion

A Python project for generating images locally using state-of-the-art diffusion models.

## Features

- **FLUX.1** support (schnell/dev variants)
- **SDXL-Turbo** for fast generation with lower VRAM
- Memory-efficient generation
- CLI interface

## Requirements

- NVIDIA GPU with 8GB+ VRAM (12GB+ recommended for FLUX)
- Python 3.11+
- CUDA-capable GPU (will run on CPU but very slowly)

## Installation

Project uses `uv` for dependency management. Dependencies are already installed.

## Usage

### Test Installation

```bash
cd image-gen
uv run python test_simple.py
```

### Generate with SDXL-Turbo (Fast, Low VRAM)

```bash
cd image-gen

# Single step generation (very fast)
uv run python generate_sd.py "a majestic mountain landscape at sunset"

# With custom settings
uv run python generate_sd.py "cyberpunk city" --width 768 --height 512 --steps 4
```

### Generate with FLUX.1 (Best Quality, Needs 12GB+ VRAM)

```bash
cd image-gen

# Fast generation with FLUX.1-schnell (4 steps)
uv run python generate.py "a photorealistic portrait of a robot"

# High quality with FLUX.1-dev
uv run python generate.py "oil painting of a sunset" --model dev --steps 50
```

## Models

### SDXL-Turbo
- **VRAM**: 6-8GB
- **Speed**: 1-4 steps (1-5 seconds)
- **Quality**: Good
- **Best for**: Quick iterations, testing prompts

### FLUX.1-schnell
- **VRAM**: 12GB+
- **Speed**: 4 steps (10-20 seconds)
- **Quality**: Excellent
- **Best for**: High-quality images quickly

### FLUX.1-dev
- **VRAM**: 12GB+
- **Speed**: 50 steps (1-3 minutes)
- **Quality**: State-of-the-art
- **Best for**: Maximum quality

## Tips

1. Start with SDXL-Turbo to test prompts quickly
2. Use FLUX.1-schnell for good quality/speed balance
3. Images are saved to `outputs/` directory
4. Use `--seed` for reproducible results

## Troubleshooting

- **Out of Memory**: Use SDXL-Turbo or reduce resolution
- **Slow Generation**: Ensure CUDA is available (`nvidia-smi`)
- **Import Errors**: Wait for `uv add` to complete downloading