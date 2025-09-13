#!/usr/bin/env python3
"""
Local Image Generation with Stable Diffusion
Lightweight alternative using SDXL-Turbo for fast generation
"""

import torch
from diffusers import AutoPipelineForText2Image
import argparse
from pathlib import Path
import os

def generate_image(
    prompt: str,
    width: int = 512,
    height: int = 512,
    steps: int = 1,
    seed: int = None,
    output_dir: str = "outputs"
):
    """Generate an image using SDXL-Turbo (fast, lightweight model)."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Loading SDXL-Turbo model...")
    
    # Load lightweight SDXL-Turbo model (faster download, less VRAM)
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
    # Use CPU if CUDA not available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        device = "cuda"
    else:
        pipe = pipe.to("cpu")
        device = "cpu"
        print("Note: Running on CPU - generation will be slow")
    
    # Set seed for reproducibility
    if seed is not None:
        generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None
    
    print(f"Generating image with prompt: '{prompt}'")
    print(f"Resolution: {width}x{height}, Steps: {steps}")
    
    # Generate image (SDXL-Turbo works best with 1-4 steps)
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=0.0,  # SDXL-Turbo doesn't use guidance
        width=width,
        height=height,
        generator=generator
    ).images[0]
    
    # Save image
    output_path = Path(output_dir) / f"sdxl_turbo_{len(os.listdir(output_dir))}.png"
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate images with SDXL-Turbo")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--steps", type=int, default=1, help="Number of inference steps (1-4 recommended)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    generate_image(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()