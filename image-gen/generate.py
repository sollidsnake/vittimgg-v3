#!/usr/bin/env python3
"""
Local Image Generation with FLUX.1
Supports FLUX.1-schnell (fast) and FLUX.1-dev models
"""

import torch
from diffusers import FluxPipeline
import argparse
from pathlib import Path
import os

def generate_image(
    prompt: str,
    model: str = "schnell",
    width: int = 1024,
    height: int = 1024,
    steps: int = None,
    seed: int = None,
    output_dir: str = "outputs"
):
    """Generate an image using FLUX.1 model."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Select model
    if model == "schnell":
        model_id = "black-forest-labs/FLUX.1-schnell"
        default_steps = 4  # Schnell is optimized for 4 steps
    else:
        model_id = "black-forest-labs/FLUX.1-dev"
        default_steps = 50
    
    if steps is None:
        steps = default_steps
    
    print(f"Loading {model_id}...")
    
    # Load pipeline
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    )
    
    # Enable memory efficient options
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    
    # Set seed for reproducibility
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)
    else:
        generator = None
    
    print(f"Generating image with prompt: '{prompt}'")
    print(f"Resolution: {width}x{height}, Steps: {steps}")
    
    # Generate image
    image = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        generator=generator,
        guidance_scale=0.0 if model == "schnell" else 3.5
    ).images[0]
    
    # Save image
    output_path = Path(output_dir) / f"flux_{model}_{len(os.listdir(output_dir))}.png"
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate images with FLUX.1")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--model", choices=["schnell", "dev"], default="schnell",
                        help="FLUX model to use (schnell=fast, dev=quality)")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", type=int, help="Number of inference steps")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    generate_image(
        prompt=args.prompt,
        model=args.model,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()