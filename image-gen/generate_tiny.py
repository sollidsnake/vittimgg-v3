#!/usr/bin/env python3
"""
Minimal image generation with tiny models for testing
Uses smaller models that download quickly
"""

import torch
from diffusers import DiffusionPipeline
import argparse
from pathlib import Path
import os

def generate_image(
    prompt: str,
    model: str = "tiny",
    width: int = 256,
    height: int = 256,
    steps: int = 25,
    seed: int = None,
    output_dir: str = "outputs"
):
    """Generate an image using tiny/test models."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Model selection
    if model == "tiny":
        # Tiny SD model for testing (very small download)
        model_id = "segmind/tiny-sd"
        print(f"Loading Tiny SD model (fast download)...")
    else:
        # Small SD model  
        model_id = "segmind/small-sd"
        print(f"Loading Small SD model...")
    
    # Load pipeline
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )
    
    # Use CPU (these models are small enough for CPU)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        device = "cuda"
    else:
        pipe = pipe.to("cpu")
        device = "cpu"
    
    # Set seed for reproducibility
    if seed is not None:
        generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None
    
    print(f"Generating image with prompt: '{prompt}'")
    print(f"Resolution: {width}x{height}, Steps: {steps}")
    
    # Generate image
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        width=width,
        height=height,
        generator=generator
    ).images[0]
    
    # Save image
    output_path = Path(output_dir) / f"{model}_{len(os.listdir(output_dir))}.png"
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate images with tiny models")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--model", choices=["tiny", "small"], default="tiny",
                        help="Model size to use")
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
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