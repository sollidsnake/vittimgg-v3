#!/usr/bin/env python3
"""
Simple test to verify installation
"""

def test_imports():
    """Test if required packages are installed."""
    try:
        print("Testing imports...")
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import diffusers
        print(f"✓ Diffusers {diffusers.__version__}")
        
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠ CUDA not available - will run on CPU (slow)")
        
        print("\nInstallation successful! You can now generate images.")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nPlease wait for dependencies to finish installing.")
        return False

if __name__ == "__main__":
    test_imports()