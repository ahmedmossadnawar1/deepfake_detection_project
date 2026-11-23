# check_installation.py
import sys

def check_environment():
    """Full environment and library check"""
    
    print("=" * 60)
    print("🔍 Checking development environment")
    print("=" * 60)
    
    # Python Version
    print(f"\n✅ Python Version: {sys.version.split()[0]}")
    
    # Core libraries
    libraries = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'flask': 'Flask',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow',
        'tqdm': 'tqdm',
        'pytorch_grad_cam': 'Grad-CAM'
    }
    
    print("\n📦 Installed libraries:")
    print("-" * 60)
    
    missing = []
    versions = {}
    
    for lib, name in libraries.items():
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'Unknown')
            versions[name] = version
            print(f"✅ {name:20s} - v{version}")
        except ImportError:
            print(f"❌ {name:20s} - not installed")
            missing.append(lib)
    
    # PyTorch and CUDA info
    if 'torch' not in missing:
        import torch
        print("\n" + "=" * 60)
        print("🖥️  PyTorch & Hardware Info")
        print("=" * 60)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            device = "cuda"
        else:
            print("⚠️  Warning: CUDA not available, using CPU")
            print("   Training will be much slower")
            device = "cpu"
        
        print(f"\n🎯 Device to use: {device.upper()}")
    
    # Final result
    print("\n" + "=" * 60)
    if missing:
        print(f"❌ Missing libraries: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}")
        return False
    else:
        print("✅ Everything is ready!")
        print("=" * 60)
        return True

if __name__ == "__main__":
    success = check_environment()
    
    if success:
        print("\n🚀 Next steps:")
        print("   1. Download a dataset from Kaggle")
        print("   2. Run: python organize_dataset.py")
    
    sys.exit(0 if success else 1)
