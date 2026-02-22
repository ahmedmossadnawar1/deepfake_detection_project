# check_installation.py
import sys

def check_environment():
    """فحص شامل للبيئة والمكتبات"""
    
    print("=" * 60)
    print("🔍 فحص بيئة التطوير")
    print("=" * 60)
    
    # Python Version
    print(f"\n✅ Python Version: {sys.version.split()[0]}")
    
    # المكتبات الأساسية
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
    
    print("\n📦 المكتبات المثبتة:")
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
            print(f"❌ {name:20s} - غير مثبتة")
            missing.append(lib)
    
    # معلومات PyTorch و CUDA
    if 'torch' not in missing:
        import torch
        print("\n" + "=" * 60)
        print("🖥️  معلومات PyTorch & Hardware")
        print("=" * 60)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            device = "cuda"
        else:
            print("⚠️  تحذير: CUDA غير متاح، سيتم استخدام CPU")
            print("   التدريب سيكون أبطأ بكثير")
            device = "cpu"
        
        print(f"\n🎯 Device للاستخدام: {device.upper()}")
    
    # النتيجة النهائية
    print("\n" + "=" * 60)
    if missing:
        print(f"❌ مكتبات ناقصة: {', '.join(missing)}")
        print(f"   نفذ: pip install {' '.join(missing)}")
        return False
    else:
        print("✅ كل شيء جاهز للبدء!")
        print("=" * 60)
        return True

if __name__ == "__main__":
    success = check_environment()
    
    if success:
        print("\n🚀 الخطوة التالية:")
        print("   1. حمل Dataset من Kaggle")
        print("   2. نفذ: python organize_dataset.py")
    
    sys.exit(0 if success else 1)
