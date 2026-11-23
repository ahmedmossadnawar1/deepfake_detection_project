# check_folders.py - list actual folder names
import os
from pathlib import Path

path = input("أدخل المسار: ").strip().strip('"').strip("'")
path = Path(path)

print("\n" + "="*60)
print("📁 المجلدات الموجودة:")
print("="*60)

if not path.exists():
    print("❌ المسار غير موجود!")
else:
    folders = [f for f in path.iterdir() if f.is_dir()]
    
    for folder in sorted(folders):
        # Count images
        images = list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + list(folder.glob('*.png'))
        print(f"\n📂 {folder.name}")
        print(f"   عدد الصور: {len(images):,}")
        
        # Show first 3 filenames
        if images:
            print(f"   أمثلة: {[img.name for img in images[:3]]}")

print("\n" + "="*60)
