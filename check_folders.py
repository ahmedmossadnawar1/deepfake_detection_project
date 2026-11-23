# check_folders.py - list actual folder names
import os
from pathlib import Path

path = input("Enter path: ").strip().strip('"').strip("'")
path = Path(path)

print("\n" + "="*60)
print("📁 Existing folders:")
print("="*60)

if not path.exists():
    print("❌ Path does not exist!")
else:
    folders = [f for f in path.iterdir() if f.is_dir()]
    
    for folder in sorted(folders):
        # Count images
        images = list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + list(folder.glob('*.png'))
        print(f"\n📂 {folder.name}")
        print(f"   Images: {len(images):,}")
        
        # Show first 3 filenames
        if images:
            print(f"   Examples: {[img.name for img in images[:3]]}")

print("\n" + "="*60)
