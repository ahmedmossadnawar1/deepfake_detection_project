# organize_dataset.py
import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def organize_kaggle_dataset(source_dir, target_dir='data'):
    """
    Organize a dataset from Kaggle
    Supports: 140k Real and Fake Faces or FaceForensics++
    """
    
    print("=" * 60)
    print("📁 Organizing dataset")
    print("=" * 60)
    
    # Create the folder structure
    folders = {
        'train': ['real', 'fake'],
        'validation': ['real', 'fake'],
        'test': ['real', 'fake']
    }
    
    for split, categories in folders.items():
        for category in categories:
            os.makedirs(f'{target_dir}/{split}/{category}', exist_ok=True)
    
    print("\n✅ Structure created")
    
    # Find images
    print("\n🔍 Looking for images...")
    
    real_images = []
    fake_images = []
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                full_path = os.path.join(root, file)
                
                # Detect the class from the path or filename
                path_lower = full_path.lower()
                
                if any(x in path_lower for x in ['real', 'original', 'authentic']):
                    real_images.append(full_path)
                elif any(x in path_lower for x in ['fake', 'manipulated', 'synthetic', 'deepfake']):
                    fake_images.append(full_path)
    
    print(f"\n📊 Statistics:")
    print(f"   Real Images: {len(real_images):,}")
    print(f"   Fake Images: {len(fake_images):,}")
    print(f"   Total: {len(real_images) + len(fake_images):,}")
    
    if len(real_images) == 0 or len(fake_images) == 0:
        print("\n❌ Error: no images found!")
        print("   Make sure:")
        print("   1. The path is correct")
        print("   2. Folders contain 'real' or 'fake' in their names")
        return False
    
    # Balance the data
    min_count = min(len(real_images), len(fake_images))
    max_samples = min(min_count, 10000)  # cap at 10k per class for CPU training
    
    print(f"\n⚖️  Balancing data:")
    print(f"   Using {max_samples:,} images per class")
    
    random.seed(42)
    real_images = random.sample(real_images, max_samples)
    fake_images = random.sample(fake_images, max_samples)
    
    # Split: 70% train, 15% val, 15% test
    def split_dataset(images):
        random.shuffle(images)
        total = len(images)
        
        train_end = int(0.70 * total)
        val_end = int(0.85 * total)
        
        return {
            'train': images[:train_end],
            'validation': images[train_end:val_end],
            'test': images[val_end:]
        }
    
    real_splits = split_dataset(real_images)
    fake_splits = split_dataset(fake_images)
    
    print(f"\n📋 Split:")
    for split in ['train', 'validation', 'test']:
        print(f"   {split.capitalize():12s}: {len(real_splits[split]):,} real + {len(fake_splits[split]):,} fake")
    
    # Copy files
    print("\n📦 Copying files...")
    
    for split in ['train', 'validation', 'test']:
        # Real images
        print(f"\n   {split}/real...")
        for i, img_path in enumerate(tqdm(real_splits[split], desc=f"   Real {split}")):
            try:
                ext = os.path.splitext(img_path)[1]
                new_name = f"{i:06d}{ext}"
                shutil.copy2(img_path, f'{target_dir}/{split}/real/{new_name}')
            except Exception as e:
                print(f"   ⚠️  Error copying {img_path}: {e}")
        
        # Fake images
        print(f"   {split}/fake...")
        for i, img_path in enumerate(tqdm(fake_splits[split], desc=f"   Fake {split}")):
            try:
                ext = os.path.splitext(img_path)[1]
                new_name = f"{i:06d}{ext}"
                shutil.copy2(img_path, f'{target_dir}/{split}/fake/{new_name}')
            except Exception as e:
                print(f"   ⚠️  Error copying {img_path}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Dataset organized successfully!")
    print("=" * 60)
    
    # Final summary
    print("\n📊 Final summary:")
    for split in ['train', 'validation', 'test']:
        real_count = len(os.listdir(f'{target_dir}/{split}/real'))
        fake_count = len(os.listdir(f'{target_dir}/{split}/fake'))
        print(f"   {split.capitalize():12s}: {real_count:,} real + {fake_count:,} fake = {real_count + fake_count:,} total")
    
    return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎯 Dataset organizer for deepfake detection")
    print("=" * 60)
    
    source = input("\n📂 Enter the path of the Kaggle dataset folder: ").strip()
    
    # Strip quotes if present
    source = source.strip('"').strip("'")
    
    if not os.path.exists(source):
        print(f"\n❌ Path does not exist: {source}")
        print("\nMake sure:")
        print("1. You copied the path correctly")
        print("2. The folder is extracted")
    else:
        success = organize_kaggle_dataset(source)
        
        if success:
            print("\n🚀 Next steps:")
            print("   python explore_data.py")
