# explore_data.py
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from pathlib import Path

def explore_dataset(data_dir='data'):
    """Full dataset analysis"""
    
    print("=" * 60)
    print("🔍 Exploring dataset")
    print("=" * 60)
    
    # Collect statistics
    stats = {}
    for split in ['train', 'validation', 'test']:
        split_path = Path(data_dir) / split
        if split_path.exists():
            real_count = len(list((split_path / 'real').glob('*')))
            fake_count = len(list((split_path / 'fake').glob('*')))
            stats[split] = {'real': real_count, 'fake': fake_count}
    
    # Print statistics
    print("\n📊 Dataset statistics:")
    print("-" * 60)
    total_real = sum(s['real'] for s in stats.values())
    total_fake = sum(s['fake'] for s in stats.values())
    
    for split, counts in stats.items():
        total = counts['real'] + counts['fake']
        print(f"{split.capitalize():12s}: {counts['real']:6,} real + {counts['fake']:6,} fake = {total:7,} total")
    
    print("-" * 60)
    print(f"{'Total':12s}: {total_real:6,} real + {total_fake:6,} fake = {total_real + total_fake:7,} total")
    
    balance = min(total_real, total_fake) / max(total_real, total_fake)
    print(f"\n⚖️  Balance Ratio: {balance:.2%}")
    
    # Analyze image dimensions
    print("\n📐 Analyzing image dimensions...")
    train_path = Path(data_dir) / 'train'
    
    widths, heights = [], []
    sample_size = 100
    
    for category in ['real', 'fake']:
        category_path = train_path / category
        images = list(category_path.glob('*'))[:sample_size]
        
        for img_path in images:
            try:
                img = Image.open(img_path)
                widths.append(img.width)
                heights.append(img.height)
            except:
                pass
    
    print(f"   Average width: {np.mean(widths):.0f} px")
    print(f"   Average height: {np.mean(heights):.0f} px")
    
    print("\n✅ Exploration complete!")
    print("\n🚀 Next steps:")
    print("   python model_pytorch.py")

if __name__ == "__main__":
    if os.path.exists('data/train'):
        explore_dataset()
    else:
        print("❌ data/train folder not found!")
        print("   Run: python organize_dataset.py first")
