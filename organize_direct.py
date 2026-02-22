# organize_direct.py - تنظيم Dataset من train/test/valid مباشرة
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import random

def organize_existing_structure(base_path, target_dir='data'):
    """تنظيم dataset موجود بالفعل في train/test/valid"""
    
    print("="*60)
    print("📁 تنظيم Dataset من الهيكل الموجود")
    print("="*60)
    
    base_path = Path(base_path)
    
    # البحث عن المجلدات
    train_path = base_path / 'train'
    test_path = base_path / 'test'
    valid_path = base_path / 'valid'
    
    # التأكد من وجود train
    if not train_path.exists():
        print(f"❌ مجلد train غير موجود في: {base_path}")
        return False
    
    # شوف المجلدات في train
    folders_in_train = [f for f in train_path.iterdir() if f.is_dir()]
    
    print(f"\n📂 المجلدات في train: {[f.name for f in folders_in_train]}")
    
    if len(folders_in_train) != 2:
        print(f"❌ خطأ: المفروض يكون مجلدين (real و fake)، لكن موجود {len(folders_in_train)}")
        return False
    
    # تحديد أي مجلد هو real وأي واحد fake
    folder1, folder2 = sorted(folders_in_train)
    
    print(f"\n🤔 أي مجلد هو Real وأي واحد Fake؟")
    print(f"   1. {folder1.name} = Real, {folder2.name} = Fake")
    print(f"   2. {folder1.name} = Fake, {folder2.name} = Real")
    
    choice = input("\nاختار (1 أو 2): ").strip()
    
    if choice == "1":
        real_folder = folder1
        fake_folder = folder2
    else:
        real_folder = folder2
        fake_folder = folder1
    
    print(f"\n✅ Real = {real_folder.name}")
    print(f"✅ Fake = {fake_folder.name}")
    
    # جمع الصور
    print("\n🔍 جاري عد الصور...")
    
    real_images = list(real_folder.glob('*.jpg')) + list(real_folder.glob('*.jpeg')) + list(real_folder.glob('*.png'))
    fake_images = list(fake_folder.glob('*.jpg')) + list(fake_folder.glob('*.jpeg')) + list(fake_folder.glob('*.png'))
    
    print(f"\n📊 الإحصائيات:")
    print(f"   Real: {len(real_images):,}")
    print(f"   Fake: {len(fake_images):,}")
    
    if len(real_images) == 0 or len(fake_images) == 0:
        print("❌ خطأ: أحد المجلدات فارغ!")
        return False
    
    # اختيار 10k من كل فئة للسرعة
    max_samples = 10000
    
    random.seed(42)
    if len(real_images) > max_samples:
        real_images = random.sample(real_images, max_samples)
    if len(fake_images) > max_samples:
        fake_images = random.sample(fake_images, max_samples)
    
    print(f"\n⚖️ سنستخدم: {len(real_images):,} real + {len(fake_images):,} fake")
    
    # تقسيم البيانات
    def split_data(images):
        random.shuffle(images)
        total = len(images)
        train_end = int(0.70 * total)
        val_end = int(0.85 * total)
        
        return {
            'train': images[:train_end],
            'validation': images[train_end:val_end],
            'test': images[val_end:]
        }
    
    real_splits = split_data(real_images)
    fake_splits = split_data(fake_images)
    
    print(f"\n📋 التقسيم:")
    for split in ['train', 'validation', 'test']:
        print(f"   {split:12s}: {len(real_splits[split]):,} real + {len(fake_splits[split]):,} fake")
    
    # إنشاء المجلدات
    for split in ['train', 'validation', 'test']:
        os.makedirs(f'{target_dir}/{split}/real', exist_ok=True)
        os.makedirs(f'{target_dir}/{split}/fake', exist_ok=True)
    
    print("\n✅ تم إنشاء مجلدات data")
    
    # نسخ الملفات
    print("\n📦 جاري نسخ الملفات...")
    
    for split in ['train', 'validation', 'test']:
        # Real
        print(f"\n   {split}/real...")
        for i, img_path in enumerate(tqdm(real_splits[split], desc=f"Real {split}")):
            try:
                ext = img_path.suffix
                new_name = f"{i:06d}{ext}"
                shutil.copy2(img_path, f'{target_dir}/{split}/real/{new_name}')
            except Exception as e:
                pass
        
        # Fake
        print(f"   {split}/fake...")
        for i, img_path in enumerate(tqdm(fake_splits[split], desc=f"Fake {split}")):
            try:
                ext = img_path.suffix
                new_name = f"{i:06d}{ext}"
                shutil.copy2(img_path, f'{target_dir}/{split}/fake/{new_name}')
            except Exception as e:
                pass
    
    print("\n" + "="*60)
    print("✅ تم تنظيم Dataset بنجاح!")
    print("="*60)
    
    # ملخص
    print("\n📊 الملخص النهائي:")
    for split in ['train', 'validation', 'test']:
        real_count = len(os.listdir(f'{target_dir}/{split}/real'))
        fake_count = len(os.listdir(f'{target_dir}/{split}/fake'))
        print(f"   {split:12s}: {real_count:,} real + {fake_count:,} fake = {real_count+fake_count:,} total")
    
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 تنظيم Dataset")
    print("="*60)
    
    # المسار للمجلد الأب اللي فيه train/test/valid
    base = input("\n📂 أدخل المسار للمجلد الأب (اللي فيه train): ").strip().strip('"').strip("'")
    
    success = organize_existing_structure(base)
    
    if success:
        print("\n🚀 الخطوة التالية: python explore_data.py")
    else:
        print("\n❌ فشل التنظيم. راجع الأخطاء أعلاه.")
