# evaluate_model.py - اختبار النموذج على Test Set
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_pytorch import DeepfakeDetector
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def evaluate_model(model_path='best_model.pth', data_dir='data/test'):
    """تقييم النموذج على Test Set"""
    
    print("=" * 60)
    print("🧪 تقييم النموذج على Test Set")
    print("=" * 60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device.upper()}")
    
    # Load model
    print("\n📦 تحميل النموذج...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = DeepfakeDetector(model_type='mobilenet', pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ تم تحميل النموذج من Epoch {checkpoint['epoch']}")
    print(f"   Best Val Accuracy: {checkpoint['best_val_acc']*100:.2f}%")
    
    # Test transforms
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load test data
    print("\n📊 تحميل Test Data...")
    test_dataset = datasets.ImageFolder(data_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"   Test Images: {len(test_dataset):,}")
    print(f"   Classes: {test_dataset.classes}")
    
    # Evaluate
    print("\n🔍 جاري التقييم...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("📊 نتائج التقييم")
    print("=" * 60)
    
    accuracy = (all_preds == all_labels).mean()
    print(f"\n🎯 Test Accuracy: {accuracy*100:.2f}%")
    
    # Classification Report
    print("\n📋 تقرير مفصل:")
    print("-" * 60)
    class_names = ['Fake', 'Real']
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\n📈 حفظ Confusion Matrix في: confusion_matrix.png")
    
    # Calculate per-class metrics
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "=" * 60)
    print("📊 تفاصيل إضافية")
    print("=" * 60)
    print(f"\n✅ True Negatives (Fake → Fake):  {tn:,}")
    print(f"✅ True Positives (Real → Real):  {tp:,}")
    print(f"❌ False Positives (Fake → Real): {fp:,}")
    print(f"❌ False Negatives (Real → Fake): {fn:,}")
    
    # Per-class accuracy
    fake_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0
    real_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n🎭 Fake Detection Accuracy: {fake_accuracy*100:.2f}%")
    print(f"👤 Real Detection Accuracy: {real_accuracy*100:.2f}%")
    
    # Confidence analysis
    fake_probs = all_probs[all_labels == 0]
    real_probs = all_probs[all_labels == 1]
    
    print(f"\n📊 Confidence Scores:")
    print(f"   Fake images - Avg score: {(1-fake_probs).mean():.4f} (lower = more fake)")
    print(f"   Real images - Avg score: {real_probs.mean():.4f} (higher = more real)")
    
    print("\n" + "=" * 60)
    print("✅ انتهى التقييم!")
    print("=" * 60)
    
    return accuracy, cm


if __name__ == "__main__":
    accuracy, cm = evaluate_model(
        model_path='best_model.pth',
        data_dir='data/test'
    )
