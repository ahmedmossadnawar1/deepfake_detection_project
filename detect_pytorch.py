import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

# Model Architecture (same as model_pytorch.py)
class DeepfakeDetector(nn.Module):
    def __init__(self, model_type='mobilenet', pretrained=False):
        super(DeepfakeDetector, self).__init__()
        self.model_type = model_type
        
        if model_type == 'mobilenet':
            self.base_model = models.mobilenet_v2(weights=None)
            
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        return self.base_model(x)

# Configuration
MODEL_PATH = 'best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*60)
print("🤖 Deepfake Detection AI - PyTorch Version")
print("="*60)
print(f"💻 Device: {DEVICE}")

# Load model
print("\n🔄 Loading model...")
try:
    model = DeepfakeDetector(model_type='mobilenet', pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # Extract model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'Unknown')
        acc = checkpoint.get('best_val_acc', 0)
        print(f"✅ Model loaded successfully!")
        print(f"📊 Epoch: {epoch} | Best Accuracy: {acc:.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Model loaded successfully!")
    
    model.to(DEVICE)
    model.eval()
    print()
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Image preprocessing (ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def analyze_image(img_path):
    if not os.path.exists(img_path):
        print(f"❌ Error: File not found: {img_path}")
        return
    
    print(f"📥 Analyzing: {os.path.basename(img_path)}")
    
    try:
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            confidence = output.item() * 100
        
        # Determine result
        if confidence > 50:
            result = "Real"
            final_confidence = confidence
            icon = "✅"
        else:
            result = "Fake"
            final_confidence = 100 - confidence
            icon = "⚠️"
        
        # Display result
        print("\n" + "="*60)
        print(f"{icon} Result: {result.upper()} IMAGE")
        print(f"📊 Confidence: {final_confidence:.2f}%")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}\n")

def main():
    if len(sys.argv) > 1:
        # Image path from argument
        img_path = sys.argv[1]
        analyze_image(img_path)
    else:
        # Interactive mode
        print("📌 Usage:")
        print("   py detect_pytorch.py <image_path>")
        print("\n   Example: py detect_pytorch.py photo.jpg")
        print("\n" + "-"*60)
        
        # Ask for image path
        img_path = input("\n📂 Enter image path (or drag & drop): ").strip().strip('"')
        
        if img_path:
            analyze_image(img_path)
        else:
            print("❌ No image provided!")

if __name__ == "__main__":
    main()
