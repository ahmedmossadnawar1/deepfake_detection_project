# model_pytorch.py
import torch
import torch.nn as nn
from torchvision import models

class DeepfakeDetector(nn.Module):
    """نموذج كشف Deepfakes باستخدام MobileNetV2"""
    
    def __init__(self, model_type='mobilenet', pretrained=True):
        super(DeepfakeDetector, self).__init__()
        
        self.model_type = model_type
        
        if model_type == 'mobilenet':
            self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
            
            for i, param in enumerate(self.base_model.parameters()):
                if i < len(list(self.base_model.parameters())) - 20:
                    param.requires_grad = False
            
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
            
        elif model_type == 'resnet18':
            self.base_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            
            for param in list(self.base_model.parameters())[:-30]:
                param.requires_grad = False
            
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.4),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        return self.base_model(x)
    
    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def create_model(model_type='mobilenet', device='cpu'):
    print("=" * 60)
    print("🔨 بناء النموذج")
    print("=" * 60)
    
    model = DeepfakeDetector(model_type=model_type, pretrained=True)
    model = model.to(device)
    
    trainable, total = model.count_parameters()
    print(f"\n📊 معلومات النموذج:")
    print(f"   Type: {model_type}")
    print(f"   Total Parameters: {total:,}")
    print(f"   Trainable Parameters: {trainable:,}")
    print(f"   Frozen Parameters: {total - trainable:,}")
    print(f"   Device: {device.upper()}")
    
    criterion = nn.BCELoss()
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001,
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2
    )
    
    print("\n✅ تم بناء النموذج بنجاح!")
    print("=" * 60)
    
    return model, criterion, optimizer, scheduler


def test_model():
    print("\n🧪 اختبار النموذج...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, criterion, optimizer, scheduler = create_model('mobilenet', device)
    
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    print("\n⚙️  Forward Pass...")
    output = model(dummy_input)
    
    print(f"   Input Shape: {dummy_input.shape}")
    print(f"   Output Shape: {output.shape}")
    print(f"   Output Values: {output.detach().cpu().numpy().flatten()}")
    
    dummy_labels = torch.tensor([[1.0], [0.0]]).to(device)
    loss = criterion(output, dummy_labels)
    
    print(f"\n   Loss: {loss.item():.4f}")
    print("\n✅ النموذج يعمل بشكل صحيح!")


if __name__ == "__main__":
    test_model()
