# gradcam.py - Grad-CAM visual explanations
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from model_pytorch import DeepfakeDetector
import os
from pathlib import Path

class GradCAM:
    """Grad-CAM visual explanations"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = (model_output > 0.5).long()
        
        # Backward pass
        self.model.zero_grad()
        model_output.backward(torch.ones_like(model_output))
        
        # Generate CAM
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        activations = self.activations
        
        cam = torch.sum(pooled_gradients * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()


def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET):
    """Apply a colormap to the image"""
    # Ensure activation map is in the right range
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose
    org_img_resized = cv2.resize(np.array(org_img), (224, 224))
    superimposed = cv2.addWeighted(org_img_resized, 0.6, heatmap, 0.4, 0)
    
    return heatmap, superimposed


def visualize_gradcam(model_path='best_model.pth', data_dir='data/test', num_samples=6):
    """Show Grad-CAM for test set samples"""
    
    print("=" * 60)
    print("🔍 Grad-CAM Visualization")
    print("=" * 60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device.upper()}")
    
    # Load model
    print("\n📦 Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    model = DeepfakeDetector(model_type='mobilenet', pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get target layer (last convolutional layer in MobileNetV2)
    target_layer = model.base_model.features[-1]
    
    # Create Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Get sample images
    print("\n📸 Collecting samples...")
    fake_dir = Path(data_dir) / 'fake'
    real_dir = Path(data_dir) / 'real'
    
    fake_images = list(fake_dir.glob('*.jpg'))[:num_samples//2]
    real_images = list(real_dir.glob('*.jpg'))[:num_samples//2]
    
    all_images = fake_images + real_images
    labels = ['Fake'] * len(fake_images) + ['Real'] * len(real_images)
    
    # Create visualization
    print("\n🎨 Generating visualizations...")
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples*3))
    
    for idx, (img_path, label) in enumerate(zip(all_images, labels)):
        # Load image
        original_image = Image.open(img_path).convert('RGB')
        input_tensor = transform(original_image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            prob = output.item()
            pred = 'Real' if prob > 0.5 else 'Fake'
            confidence = prob if prob > 0.5 else (1 - prob)
        
        # Generate Grad-CAM
        cam = grad_cam.generate_cam(input_tensor)
        
        # Apply colormap
        heatmap, superimposed = apply_colormap_on_image(original_image, cam)
        
        # Plot
        axes[idx, 0].imshow(original_image)
        axes[idx, 0].set_title(f'Original\nTrue: {label}', fontsize=10, fontweight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(heatmap)
        axes[idx, 1].set_title(f'Heatmap\nPred: {pred} ({confidence*100:.1f}%)', 
                              fontsize=10, fontweight='bold',
                              color='green' if pred == label else 'red')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(superimposed)
        axes[idx, 2].set_title(f'Grad-CAM\n{"✅ Correct" if pred == label else "❌ Wrong"}', 
                              fontsize=10, fontweight='bold',
                              color='green' if pred == label else 'red')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved visualization to: gradcam_visualization.png")
    
    print("\n" + "=" * 60)
    print("✅ Grad-CAM complete!")
    print("=" * 60)


if __name__ == "__main__":
    visualize_gradcam(
        model_path='best_model.pth',
        data_dir='data/test',
        num_samples=6
    )
