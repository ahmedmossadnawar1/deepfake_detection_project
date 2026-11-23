# train_pytorch.py - model training
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_pytorch import create_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time
import json

class DeepfakeTrainer:
    """Trainer class for the deepfake detection model"""
    
    def __init__(self, data_dir='data', batch_size=8, num_workers=0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("=" * 60)
        print("🎯 Setting up training environment")
        print("=" * 60)
        print(f"Device: {self.device.upper()}")
        print(f"Batch Size: {batch_size}")
        print(f"Data Directory: {data_dir}")
        
        # Data Augmentation
        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self._load_data()
        self._setup_model()
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def _load_data(self):
        """Load data"""
        print("\n📊 Loading data...")
        
        self.train_dataset = datasets.ImageFolder(
            f'{self.data_dir}/train',
            transform=self.train_transforms
        )
        
        self.val_dataset = datasets.ImageFolder(
            f'{self.data_dir}/validation',
            transform=self.val_transforms
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
        
        print(f"   Train: {len(self.train_dataset):,} images")
        print(f"   Validation: {len(self.val_dataset):,} images")
        print(f"   Classes: {self.train_dataset.classes}")
        print(f"   Batches per epoch: {len(self.train_loader)}")
    
    def _setup_model(self):
        """Set up the model"""
        self.model, self.criterion, self.optimizer, self.scheduler = create_model(
            model_type='mobilenet',
            device=self.device
        )
        
        self.best_val_acc = 0.0
        self.start_epoch = 0
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        """Evaluate on the validation set"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs=10, save_best=True):
        """Full training loop"""
        print("\n" + "=" * 60)
        print(f"🚀 Starting training for {epochs} epochs")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            epoch_start = time.time()
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Print summary
            print(f"\n📊 Epoch {epoch+1}/{self.start_epoch + epochs} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            print(f"   Learning Rate: {current_lr:.6f}")
            print(f"   Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
            
            # Save best model
            if save_best and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"   ✅ Saved best model! Val Acc: {val_acc*100:.2f}%")
            
            print("-" * 60)
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🎉 Training complete!")
        print("=" * 60)
        print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        print(f"🏆 Best Val Accuracy: {self.best_val_acc*100:.2f}%")
        
        # Save final model
        self.save_checkpoint(epoch, is_best=False)
        
        # Plot results
        self.plot_history()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if is_best:
            torch.save(checkpoint, 'best_model.pth')
        else:
            torch.save(checkpoint, 'last_model.pth')
    
    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs_range, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs_range, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs_range, [acc*100 for acc in self.history['train_acc']], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs_range, [acc*100 for acc in self.history['val_acc']], 'r-', label='Val Acc', linewidth=2)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[2].plot(epochs_range, self.history['lr'], 'g-', linewidth=2)
        axes[2].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('LR')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print(f"\n📈 Saved plot to: training_history.png")
        
        # Save history as JSON
        with open('training_history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"💾 Saved history to: training_history.json")


if __name__ == "__main__":
    # Create the trainer
    trainer = DeepfakeTrainer(
        data_dir='data',
        batch_size=8,
        num_workers=0
    )
    
    # Training
    trainer.train(epochs=10, save_best=True)
