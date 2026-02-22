import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import matplotlib.patches as mpatches

# Create figure with better proportions
fig, ax = plt.subplots(1, 1, figsize=(12, 18))
ax.set_xlim(0, 10)
ax.set_ylim(0, 40)
ax.axis('off')

# Professional color palette
colors = {
    'start_end': '#2C3E50',
    'data': '#3498DB',
    'model': '#2ECC71',
    'train': '#27AE60',
    'eval': '#E67E22',
    'deploy': '#9B59B6',
    'decision': '#E74C3C'
}

def add_box(x, y, w, h, text, color, fontsize=9):
    """Add professional rounded box"""
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                         boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor='#2C3E50',
                         linewidth=2, alpha=0.85)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, weight='bold', color='white',
            linespacing=1.5)

def add_arrow(x1, y1, x2, y2, label='', style='->', lw=2.5):
    """Add arrow between nodes"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, mutation_scale=25,
                           linewidth=lw, color='#34495E')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x+0.4, mid_y, label, fontsize=8,
                style='italic', color='#E74C3C', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def add_diamond(x, y, text, color):
    """Add decision diamond"""
    w, h = 1.5, 1
    diamond = Polygon([
        (x, y+h/2),      # Top
        (x+w/2, y),      # Right
        (x, y-h/2),      # Bottom
        (x-w/2, y)       # Left
    ], facecolor=color, edgecolor='#2C3E50', linewidth=2, alpha=0.85)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=8, weight='bold', color='white', linespacing=1.3)

# ============================================================
# FLOWCHART CONTENT
# ============================================================

y = 38.5

# 1. START
add_box(5, y, 2.5, 0.7, '🚀 START', colors['start_end'], 11)
add_arrow(5, y-0.4, 5, y-1)

# 2. DATA COLLECTION
y -= 1.5
add_box(5, y, 4.5, 1.2, '📁 DATA COLLECTION\n\n• Real Images: 500-1000\n• Fake Images: 500-1000\n• Quality Control & Balance', colors['data'])
add_arrow(5, y-0.7, 5, y-1.5)

# 3. DATA ORGANIZATION
y -= 2
add_box(5, y, 4.5, 1.3, '📂 DATASET ORGANIZATION\n\ndata/\n├── train/ (80%): real/, fake/\n└── validation/ (20%): real/, fake/', colors['data'])
add_arrow(5, y-0.75, 5, y-1.5)

# 4. PREPROCESSING
y -= 2
add_box(5, y, 4.5, 1.4, '🔄 DATA PREPROCESSING\n\n• Resize → 224×224\n• ToTensor → [0,1]\n• Normalize (ImageNet stats)\n• Augmentation (train only)', colors['data'])
add_arrow(5, y-0.8, 5, y-1.5)

# 5. DATALOADER
y -= 2
add_box(5, y, 4.5, 1, '📦 CREATE DATALOADERS\n\nBatch Size: 32 | Shuffle: True\nWorkers: 4 | Pin Memory: True', colors['data'])
add_arrow(5, y-0.6, 5, y-1.5)

# 6. MODEL ARCHITECTURE
y -= 2
add_box(5, y, 4.5, 1.5, '🧠 MODEL ARCHITECTURE\n\nBase: MobileNetV2 (Pretrained)\nFreeze: Early 130 layers\nClassifier: Dense(256)→Dense(128)→Dense(1)\nRegularization: Dropout + BatchNorm', colors['model'])
add_arrow(5, y-0.85, 5, y-1.5)

# 7. TRAINING SETUP
y -= 2
add_box(5, y, 4.5, 1.3, '⚙️ TRAINING CONFIGURATION\n\nLoss: Binary Cross Entropy\nOptimizer: Adam (lr=0.0001)\nScheduler: ReduceLROnPlateau\nEpochs: 20 | Early Stop: 5', colors['train'])
add_arrow(5, y-0.75, 5, y-1.5)

# 8. TRAINING LOOP START
y -= 2
add_diamond(5, y, 'FOR EACH\nEPOCH\n(1-20)', colors['decision'])
add_arrow(5, y-0.6, 5, y-1.3, 'Start')

# 9. TRAINING PHASE
y -= 1.8
add_box(5, y, 4.5, 1.5, '🏋️ TRAINING PHASE\n\nmodel.train()\nFor each batch:\n  • Forward pass\n  • Calculate loss\n  • Backward pass\n  • Update weights', colors['train'])
add_arrow(5, y-0.85, 5, y-1.5)

# 10. VALIDATION PHASE
y -= 2
add_box(5, y, 4.5, 1.3, '✅ VALIDATION PHASE\n\nmodel.eval()\nCalculate: Accuracy, Loss\nMetrics: Precision, Recall, F1', colors['train'])
add_arrow(5, y-0.75, 5, y-1.5)

# 11. SAVE BEST MODEL DECISION
y -= 2
add_diamond(5, y, 'Accuracy >\nBest?', colors['decision'])
add_arrow(5, y-0.6, 5, y-1.3, 'Yes')

# Save branch (right)
add_arrow(5+0.9, y, 7.5, y, 'No')
add_box(7.5, y, 2, 0.7, 'Continue', colors['train'], 8)

# 12. SAVE CHECKPOINT
y -= 1.8
add_box(5, y, 4.5, 1.1, '💾 SAVE BEST MODEL\n\nSave: model_state_dict, optimizer,\nepoch, best_accuracy\nFile: best_model.pth', colors['train'])

# Loop back arrow
add_arrow(5, y-0.6, 8.5, y-0.6, style='-', lw=2)
add_arrow(8.5, y-0.6, 8.5, y+7.5, style='-', lw=2)
add_arrow(8.5, y+7.5, 5, y+7.5, 'Next Epoch', style='->', lw=2)

add_arrow(5, y-0.6, 5, y-1.3, 'Done')

# 13. MODEL EVALUATION
y -= 1.8
add_box(5, y, 4.5, 1.6, '📈 MODEL EVALUATION\n\nLoad best_model.pth\nTest Accuracy: 94.5%\nGenerate:\n  • Confusion Matrix\n  • Training History Plot\n  • GradCAM Visualization', colors['eval'])
add_arrow(5, y-0.9, 5, y-1.6)

# 14. WEB DEPLOYMENT
y -= 2.1
add_box(5, y, 4.5, 1.4, '🌐 WEB DEPLOYMENT\n\nFramework: Flask\nBackend: app_simple.py\nFrontend: HTML/CSS/JS\nServer: localhost:5000', colors['deploy'])
add_arrow(5, y-0.8, 5, y-1.5)

# 15. USER UPLOADS IMAGE
y -= 2
add_box(5, y, 4.5, 1, '👤 USER INTERACTION\n\nUser opens browser → Upload image\nFormats: JPG, PNG', colors['deploy'])
add_arrow(5, y-0.6, 5, y-1.3)

# 16. PREPROCESSING
y -= 1.8
add_box(5, y, 4.5, 1.2, '🔧 IMAGE PREPROCESSING\n\nResize → 224×224\nNormalize → Tensor\nMove to device', colors['deploy'])
add_arrow(5, y-0.7, 5, y-1.4)

# 17. MODEL PREDICTION
y -= 1.9
add_box(5, y, 4.5, 1.1, '🤖 MODEL PREDICTION\n\nwith torch.no_grad():\n  output = model(image)\n  confidence = output × 100%', colors['model'])
add_arrow(5, y-0.65, 5, y-1.3)

# 18. DECISION
y -= 1.8
add_diamond(5, y, 'Confidence\n> 50%?', colors['decision'])

# Real branch
add_arrow(5+0.9, y, 7.2, y, 'Yes')
add_box(7.2, y, 2.2, 0.8, '✅ REAL IMAGE\nConf: XX%', '#27AE60', 9)
add_arrow(7.2, y-0.5, 7.2, y-1.2)

# Fake branch
add_arrow(5-0.9, y, 2.8, y, 'No')
add_box(2.8, y, 2.2, 0.8, '⚠️ FAKE IMAGE\nConf: XX%', '#C0392B', 9)
add_arrow(2.8, y-0.5, 2.8, y-1.2)

# Merge
add_arrow(7.2, y-1.2, 5, y-2)
add_arrow(2.8, y-1.2, 5, y-2)

# 19. DISPLAY RESULTS
y -= 2.5
add_box(5, y, 4.5, 1, '📊 DISPLAY RESULTS\n\nShow: Icon, Text, Confidence Bar\nOption: Upload Another Image', colors['deploy'])
add_arrow(5, y-0.6, 5, y-1.3)

# 20. END
y -= 1.8
add_box(5, y, 2.5, 0.7, '🏁 END', colors['start_end'], 11)

# ============================================================
# TITLE & LEGEND
# ============================================================
plt.title('🤖 DEEPFAKE DETECTION AI\nComplete Project Workflow', 
          fontsize=18, weight='bold', pad=20, color='#2C3E50')

# Professional Legend
legend_elements = [
    mpatches.Patch(facecolor=colors['data'], label='Data Processing', edgecolor='#2C3E50', linewidth=1.5),
    mpatches.Patch(facecolor=colors['model'], label='Model Architecture', edgecolor='#2C3E50', linewidth=1.5),
    mpatches.Patch(facecolor=colors['train'], label='Training & Optimization', edgecolor='#2C3E50', linewidth=1.5),
    mpatches.Patch(facecolor=colors['eval'], label='Evaluation', edgecolor='#2C3E50', linewidth=1.5),
    mpatches.Patch(facecolor=colors['deploy'], label='Deployment & Inference', edgecolor='#2C3E50', linewidth=1.5),
    mpatches.Patch(facecolor=colors['decision'], label='Decision Point', edgecolor='#2C3E50', linewidth=1.5),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
         framealpha=0.95, edgecolor='#2C3E50', fancybox=True, shadow=True)

# ============================================================
# SAVE
# ============================================================
plt.tight_layout()
plt.savefig('professional_flowchart.png', dpi=300, 
           bbox_inches='tight', facecolor='white', pad_inches=0.3)

print("\n" + "="*70)
print("✅ PROFESSIONAL FLOWCHART CREATED SUCCESSFULLY!")
print("="*70)
print(f"📄 File: professional_flowchart.png")
print(f"🎨 Resolution: 300 DPI")
print(f"📐 Size: 12×18 inches (Perfect for A3 printing)")
print(f"✨ Clean, organized, and presentation-ready!")
print(f"🎯 Total phases: 20 clear steps")
print(f"💼 Professional quality for academic submission")
print("="*70 + "\n")

plt.show()
