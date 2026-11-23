# Deepfake Detection

A deep learning project that classifies images as real or deepfake using a MobileNetV2-based model in PyTorch, with a Flask web interface for uploading and analyzing images.

## Features

- MobileNetV2 transfer-learning classifier with a custom detection head
- Flask web app: upload an image, get a real/fake prediction with a confidence score
- Training and evaluation scripts with metrics and a confusion matrix
- Grad-CAM visualizations showing which image regions influence the prediction

## Tech Stack

- Python, PyTorch, torchvision
- Flask (web app)
- scikit-learn, matplotlib, seaborn (evaluation)

## Project Structure

```
├── app_simple.py            # Flask web app (start here)
├── model_pytorch.py         # Model architecture
├── train_pytorch.py         # Training pipeline
├── evaluate_model.py        # Test-set evaluation and confusion matrix
├── detect_pytorch.py        # Command-line prediction on one image
├── gradcam.py               # Grad-CAM visualization
├── best_model.pth           # Trained model weights
├── templates/simple.html    # Web UI template
└── training_history.json    # Training metrics
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the web app

```bash
python app_simple.py
```

Open http://localhost:5000 and upload an image.

### 3. Command-line prediction

```bash
python detect_pytorch.py path/to/image.jpg
```

## Training Your Own Model

Organize images into `data/train`, `data/val`, and `data/test` folders with `real` and `fake` subfolders, then run:

```bash
python train_pytorch.py
python evaluate_model.py
```

## Results

- Best validation accuracy during training: 90.6% (measured on the project's own validation split)
- See `confusion_matrix.png` and `training_history.png` for details
- `gradcam_visualization.png` shows example Grad-CAM output

## Notes

- Helper scripts (`organize_dataset.py`, `explore_data.py`, `check_installation.py`) are included for dataset preparation and environment checks.
- This is a learning project. Its predictions are not reliable enough for real-world forensic use.
