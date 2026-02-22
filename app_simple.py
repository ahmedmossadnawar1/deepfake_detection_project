from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # ← ضيف ده
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
CORS(app)  # ← ضيف ده

# Model Architecture
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
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

# Load model
print("🔄 Loading model...")
DEVICE = torch.device('cpu')
model = DeepfakeDetector()
checkpoint = torch.load('best_model.pth', map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✅ Model loaded!")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('simple.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("📥 Received prediction request")  # ← Debug
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        print(f"📄 File: {file.filename}")  # ← Debug
        
        img = Image.open(file.stream).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            confidence = output.item() * 100
        
        if confidence > 50:
            result = 'Real'
            final_conf = confidence
        else:
            result = 'Fake'
            final_conf = 100 - confidence
        
        print(f"✅ Result: {result} ({final_conf:.2f}%)")  # ← Debug
        
        return jsonify({
            'result': result,
            'confidence': round(final_conf, 2)
        })
    except Exception as e:
        print(f"❌ Error: {e}")  # ← Debug
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🌐 Server starting on http://127.0.0.1:5000")
    print("🔗 Open this URL in your browser\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
