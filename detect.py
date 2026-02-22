import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys
import os

# Load model
MODEL_PATH = 'deepfake_detector_model.h5'

print("="*60)
print("🤖 Deepfake Detection AI - Command Line Version")
print("="*60)

print("\n🔄 Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

def preprocess_image(img_path):
    """Preprocess image for prediction"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def analyze_image(img_path):
    """Analyze single image"""
    if not os.path.exists(img_path):
        print(f"❌ Error: File not found: {img_path}")
        return
    
    print(f"📥 Analyzing: {img_path}")
    
    try:
        # Preprocess
        img_array = preprocess_image(img_path)
        
        # Predict
        prediction = model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0]) * 100
        
        # Determine result
        if confidence > 50:
            result = "Real"
            final_confidence = confidence
            icon = "✅"
            color = "green"
        else:
            result = "Fake"
            final_confidence = 100 - confidence
            icon = "⚠️"
            color = "red"
        
        # Display result
        print("\n" + "="*60)
        print(f"{icon} Result: {result.upper()} IMAGE")
        print(f"📊 Confidence: {final_confidence:.2f}%")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}\n")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Image path provided as argument
        img_path = sys.argv[1]
        analyze_image(img_path)
    else:
        # Interactive mode
        print("📌 Usage:")
        print("   python detect.py <image_path>")
        print("\n   Example: python detect.py image.jpg")
        print("\n" + "-"*60)
        
        # Ask for image path
        img_path = input("\n📂 Enter image path (or drag & drop): ").strip().strip('"')
        
        if img_path:
            analyze_image(img_path)
        else:
            print("❌ No image provided!")

if __name__ == "__main__":
    main()
