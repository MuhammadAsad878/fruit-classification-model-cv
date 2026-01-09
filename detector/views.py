import os
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from tensorflow.keras.preprocessing import image

# --- NEW IMPORTS FOR REBUILDING MODEL ---
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# 1. Define the Architecture EXACTLY as it was in Colab
def build_model():
    # Load MobileNetV2 without weights first (structure only)
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(7, activation='softmax') # 7 Classes
    ])
    return model

# 2. Load the Weights into the Architecture
WEIGHTS_PATH = os.path.join(settings.BASE_DIR, 'detector', 'ml_models', 'fruit_weights.weights.h5')
CLASSES = ['Apple', 'Banana', 'Mango', 'Orange', 'Pineapple', 'Strawberry', 'Watermelon']

model = None
try:
    print("--------------------------------------------------")
    print(f"🏗️  Rebuilding Model Architecture Locally...")
    model = build_model()
    
    print(f"🔍 Loading weights from: {WEIGHTS_PATH}")
    if os.path.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)
        print("✅ Weights loaded successfully! System Ready.")
    else:
        print("❌ Error: Weights file not found at the specified path.")
        model = None
    print("--------------------------------------------------")
except Exception as e:
    print(f"❌ Critical Model Error: {e}")
    model = None

# 3. Prediction Logic (Same as before)
def predict_image(request):
    context = {}
    
    # Check if model is ready
    if model is None:
        context['error'] = "Model failed to load. Check server terminal."
        return render(request, 'detector/index.html', context)

    if request.method == 'POST' and 'image' in request.FILES:
        try:
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_url = fs.url(filename)
            file_path = os.path.join(settings.MEDIA_ROOT, filename)

            # Preprocess
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Predict
            predictions = model.predict(img_array)
            probs = predictions[0]

            # Parse Results
            stats = []
            for i, prob in enumerate(probs):
                stats.append({
                    'fruit': CLASSES[i],
                    'score': round(prob * 100, 2)
                })
            stats.sort(key=lambda x: x['score'], reverse=True)
            top_result = stats[0]

            context = {
                'result': top_result['fruit'],
                'confidence': top_result['score'],
                'all_stats': stats,
                'image_url': file_url
            }
            print(f"✅ Prediction: {top_result['fruit']}")

        except Exception as e:
            print(f"❌ Processing Error: {e}")
            context['error'] = str(e)

    return render(request, 'detector/index.html', context)