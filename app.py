from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub  # Import added for ResNet loading
from PIL import Image
import io
import cv2
from transformers import ViTFeatureExtractor, ViTModel
import torch
import uvicorn

app = FastAPI()
model = tf.keras.models.load_model("deepfake_model.h5")

# Load ViT (768 dimensions)
vit_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# Load the EXACT SAME ResNet model used for training (2048 dimensions)
facenet_resnet = hub.load("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5")

# FIX: Set IMAGE_SIZE to 160 to match the initial resize in the training script
IMAGE_SIZE = 160


def preprocess_image(image_bytes):
    # Opens image bytes, converts to RGB, and resizes to 160x160
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    # Returns a 0-255 NumPy array
    return np.array(image)


def get_facenet_embedding(img_array):
    # 1. Normalize the input data (from 0-255 to 0-1)
    normalized_img = img_array / 255.0

    # 2. Cast the image data to tf.float32 (Fix 1: dtype mismatch)
    input_tensor_32 = tf.cast(normalized_img, tf.float32)

    # 3. Add the batch dimension (for a single image)
    input_tensor_batch = input_tensor_32[tf.newaxis, ...]

    # 4. Define the momentum as a tf.float32 scalar tensor (Fix 2: argument mismatch)
    # The default for batch normalization momentum is often 0.99 for inference,
    # but the API requires it as a TensorSpec, not a Python float.
    momentum_tensor = tf.constant(0.99, dtype=tf.float32)

    # The expected positional arguments are:
    # (inputs, training, use_projection, batch_norm_momentum)
    # We are using it for feature extraction (inference), so 'training' is False.
    # We will assume 'use_projection' is also False as per the error log options.
    embedding = facenet_resnet(
        input_tensor_batch,
        False,  # training
        False,  # use_projection
        momentum_tensor  # batch_norm_momentum
    )

    # The output is 2048 dimensions
    return embedding.numpy()[0]


def get_vit_features(img_array):
    # ViT preprocessing (768 dimensions)
    image = Image.fromarray(img_array).resize((224, 224))
    inputs = vit_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def get_edge_features(img_array):
    # Edge features (1000 dimensions)
    # The input img_array is 160x160 and 0-255
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    # Resize to 256x256 before flattening/truncating as per the training script
    edges_resized = cv2.resize(edges, (256, 256))
    return edges_resized.flatten()[:1000]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Preprocess image to 160x160 array (0-255)
    img_array = preprocess_image(image_bytes)

    try:
        facenet_feat = get_facenet_embedding(img_array)  # 2048 features
        vit_feat = get_vit_features(img_array)  # 768 features
        edge_feat = get_edge_features(img_array)  # 1000 features

        # Total features: 2048 + 768 + 1000 = 3816
        combined = np.concatenate([facenet_feat, vit_feat, edge_feat])

        # Reshape for the model (1 sample, 3816 features)
        input_array = combined.reshape(1, -1)

        prediction = model.predict(input_array)[0][0]
        label = "fake" if prediction > 0.5 else "real"
        return {"prediction": float(prediction), "label": label}
    except Exception as e:
        # Note: If you still get a shape mismatch, check your training CSV again
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)