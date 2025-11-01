import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import torch

# ... other imports ...

# Define the path where you want to save the features
FEATURE_DIR = "C:\\Users\\gk414\\PycharmProjects\\PythonProject2\\features"

# Check if the directory exists and create it if it doesn't
if not os.path.exists(FEATURE_DIR):
    os.makedirs(FEATURE_DIR)
    print(f"Created feature directory: {FEATURE_DIR}")
# Paths
BASE_PATH = r"C:\Users\gk414\PycharmProjects\PythonProject2\Ddataset"
OUTPUT_PATH = r"C:\Users\gk414\PycharmProjects\PythonProject2\features"

# Load models
facenet = hub.load("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5")
vit_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

def preprocess_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(160, 160))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return img_array / 255.0

def get_facenet_embedding(img_array):
    img_resized = tf.image.resize(img_array, (160, 160))
    embedding = facenet(img_resized[tf.newaxis, ...])
    return embedding.numpy()[0]

def get_vit_features(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    inputs = vit_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_edge_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    edges = cv2.Canny(img, 100, 200)
    return edges.flatten()[:1000]

def extract_features(split):
    for label_name, label in [("real", 0), ("fake", 1)]:
        folder = os.path.join(BASE_PATH, split, label_name)
        output_csv = os.path.join(OUTPUT_PATH, f"{split}_features.csv")
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.png')):
                full_path = os.path.join(folder, fname)
                try:
                    img_array = preprocess_image(full_path)
                    facenet_feat = get_facenet_embedding(img_array)
                    vit_feat = get_vit_features(full_path)
                    edge_feat = get_edge_features(full_path)
                    combined = np.concatenate([facenet_feat, vit_feat, edge_feat])
                    df = pd.DataFrame([combined], columns=[f'f{i}' for i in range(len(combined))])
                    df['label'] = label
                    df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")

# Run for all splits
for split in ["train", "val", "test"]:
    extract_features(split)