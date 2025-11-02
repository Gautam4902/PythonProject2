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


vit_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")


facenet_resnet = hub.load("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5")

IMAGE_SIZE = 160

@app.get("/")
def health_check():
    return {"status": "OK"}



def preprocess_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
   
    return np.array(image)


def get_facenet_embedding(img_array):

    normalized_img = img_array / 255.0

    input_tensor_32 = tf.cast(normalized_img, tf.float32)


    input_tensor_batch = input_tensor_32[tf.newaxis, ...]


    momentum_tensor = tf.constant(0.99, dtype=tf.float32)


    embedding = facenet_resnet(
        input_tensor_batch,
        False,  # training
        False,  # use_projection
        momentum_tensor  # batch_norm_momentum
    )

   
    return embedding.numpy()[0]


def get_vit_features(img_array):

    image = Image.fromarray(img_array).resize((224, 224))
    inputs = vit_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def get_edge_features(img_array):

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    edges_resized = cv2.resize(edges, (256, 256))
    return edges_resized.flatten()[:1000]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()


    img_array = preprocess_image(image_bytes)

    try:
        facenet_feat = get_facenet_embedding(img_array) 
        vit_feat = get_vit_features(img_array) 
        edge_feat = get_edge_features(img_array) 

        combined = np.concatenate([facenet_feat, vit_feat, edge_feat])

   
        input_array = combined.reshape(1, -1)

        prediction = model.predict(input_array)[0][0]
        label = "fake" if prediction > 0.5 else "real"
        return {"prediction": float(prediction), "label": label}
    except Exception as e:

        return {"error": str(e)}


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
