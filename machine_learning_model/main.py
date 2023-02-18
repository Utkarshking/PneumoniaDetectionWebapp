from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import requests
import io
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image
from pydantic import BaseModel

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("cnn_v2.h5")
classes = ["Normal","Pneumonia"]

class ImageURL(BaseModel):
    url: str

def read_file_as_image(data):
    img = io.BytesIO(data)
    img = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    image = np.vstack([x])
    return image

@app.get("/ping")
async def ping():
    return "Hello"

@app.post("/predict")
async def predict(image_url: ImageURL):
    try:
        response = requests.get(image_url.url, timeout=10.0)
        response.raise_for_status()
        image = read_file_as_image(response.content)
        pred = model.predict(image, batch_size=10)
        ans = int(pred[0][0])
        return {"result": classes[ans]}
    except Exception as e:
        print(e)
        return {"error": "Failed to classify image."}

