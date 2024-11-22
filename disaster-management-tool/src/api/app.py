from fastapi import FastAPI
import torch
from pydantic import BaseModel
from model import DisasterModel
from utils.config_loader import load_config
import torch
import cv2
import numpy as np

# Load config and model
config = load_config("config.yaml")
model = DisasterModel()
model.load_state_dict(torch.load(config['model']['save_path']))
model.eval()

# FastAPI app
app = FastAPI()

class ImageRequest(BaseModel):
    image: str  # Base64 image or image path

@app.post("/predict/")
async def predict(request: ImageRequest):
    # Read image and preprocess
    img_data = request.image  # Assuming base64 input, decode here
    img = cv2.imread(img_data)
    img = cv2.resize(img, tuple(config['image_processing']['resize_dim']))
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict using model
    with torch.no_grad():
        prediction = model(torch.tensor(img).float())
    
    return {"prediction": prediction.numpy().tolist()}
