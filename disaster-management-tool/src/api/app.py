from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import SiameseNetwork, EncoderDecoderDamageDetection
import numpy as np
from io import BytesIO
from PIL import Image
import base64

app = FastAPI()

# Load pre-trained models
siamese_model = SiameseNetwork().cuda()
encoder_decoder_model = EncoderDecoderDamageDetection().cuda()

class ImageData(BaseModel):
    image: str  # Base64-encoded image

@app.post("/predict/")
async def predict(image_data: ImageData):
    # Decode base64 image
    img_data = base64.b64decode(image_data.image)
    img = Image.open(BytesIO(img_data))
    img = np.array(img.resize((224, 224)))
    img = img.transpose((2, 0, 1))  # Convert to CHW format
    img = torch.tensor(img).float().unsqueeze(0).cuda()

    # Prediction
    with torch.no_grad():
        output = siamese_model(img, img)  # Example: compare image with itself
        damage_prediction = encoder_decoder_model(output)
    
    return {"damage_prediction": damage_prediction.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
