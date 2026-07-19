import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "digit_ann_model.pth"

# FASTAPI APP
app = FastAPI(
    title="Handwritten Digit Recognition API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REQUEST MODEL
class DigitInput(BaseModel):
    pixels: list[float]

# MODEL
class DigitANN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 10)

        )

    def forward(self, x):
        return self.network(x)

# LOAD MODEL
model = DigitANN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

print("Model Loaded Successfully")

# ROOT
@app.get("/")
def home():

    return {
        "status": "success",
        "message": "Handwritten Digit Recognition API Running"
    }

# HEALTH CHECK
@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

# PREDICT
@app.post("/predict")
def predict(data: DigitInput):

    try:

        image = np.array(
            data.pixels,
            dtype=np.float32
        )

        if len(image) != 784:

            return {
                "status": "error",
                "message": "Input must contain exactly 784 values."
            }

        # Values should already be between 0 and 1
        image = image.reshape(1, 28, 28)

        image_tensor = torch.tensor(
            image,
            dtype=torch.float32
        )

        with torch.no_grad():

            output = model(image_tensor)

            probs = torch.softmax(
                output,
                dim=1
            )

            prediction = torch.argmax(
                probs,
                dim=1
            ).item()

            confidence = float(
                probs.max().item()
            )

        return {
            "status": "success",
            "prediction": prediction,
            "confidence": round(confidence, 4)
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }