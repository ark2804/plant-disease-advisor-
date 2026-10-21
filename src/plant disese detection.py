"""
FastAPI server for the plant disease advisor.

Run with:
    uvicorn src.api:app --reload --port 8000

Then POST an image file to /predict?plant=mango or open http://localhost:8000 in your browser!
"""
import io
import json
import os
import sys

# Ensure src directory is in sys.path regardless of launch cwd
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from dataset import VAL_TRANSFORMS
from model import build_model, get_device
from recommend import get_recommendation, load_treatment_db

MODELS_DIR = os.path.join(SRC_DIR, "..", "models")
WEB_DIR = os.path.join(SRC_DIR, "..", "web")
SUPPORTED_PLANTS = ["mango", "neem", "tamarind"]

app = FastAPI(
    title="Plant Leaf Disease Advisor",
    description="Deep learning powered crop disease diagnosis and agronomist treatment advisor",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = get_device()
_model_cache = {}
_treatment_db = load_treatment_db()


def get_model_and_classes(plant: str):
    if plant in _model_cache:
        return _model_cache[plant]

    classes_path = os.path.join(MODELS_DIR, f"{plant}_classes.json")
    model_path = os.path.join(MODELS_DIR, f"{plant}_model.pth")

    if not os.path.exists(classes_path) or not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"No trained model found for '{plant}'. Train it first with "
                f"train.py before calling this endpoint."
            ),
        )

    with open(classes_path, "r") as f:
        class_names = json.load(f)

    model = build_model(num_classes=len(class_names), freeze_backbone=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    _model_cache[plant] = (model, class_names)
    return model, class_names


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Plant Leaf Disease Advisor API is running!</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


@app.get("/health")
@app.get("/api/health")
def health():
    trained = [
        p for p in SUPPORTED_PLANTS
        if os.path.exists(os.path.join(MODELS_DIR, f"{p}_model.pth"))
    ]
    return {
        "status": "ok",
        "device": str(device),
        "supported_plants": SUPPORTED_PLANTS,
        "trained_plants": trained,
    }


@app.get("/api/plants")
def get_plants_info():
    data = {}
    for p in SUPPORTED_PLANTS:
        classes_path = os.path.join(MODELS_DIR, f"{p}_classes.json")
        classes = []
        if os.path.exists(classes_path):
            with open(classes_path, "r") as f:
                classes = json.load(f)
        treatments = _treatment_db.get(p, {})
        data[p] = {
            "name": p.capitalize(),
            "classes": classes,
            "treatment_count": len([k for k in treatments.keys() if not k.startswith("_")]),
            "is_trained": os.path.exists(os.path.join(MODELS_DIR, f"{p}_model.pth")),
        }
    return data


@app.post("/predict")
@app.post("/api/predict")
async def predict(plant: str, file: UploadFile = File(...)):
    plant = plant.lower()
    if plant not in SUPPORTED_PLANTS:
        raise HTTPException(
            status_code=400,
            detail=f"'{plant}' not supported. Choose from {SUPPORTED_PLANTS}",
        )

    model, class_names = get_model_and_classes(plant)

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    tensor = VAL_TRANSFORMS(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probs, dim=0)

        # Get all class probabilities sorted by confidence
        sorted_indices = torch.argsort(probs, descending=True)
        top_predictions = [
            {
                "disease": class_names[idx.item()],
                "confidence": round(probs[idx.item()].item(), 4),
                "percentage": round(probs[idx.item()].item() * 100, 1),
            }
            for idx in sorted_indices[:min(5, len(class_names))]
        ]

    disease = class_names[pred_idx.item()]
    recommendation = get_recommendation(plant, disease, _treatment_db)

    return {
        "plant": plant,
        "disease": disease,
        "confidence": round(confidence.item(), 4),
        "percentage": round(confidence.item() * 100, 1),
        "is_healthy": "healthy" in disease.lower(),
        "top_predictions": top_predictions,
        "recommendation": recommendation,
    }

