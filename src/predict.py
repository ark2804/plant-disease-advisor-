"""
Run inference on a single leaf image from the command line.

Example:
    python src/predict.py --plant mango --image path/to/leaf.jpg
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import VAL_TRANSFORMS
from model import build_model, get_device
from recommend import get_recommendation, load_treatment_db

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_model_and_classes(plant: str, device):
    classes_path = os.path.join(MODELS_DIR, f"{plant}_classes.json")
    model_path = os.path.join(MODELS_DIR, f"{plant}_model.pth")

    if not os.path.exists(classes_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found for '{plant}'. "
            f"Run: python src/train.py --plant {plant} --data_dir data/{plant}"
        )

    with open(classes_path) as f:
        class_names = json.load(f)

    model = build_model(num_classes=len(class_names), freeze_backbone=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, class_names


def predict_image(plant: str, image_path: str) -> dict:
    device = get_device()
    model, class_names = load_model_and_classes(plant, device)

    image = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORMS(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probs, dim=0)

    disease = class_names[pred_idx.item()]
    db = load_treatment_db()
    recommendation = get_recommendation(plant, disease, db)

    return {
        "plant": plant,
        "disease": disease,
        "confidence": round(confidence.item(), 4),
        "recommendation": recommendation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", required=True, choices=["mango", "neem", "tamarind"])
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    result = predict_image(args.plant, args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
