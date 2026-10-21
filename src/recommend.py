"""
Loads treatment_db.json and looks up fertilizer/pesticide advice
for a given (plant, disease) pair.
"""
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "treatment_db.json")


def load_treatment_db(path: str = DB_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_recommendation(plant: str, disease: str, db: dict = None) -> dict:
    """
    Returns the treatment dict for a plant+disease pair, or a fallback
    message if the combination isn't in the knowledge base yet.
    """
    if db is None:
        db = load_treatment_db()

    plant_entries = db.get(plant.lower(), {})
    entry = plant_entries.get(disease)

    if entry is None:
        return {
            "cause": "Not yet documented in the knowledge base",
            "pesticide": "N/A",
            "fertilizer": "N/A",
            "cultural_practice": "N/A",
            "severity_note": (
                f"No treatment entry found for '{disease}' under '{plant}'. "
                f"Add it to treatment_db.json."
            ),
        }
    return entry
