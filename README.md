# Plant Leaf Disease Detection & Treatment Advisor
### Mango • Neem • Tamarind

A two-stage system:
1. **Vision model** — a CNN (transfer learning) classifies a leaf photo into a disease class, per plant species.
2. **Recommendation engine** — a JSON knowledge base maps the predicted disease to fertilizer / pesticide / cultural-practice advice.

This repo gives you working, runnable code for training, inference, and a REST API + simple test webpage.
**It does not include the actual image datasets** — I can't download from Kaggle/Mendeley from this environment (no network access to those domains). Section 2 below tells you exactly how to get them yourself (it's a 5-minute step).

---

## 1. Project structure

```
plant-disease-project/
├── data/
│   ├── mango/train/<ClassName>/*.jpg   ← you populate this
│   ├── mango/val/<ClassName>/*.jpg
│   ├── neem/train/<ClassName>/*.jpg
│   ├── neem/val/<ClassName>/*.jpg
│   ├── tamarind/train/<ClassName>/*.jpg
│   └── tamarind/val/<ClassName>/*.jpg
├── models/                     ← trained .pth models + class_index.json land here
├── src/
│   ├── model.py                 model builder (transfer learning)
│   ├── dataset.py                data loading + augmentation
│   ├── train.py                  training script (run this per plant)
│   ├── predict.py                 CLI inference on a single image
│   ├── recommend.py               treatment lookup logic
│   └── api.py                     FastAPI server
├── scripts/
│   ├── download_mango_dataset.md
│   ├── download_neem_dataset.md
│   └── tamarind_collection_guide.md
├── web/
│   └── index.html                simple browser test page
├── treatment_db.json             disease → fertilizer/pesticide knowledge base
└── requirements.txt
```

## 2. Get the datasets

Read the three files in `scripts/`:
- `download_mango_dataset.md` — Kaggle datasets, ready to use
- `download_neem_dataset.md` — Mendeley datasets, ready to use
- `tamarind_collection_guide.md` — **no public leaf dataset exists**, so this explains how to collect and label your own field photos

Once downloaded, arrange images into the `data/<plant>/train/<ClassName>/` and `data/<plant>/val/<ClassName>/` folders (80/20 split). Folder name = class name, e.g.:

```
data/mango/train/Anthracnose/img001.jpg
data/mango/train/Healthy/img002.jpg
```

## 3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Train a model (run once per plant)

```bash
python src/train.py --plant mango --data_dir data/mango --epochs 15
python src/train.py --plant neem --data_dir data/neem --epochs 15
python src/train.py --plant tamarind --data_dir data/tamarind --epochs 15
```

Each run saves:
- `models/<plant>_model.pth` — trained weights
- `models/<plant>_classes.json` — index-to-class-name mapping

Adjust `--epochs`, `--batch_size`, `--lr` as needed. Start with 10–15 epochs on top of the frozen pretrained backbone; unfreeze more layers later if accuracy plateaus.

## 5. Run inference from the command line

```bash
python src/predict.py --plant mango --image path/to/leaf.jpg
```

Outputs the predicted disease, confidence score, and the fertilizer/pesticide recommendation.

## 6. Run the API server

```bash
uvicorn src.api:app --reload --port 8000
```

Then either:
- Open `web/index.html` in your browser (it posts to `http://localhost:8000/predict`), or
- Call it directly:

```bash
curl -X POST "http://localhost:8000/predict?plant=mango" \
  -F "file=@path/to/leaf.jpg"
```

Response:
```json
{
  "plant": "mango",
  "disease": "Anthracnose",
  "confidence": 0.94,
  "recommendation": {
    "cause": "Fungal (Colletotrichum gloeosporioides)",
    "pesticide": "Carbendazim 50% WP @ 1g/L, spray at 10-15 day intervals",
    "fertilizer": "Balanced NPK with adequate potassium to strengthen resistance",
    "cultural_practice": "Prune and destroy infected twigs; avoid overhead irrigation",
    "severity_note": "Repeat spray through flowering and fruiting stages"
  }
}
```

## 7. Update the treatment knowledge base

Edit `treatment_db.json` directly. Structure:

```json
{
  "mango": {
    "Anthracnose": { "cause": "...", "pesticide": "...", "fertilizer": "...", "cultural_practice": "...", "severity_note": "..." }
  }
}
```

**Important:** the entries I've pre-filled for mango and neem are drawn from general agronomy guidance to get you started — verify every pesticide name/dosage against your local agricultural extension department or a licensed agronomist before this touches real farming decisions. Tamarind entries are left as placeholders since your disease classes aren't defined yet.

## 8. Next steps / improvements

- Add Grad-CAM visualization to `predict.py` so the model shows *why* it made a prediction (builds trust with users)
- Add a "plant identifier" model first if users won't specify which of the 3 species they're uploading
- Track prediction confidence — below a threshold (e.g. 60%), respond with "uncertain, consult an expert" instead of a guess
- Once you have more tamarind data, retrain with a larger validation set for a reliable accuracy estimate
