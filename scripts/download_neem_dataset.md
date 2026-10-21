# Downloading the Neem Leaf Disease Dataset

## Option 1: Mendeley Neem Leaf Disease Dataset (2,544 images, 6 disease classes + healthy)
https://data.mendeley.com/datasets/sjtxmcv5d4

Classes: Alternaria, Dieback, Leaf blight, Leaf miners,
Leaf miners_Powdery mildew, Powdery mildew, Healthy.

This matches the class names already pre-filled in `treatment_db.json`
under `"neem"` — use it as-is if you go with this dataset.

## Option 2: NeemLeaf (newer, larger, field-collected, 2026)
https://data.mendeley.com/datasets/h3786mkn7t

6 classes: Healthy, Diseased, Yellow, Young/Growing, Shadowed, plus
whole-tree images. Less fine-grained on disease *type* (just "Diseased"
as one bucket) — better if you want a broader healthy-vs-stressed model,
worse if you want specific disease names for the recommendation step.
If you use this one, you'll need to either sub-label the "Diseased" images
yourself or use Option 1 for disease-type granularity.

## How to download
Mendeley datasets don't require an account — just:
1. Open the dataset page above.
2. Click the download button (usually a ZIP of all images).
3. Unzip into `data/neem/`.
4. Split into `train/` and `val/` (80/20) using the same approach as the
   mango split script in `download_mango_dataset.md`, adjusting the
   `SRC` path.

## Reminder
Class folder names must exactly match the keys under `"neem"` in
`treatment_db.json`, or you'll need to update that file to match.
