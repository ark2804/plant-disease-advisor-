# Downloading the Mango Leaf Disease Dataset

Several ready-to-use public datasets exist. Pick one (or combine them for more data).

## Option 1: MangoLeafBD (most widely used, ~8 classes)
Kaggle: https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset

Classes typically include: Anthracnose, Bacterial Canker, Cutting Weevil,
Die Back, Gall Midge, Healthy, Powdery Mildew, Sooty Mould.

## Option 2: Combined dataset (larger, merges multiple sources)
Kaggle: https://www.kaggle.com/datasets/binhpt310/mango-combined-dataset

## How to download via Kaggle CLI

1. Create a free Kaggle account and go to Account settings -> "Create New API Token".
   This downloads a `kaggle.json` file.
2. Install the CLI and place your credentials:
   ```bash
   pip install kaggle
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Download and unzip:
   ```bash
   kaggle datasets download -d aryashah2k/mango-leaf-disease-dataset -p data/mango --unzip
   ```
4. Reorganize the extracted folders so you end up with:
   ```
   data/mango/train/Anthracnose/*.jpg
   data/mango/train/Healthy/*.jpg
   ... (one folder per class)
   data/mango/val/Anthracnose/*.jpg
   ...
   ```
   Most Kaggle mango datasets ship as one folder per class with no train/val
   split — write a small script (or do it manually) to move ~20% of each
   class's images into `val/` and the rest into `train/`.

   Quick 80/20 split script:
   ```bash
   python - <<'EOF'
   import os, shutil, random

   SRC = "data/mango/raw_downloaded_folder"  # adjust to wherever Kaggle extracted to
   TRAIN = "data/mango/train"
   VAL = "data/mango/val"
   SPLIT = 0.8

   for cls in os.listdir(SRC):
       cls_dir = os.path.join(SRC, cls)
       if not os.path.isdir(cls_dir):
           continue
       files = os.listdir(cls_dir)
       random.shuffle(files)
       cut = int(len(files) * SPLIT)
       os.makedirs(os.path.join(TRAIN, cls), exist_ok=True)
       os.makedirs(os.path.join(VAL, cls), exist_ok=True)
       for f in files[:cut]:
           shutil.copy(os.path.join(cls_dir, f), os.path.join(TRAIN, cls, f))
       for f in files[cut:]:
           shutil.copy(os.path.join(cls_dir, f), os.path.join(VAL, cls, f))
   EOF
   ```

## Important
Make sure the class folder names you end up with exactly match the keys
under `"mango"` in `treatment_db.json` (e.g. `Anthracnose`, `Healthy`),
or update `treatment_db.json` to match your folder names.
