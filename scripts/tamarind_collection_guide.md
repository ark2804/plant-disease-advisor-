# Tamarind Leaf Disease Data — Collection Guide

**No public tamarind leaf disease image dataset currently exists.**
(The one "Tamarind" dataset on Mendeley — https://data.mendeley.com/datasets/thd2zdrdzx —
is about tamarind *pods*, shelled vs unshelled, not leaves. Not usable here.)

You'll need to collect and label your own images. Here's a practical plan.

## 1. Decide your disease classes first

Common tamarind leaf issues reported in agricultural literature include:
- Leaf spot (Cercospora spp.)
- Leaf rust
- Leaf blight
- Dieback
- Powdery mildew
- Healthy

**Before you start photographing, confirm with a local agricultural
extension officer, horticulture department, or plant pathologist which
of these (or others) are actually present in your region** — don't just
guess from symptoms. Misdiagnosed training labels will make the model
confidently wrong later.

## 2. Target sample size

Aim for a minimum of 150–300 images per class (more is better). With
5-6 classes that's roughly 900–1,800 images total minimum for a usable
first model. Fewer than ~100 images/class will likely overfit badly with
a deep CNN, even using transfer learning.

## 3. Collection protocol (do this consistently across every photo)

- **Background:** place the leaf against a plain, contrasting background
  (a sheet of card or cloth) to reduce noise.
- **Lighting:** natural daylight, avoid direct flash or harsh shadows.
- **Distance/framing:** fill most of the frame with the leaf; take both
  a whole-leaf shot and a close-up of the lesion/symptom.
- **Multiple angles:** front and back of the leaf (some fungal symptoms
  are more visible on the underside).
- **Diversity:** photograph leaves at different severity stages (early,
  moderate, advanced) and from different trees/locations, not just one tree.
- **Balance:** try to collect roughly similar numbers of healthy vs. each
  disease class, so the model doesn't just learn to always predict
  "healthy".

## 4. Labeling and organizing

As you collect, immediately sort into folders — don't batch-label later
from memory:
```
data/tamarind/train/Healthy/img001.jpg
data/tamarind/train/LeafSpot/img002.jpg
data/tamarind/val/Healthy/img050.jpg
...
```
Keep ~80% of each class in `train/` and ~20% in `val/`.

## 5. Update the knowledge base

Once your classes are confirmed, replace the `_PLACEHOLDER_leaf_spot`
entry in `treatment_db.json` under `"tamarind"` with real entries — one
per class, named to exactly match your folder names. Get the actual
fertilizer/pesticide recommendations from your local agricultural
extension office, since tamarind treatment guidance is less standardized
online than for mango.

## 6. If you need more data before you have enough field photos

Consider:
- Reaching out to agricultural universities/research stations in
  tamarind-growing regions (Tamil Nadu, Andhra Pradesh, Karnataka in
  India, for example) — some maintain reference image collections.
- Using web-scraped images only as a *last resort supplement*, and only
  if you can verify the diagnosis is correct (mislabeled scraped images
  are worse than no data).
- Starting with a coarser Healthy vs. Diseased binary classifier for
  tamarind while you build up enough data for full disease-type
  classification.
