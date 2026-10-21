"""
Dataset loading utilities.

Expects folder layout:
    data/<plant>/train/<ClassName>/*.jpg
    data/<plant>/val/<ClassName>/*.jpg

Uses torchvision.datasets.ImageFolder, so class names come directly from
the subfolder names - make sure these EXACTLY match the keys used in
treatment_db.json for that plant.
"""
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMG_SIZE = 224

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    for d in (train_dir, val_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Expected folder not found: {d}\n"
                f"Populate it with subfolders per disease class, e.g. "
                f"{d}/Healthy/*.jpg, {d}/Anthracnose/*.jpg ..."
            )

    train_dataset = datasets.ImageFolder(train_dir, transform=TRAIN_TRANSFORMS)
    val_dataset = datasets.ImageFolder(val_dir, transform=VAL_TRANSFORMS)

    if len(train_dataset.classes) != len(val_dataset.classes):
        raise ValueError(
            f"Train classes {train_dataset.classes} and val classes "
            f"{val_dataset.classes} don't match. Make sure both train/ and "
            f"val/ have the exact same set of class subfolders."
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_dataset.classes
