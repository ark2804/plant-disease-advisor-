"""
Train a disease classifier for one plant species.

Example:
    python src/train.py --plant mango --data_dir data/mango --epochs 15
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model, unfreeze_backbone, get_device


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", required=True, choices=["mango", "neem", "tamarind"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--unfreeze_at_epoch", type=int, default=8,
                         help="Epoch at which to unfreeze backbone layers for fine-tuning")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )
    print(f"Classes for {args.plant}: {class_names}")

    model = build_model(num_classes=len(class_names), freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_at_epoch:
            print("Unfreezing backbone layers for fine-tuning...")
            unfreeze_backbone(model)
            optimizer = optim.Adam(model.parameters(), lr=args.lr / 10)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(args.out_dir, f"{args.plant}_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  -> new best model saved to {model_path} (val_acc={val_acc:.4f})")

    classes_path = os.path.join(args.out_dir, f"{args.plant}_classes.json")
    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class index mapping saved to {classes_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
