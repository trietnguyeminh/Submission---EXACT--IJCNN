#!/usr/bin/env python3
"""
Nom Character OCR — improved baseline for the UITAIC challenge.

Stays strictly within the contest rules:
  - Only torch + torchvision are used.
  - Pretrained weights come from torchvision (ImageNet) only.
  - No external data, no test data for training/pseudo-labeling.

Key differences vs. the provided baseline:
  - Path bug fixed: dataset uses test/, not public_test/.
  - Default backbone is ResNet-50 (more capacity for 3923 classes).
  - Input size 96x96 (preserves stroke detail vs. 64x64).
  - ImageNet normalization (matches the pretrained weights).
  - Stronger augmentation: rotation, affine, color jitter, perspective,
    RandomErasing. NO horizontal/vertical flips (Nom chars are not
    flip-invariant — flipping creates a different character).
  - CrossEntropy with label_smoothing=0.1.
  - AdamW + linear warmup + cosine schedule.
  - Optional Test-Time Augmentation (TTA): averages logits from the
    center view and a few small-shift / scale variants.
  - Submission is auto-zipped into submission.zip with submission.csv
    placed directly inside (no parent folder), per the contest spec.
"""

import os
import json
import math
import random
import zipfile
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm


# ==========================================================================
# Reproducibility
# ==========================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================================================================
# Dataset
# ==========================================================================
class NomOCRDataset(Dataset):
    """
    Single-character Nom OCR dataset.
    Split-folder ('train' or 'test') is passed in so the same class works
    for both training and inference.
    """

    def __init__(self, root_dir, dataframe, split_folder, char2idx=None,
                 transform=None, is_train=True):
        self.root_dir = root_dir
        self.dataframe = dataframe.reset_index(drop=True)
        self.split_folder = split_folder  # e.g. "train" or "test"
        self.char2idx = char2idx
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = row['image']
        img_path = os.path.join(self.root_dir, self.split_folder, "images", img_name)

        # Always RGB so that ImageNet-pretrained 3-channel models work.
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label_idx = self.char2idx[row['label']]
        else:
            label_idx = -1
        return image, label_idx


# ==========================================================================
# Model
# ==========================================================================
def get_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a torchvision classification model with the head replaced for
    `num_classes` outputs. Allowed backbones (any classifier from
    torchvision.models works, but these are the practical picks):
      - resnet18 / resnet34 / resnet50
      - efficientnet_b0 / efficientnet_b1
    """
    name = name.lower()

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)

    elif name == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b1(weights=weights)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)

    else:
        raise ValueError(f"Unknown model: {name}")

    print(f"[*] Built {name} (pretrained={pretrained}), head -> {num_classes} classes.")
    return model


# ==========================================================================
# Transforms
# ==========================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int):
    """
    Note: NO horizontal/vertical flips. Flipping a Nom character produces
    a different (or non-existent) character — it is a label-changing
    augmentation here.
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # Small rotations only — keeps the character recognizable.
        transforms.RandomRotation(degrees=5, fill=255),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.06, 0.06),
            scale=(0.92, 1.08),
            shear=3,
            fill=255,
        ),
        transforms.RandomPerspective(distortion_scale=0.10, p=0.3, fill=255),
        transforms.ColorJitter(brightness=0.20, contrast=0.20),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # Erase a small patch — robustness to stains/cracks in scans.
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.5, 2.0)),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


# ==========================================================================
# LR schedule: linear warmup -> cosine decay
# ==========================================================================
def build_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==========================================================================
# Splitting
# ==========================================================================
def make_split(df, val_frac=0.15, seed=42):
    """
    Stratified split that respects singleton classes:
      - Classes with 1 sample: all go to train (no val signal possible).
      - Classes with >=2 samples: stratified split.
    """
    counts = df['label'].value_counts()
    singles = set(counts[counts == 1].index)
    df_single = df[df['label'].isin(singles)]
    df_multi = df[~df['label'].isin(singles)]

    if len(df_multi) > 0:
        train_multi, val_multi = train_test_split(
            df_multi, test_size=val_frac, random_state=seed,
            stratify=df_multi['label']
        )
        train_df = pd.concat([train_multi, df_single]).reset_index(drop=True)
        val_df = val_multi.reset_index(drop=True)
    else:
        train_df = df_single.reset_index(drop=True)
        val_df = pd.DataFrame(columns=df.columns)
    return train_df, val_df


# ==========================================================================
# One training epoch
# ==========================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_label):
    model.train()
    loss_sum, n, preds, targets = 0.0, 0, [], []
    pbar = tqdm(loader, desc=epoch_label, leave=False)
    for images, y in pbar:
        images, y = images.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        loss_sum += loss.item() * bs
        n += bs
        preds.append(logits.argmax(1).detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
        pbar.set_postfix(loss=f"{loss_sum / n:.4f}")

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return loss_sum / max(1, n), f1_score(targets, preds, average='macro', zero_division=0)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, epoch_label):
    model.eval()
    if len(loader) == 0:
        return 0.0, 0.0
    loss_sum, n, preds, targets = 0.0, 0, [], []
    pbar = tqdm(loader, desc=epoch_label, leave=False)
    for images, y in pbar:
        images, y = images.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, y)
        bs = images.size(0)
        loss_sum += loss.item() * bs
        n += bs
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return loss_sum / max(1, n), f1_score(targets, preds, average='macro', zero_division=0)


# ==========================================================================
# Inference with optional TTA
# ==========================================================================
@torch.no_grad()
def predict_test(model, loader, device, tta: bool = True):
    model.eval()
    all_logits = []
    for images, _ in tqdm(loader, desc="Predict"):
        images = images.to(device, non_blocking=True)
        logits = model(images).softmax(1)

        if tta:
            # No flips (would change the character). Use small zoom + shift.
            shifts = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]
            tta_logits = logits.clone()
            for dx, dy in shifts[1:]:
                shifted = torch.roll(images, shifts=(dy, dx), dims=(2, 3))
                tta_logits += model(shifted).softmax(1)
            logits = tta_logits / (1 + len(shifts) - 1)

        all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    return all_logits.argmax(1).numpy()


# ==========================================================================
# Main
# ==========================================================================
def main():
    p = argparse.ArgumentParser(description="Improved Nom Character OCR")
    p.add_argument("--mode", choices=["train", "submit", "both"], default="both")
    p.add_argument("--data-dir", default="dataset",
                   help="Folder containing train/ and test/")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--model", default="resnet50",
                   choices=["resnet18", "resnet34", "resnet50",
                            "efficientnet_b0", "efficientnet_b1"])
    p.add_argument("--img-size", type=int, default=96)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-tta", action="store_true", help="Disable TTA at inference")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train from scratch (not recommended)")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab_path = out / "vocab.json"
    weights_path = out / "best_model.pth"
    sub_csv_path = out / "submission.csv"
    sub_zip_path = out / "submission.zip"

    train_tf, eval_tf = build_transforms(args.img_size)

    # --------------------------- TRAIN ---------------------------
    if args.mode in ("train", "both"):
        labels_csv = Path(args.data_dir) / "train" / "labels.csv"
        if not labels_csv.exists():
            raise FileNotFoundError(f"Missing {labels_csv}")

        print(f"[*] Reading {labels_csv}")
        df = pd.read_csv(labels_csv)
        unique_chars = sorted(df['label'].unique().tolist())
        char2idx = {c: i for i, c in enumerate(unique_chars)}
        idx2char = {i: c for c, i in char2idx.items()}
        num_classes = len(unique_chars)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump({"vocab": unique_chars}, f, ensure_ascii=False, indent=2)
        print(f"[*] {len(df)} samples, {num_classes} classes. Vocab -> {vocab_path}")

        train_df, val_df = make_split(df, val_frac=args.val_frac, seed=args.seed)
        print(f"[*] Split: {len(train_df)} train / {len(val_df)} val")

        train_ds = NomOCRDataset(args.data_dir, train_df, "train",
                                 char2idx=char2idx, transform=train_tf, is_train=True)
        val_ds = NomOCRDataset(args.data_dir, val_df, "train",
                               char2idx=char2idx, transform=eval_tf, is_train=True)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

        model = get_model(args.model, num_classes,
                          pretrained=not args.no_pretrained).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)

        best_val_f1 = -1.0
        history = []
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                f"Epoch {epoch:02d}/{args.epochs} [train]"
            )
            va_loss, va_f1 = eval_one_epoch(
                model, val_loader, criterion, device,
                f"Epoch {epoch:02d}/{args.epochs} [val]"
            )
            scheduler.step()
            lr_now = optimizer.param_groups[0]["lr"]
            history.append({"epoch": epoch, "train_loss": tr_loss, "train_f1": tr_f1,
                            "val_loss": va_loss, "val_f1": va_f1, "lr": lr_now})
            print(f"E{epoch:02d} | lr {lr_now:.2e} | "
                  f"train loss {tr_loss:.4f} f1 {tr_f1:.4f} | "
                  f"val loss {va_loss:.4f} f1 {va_f1:.4f}")
            if va_f1 > best_val_f1:
                best_val_f1 = va_f1
                torch.save(model.state_dict(), weights_path)
                print(f"  ↳ new best (val macro-F1 = {best_val_f1:.4f}) saved to {weights_path}")

        # Persist history (handy for plotting)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"[*] Training done. Best val macro-F1 = {best_val_f1:.4f}")

    # --------------------------- SUBMIT ---------------------------
    if args.mode in ("submit", "both"):
        if not vocab_path.exists():
            raise FileNotFoundError(f"Missing vocab.json — run training first.")
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights — run training first.")

        with open(vocab_path, encoding="utf-8") as f:
            unique_chars = json.load(f)["vocab"]
        num_classes = len(unique_chars)
        idx2char = {i: c for i, c in enumerate(unique_chars)}

        # Re-instantiate the same architecture and load weights.
        model = get_model(args.model, num_classes, pretrained=False).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))

        sample_csv = Path(args.data_dir) / "test" / "sample_submission.csv"
        if not sample_csv.exists():
            raise FileNotFoundError(f"Missing {sample_csv}")
        sub_df = pd.read_csv(sample_csv)
        print(f"[*] Test entries: {len(sub_df)}")

        test_ds = NomOCRDataset(args.data_dir, sub_df, "test",
                                transform=eval_tf, is_train=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

        pred_idx = predict_test(model, test_loader, device, tta=not args.no_tta)
        sub_df['label'] = [idx2char[i] for i in pred_idx]
        sub_df.to_csv(sub_csv_path, index=False)
        print(f"[*] Wrote {sub_csv_path}")

        # Per contest spec: submission.zip must contain submission.csv at
        # the root (no parent folder).
        with zipfile.ZipFile(sub_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(sub_csv_path, arcname="submission.csv")
        print(f"[*] Wrote {sub_zip_path} (contains submission.csv at root)")


if __name__ == "__main__":
    main()
