# 01_train_teacher.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
import json

from src.data_utils.dataloaders import get_global_dataloader

# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_labels = []
    all_logits = []

    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.float().to(device).unsqueeze(1)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        all_labels.append(yb.cpu().numpy())
        all_logits.append(out.detach().cpu().numpy())

    labels = np.vstack(all_labels)
    logits = np.vstack(all_logits)

    # AUC (use raw logits)
    auc = roc_auc_score(labels, logits)

    # Accuracy using logits threshold at 0 (since using BCEWithLogits)
    preds = (logits >= 0).astype(int)
    labels_int = labels.astype(int)
    acc = (preds == labels_int).mean()

    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, auc, acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device)
            yb = yb.float().to(device).unsqueeze(1)

            out = model(xb)
            loss = criterion(out, yb)

            total_loss += loss.item() * xb.size(0)
            all_labels.append(yb.cpu().numpy())
            all_logits.append(out.cpu().numpy())

    labels = np.vstack(all_labels)
    logits = np.vstack(all_logits)

    # AUC (use raw logits)
    auc = roc_auc_score(labels, logits)

    # Accuracy using logits threshold at 0 (since using BCEWithLogits)
    preds = (logits >= 0).astype(int)
    labels_int = labels.astype(int)
    acc = (preds == labels_int).mean()

    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, auc, acc


# -------------------------
# Main
# -------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load global teacher dataloader
    global_loader, global_ds = get_global_dataloader(
        json_path=args.partitions_json,
        batch_size=args.batch_size,
        img_size=args.img_size,
        shuffle=True,
        num_workers=0
    )

    # Build pretrained ResNet18 (RGB input, output=1 logit)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    # Compute pos_weight
    labels = [lab for _, lab, in [(p, l) for p, l in global_ds.files]]
    pos_idx = 1 if "PNEUMONIA" in global_ds.transform.__str__() else 1
    n_pos = sum(1 for l in labels if l == pos_idx)
    n_neg = len(labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train simple epochs
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc, train_acc = train_one_epoch(model, global_loader, optimizer, criterion, device)
        # small dataset → same loader used for validation
        val_loss, val_auc, val_acc = train_loss, train_auc, train_acc

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_auc": train_auc,
            "train_acc": float(train_acc),
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_acc": float(val_acc)
        })

        print(f"[{epoch}] loss={train_loss:.4f} | auc={train_auc:.4f} | acc={train_acc:.4f}")

        # Save best model
        torch.save(model.state_dict(), f"{args.save_dir}/teacher_model.pth")

    # Save training history
    with open(f"{args.save_dir}/teacher_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Teacher training complete.")
    print("Saved:", f"{args.save_dir}/teacher_model.pth")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions_json", type=str, default="outputs/partitions/partitions.json")
    parser.add_argument("--save_dir", type=str, default="outputs/models")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    main(args)
