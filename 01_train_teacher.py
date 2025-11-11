import torch, argparse, os
from torch.utils.data import DataLoader
from src.models.teacher_model import build_teacher
from src.data_utils.datasets import ImageListDataset
from src.core.loss import KDLoss
from config import CONFIG

def train_one_epoch(model, loader, opt, device):
    model.train()
    ce = torch.nn.CrossEntropyLoss()
    for x,y,_ in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = ce(model(x), y)
        loss.backward()
        opt.step()

def validate(model, loader, device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y,_ in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item(); total += y.numel()
    return correct/total

def main():
    cfg = CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_items = []
    val_items = []

    train_ds = ImageListDataset(train_items, transform=None)  # plug your tfms
    val_ds   = ImageListDataset(val_items,   transform=None)
    train_dl = DataLoader(train_ds, batch_size=cfg["federation"]["batch_size"], shuffle=True, num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["federation"]["batch_size"], shuffle=False, num_workers=4)

    model = build_teacher(num_classes=cfg["data"]["num_classes"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["federation"]["optimizer"]["lr"], weight_decay=cfg["federation"]["optimizer"]["weight_decay"])

    best = 0.0
    for epoch in range(10):  # adjust
        train_one_epoch(model, train_dl, opt, device)
        acc = validate(model, val_dl, device)
        if acc > best:
            best = acc
            os.makedirs("outputs/models", exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "arch": "resnet50",
                "num_classes": cfg["data"]["num_classes"],
                "epoch": epoch,
                "metrics": {"val_acc": acc},
                "version": "v1"
            }, "outputs/models/teacher_model.pth")
    print("best_val_acc:", best)

if __name__ == "__main__":
    main()
