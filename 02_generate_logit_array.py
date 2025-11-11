import torch, argparse, os
from torch.utils.data import DataLoader
from src.models.teacher_model import build_teacher
from src.data_utils.datasets import ImageListDataset
from config import CONFIG

@torch.no_grad()
def infer_logits(model, loader, device):
    ids, logits = [], []
    for x, _, sid in loader:
        x = x.to(device)
        log = model(x)              # [B, C]
        logits.append(log.cpu())
        ids.extend(sid)
    return ids, torch.cat(logits, 0)

def main():
    cfg = CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: build items over which you want teacher predictions
    all_items = []  # [{"path":..., "label":..., "id":...}, ...]
    ds = ImageListDataset(all_items, transform=None)  # same transforms as teacher val
    dl = DataLoader(ds, batch_size=cfg["federation"]["batch_size"], shuffle=False, num_workers=4)

    ckpt = torch.load(cfg["logging"]["teacher_ckpt"], map_location="cpu")
    model = build_teacher(num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().to(device)

    ids, logits = infer_logits(model, dl, device)
    os.makedirs("outputs/logit_array", exist_ok=True)
    torch.save({
        "ids": ids,
        "logits": logits.float(),
        "T": cfg["kd"]["temperature"],
        "meta": {"arch": ckpt.get("arch",""), "num_classes": ckpt["num_classes"], "teacher_version": ckpt.get("version","v1")}
    }, "outputs/logit_array/logit_server_array.pt")
    print("Saved logits:", logits.shape)

if __name__ == "__main__":
    main()
