import os, torch, argparse, json
from torch.utils.data import DataLoader
from collections import defaultdict
from src.models.student_model import build_student
from src.data_utils.datasets import ImageListDataset
from src.core.loss import KDLoss
from config import CONFIG
import pandas as pd

def build_logit_lookup(array_path):
    blob = torch.load(array_path, map_location="cpu")
    ids, logits = blob["ids"], blob["logits"]
    lut = {sid: logits[i] for i, sid in enumerate(ids)}
    T = blob.get("T", 4.0)
    return lut, float(T)

def collate_with_logits(batch, lut):
    # batch: List[(x, y, sid)]
    xs, ys, sids, tlogs = [], [], [], []
    for x,y,sid in batch:
        xs.append(x); ys.append(y); sids.append(sid)
        tlogs.append(lut.get(sid))  # None if missing
    X = torch.stack(xs, 0)
    Y = torch.tensor(ys, dtype=torch.long)
    # pad missing with None -> handled later
    return X, Y, sids, tlogs

def train_one_site(site_id, items, lut, T, cfg, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_student(num_classes=cfg["data"]["num_classes"]).to(device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg["federation"]["optimizer"]["lr"],
                            weight_decay=cfg["federation"]["optimizer"]["weight_decay"])
    kd = KDLoss(T=T, lambda_kd=cfg["kd"]["lambda_kd"])
    bs = cfg["federation"]["batch_size"]
    loader = DataLoader(
        ImageListDataset(items, transform=None),
        batch_size=bs, shuffle=True, num_workers=4,
        collate_fn=lambda b: collate_with_logits(b, lut)
    )

    model.train()
    for ep in range(cfg["federation"]["local_epochs"]):
        for X, Y, sids, tlogs in loader:
            X, Y = X.to(device), Y.to(device)
            # stack available teacher logits; None -> fall back to CE
            if tlogs[0] is None:
                t = None
            else:
                # Some batches may mix present/missing; handle per-sample
                has = [i for i,t in enumerate(tlogs) if t is not None]
                if len(has) == len(tlogs):
                    t = torch.stack(tlogs, 0).to(device)  # [B,C]
                else:
                    # compute loss per sample
                    logits = model(X)
                    loss = 0.0
                    for i in range(X.size(0)):
                        ti = tlogs[i].to(device) if tlogs[i] is not None else None
                        loss = loss + kd(logits[i:i+1], Y[i:i+1], ti)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    continue

            logits = model(X)
            loss = kd(logits, Y, t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    # save student
    os.makedirs(os.path.join(cfg["logging"]["student_dir"]), exist_ok=True)
    path = os.path.join(cfg["logging"]["student_dir"], f"student_client_{site_id}.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "site_id": site_id,
        "num_classes": cfg["data"]["num_classes"],
        "version": "v1"
    }, path)
    return path

def main():
    cfg = CONFIG
    # 1) Load the array (broadcast)
    lut, T = build_logit_lookup(cfg["logging"]["logit_array"])

    # 2) Build per-site splits
    # TODO: plug `src/data_utils/partition.py` returning dict: {site_id: List[items]}
    sites = {}  # e.g., {"0": [...], "1": [...], ...}

    logs = []
    for site_id, items in sites.items():
        ckpt_path = train_one_site(site_id, items, lut, T, cfg, cfg["logging"]["student_dir"])
        logs.append({"site_id": site_id, "ckpt": ckpt_path, "num_samples": len(items)})

    os.makedirs("outputs/logs", exist_ok=True)
    df = pd.DataFrame(logs)
    df.to_csv("outputs/logs/simulation_log.csv", index=False)
    print("Done.")

if __name__ == "__main__":
    main()
