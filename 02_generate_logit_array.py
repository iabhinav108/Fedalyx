# 02_generate_logit_array.py
"""
Generate and save teacher logits for every image in the Client Pool (D).
Saves a dict: { resolved_path_str : logits_numpy_array } to out_path.
Minimal comments only.
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from tqdm import tqdm

from src.data_utils.datasets import FileListDataset, load_partitions_json
from src.data_utils.dataloaders import build_transforms  # reuse your transform builder

# build teacher model (ResNet18 -> single logit)
def build_teacher(num_classes=1, ckpt_path=None, device="cuda"):
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)  # single logit (B,1)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        # if state is a full checkpoint dict, try common keys
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    return model.to(device).eval()

def collate_all_client_files(clients_dict):
    files = []
    for cid, lst in sorted(clients_dict.items(), key=lambda x: int(x[0])):
        files.extend(lst)
    return files

def main(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    # load partitions and collate client pool files
    _, clients, _ = load_partitions_json(args.partitions_json)
    all_files = collate_all_client_files(clients)   # list of (path,label)
    if len(all_files) == 0:
        raise RuntimeError("No client files found in partitions.json")

    # dataset & loader (shuffle=False to keep stable order)
    tf = build_transforms(img_size=args.img_size, is_train=False, use_clahe=args.use_clahe)
    ds = FileListDataset(all_files, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # load teacher
    teacher = build_teacher(ckpt_path=args.teacher_ckpt, device=device)

    logits_map = {}
    paths_order = []

    with torch.no_grad():
        for xb, yb, paths in tqdm(loader, desc="Generating logits"):
            xb = xb.to(device)
            out = teacher(xb)            # (B,1)
            out_cpu = out.detach().cpu().numpy()
            for i, p in enumerate(paths):
                # canonicalize path string (POSIX style)
                key = str(Path(p).resolve().as_posix())
                logits_map[key] = out_cpu[i].astype(np.float32)  # store numpy float32
                paths_order.append(key)

    # save
    out_p = Path(args.out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"logits_map": logits_map, "paths_order": paths_order}, str(out_p))
    print(f"Saved logits -> {out_p}  (num_items={len(paths_order)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, required=True, help="Path to teacher_model.pth")
    parser.add_argument("--partitions_json", type=str, default="outputs/partitions/partitions.json")
    parser.add_argument("--out_path", type=str, default="outputs/logit_array/logit_server_array.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)  # 0 avoids pickling issues on Windows
    parser.add_argument("--use_clahe", action="store_true", help="Apply CLAHE during forward (matches dataloader)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    main(args)
