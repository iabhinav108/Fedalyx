# scripts/preview_split.py  (optional helper)
import collections
from torch.utils.data import DataLoader
from config import CONFIG
from src.data_utils.partition import load_chestxray_teacher_client

def count_by_label(items):
    c = collections.Counter([e["label"] for e in items])
    return dict(c)

if __name__ == "__main__":
    cfg = CONFIG
    (train_tfm, test_tfm), G, D, T = load_chestxray_teacher_client(
        data_root=cfg["paths"]["data_root"],
        img_size=cfg["data"]["img_size"],
        mean=cfg["data"]["normalize_mean"],
        std=cfg["data"]["normalize_std"],
        teacher_pct=cfg["data"]["teacher_split_pct"],
        seed=cfg["seed"],
    )
    print("Teacher (G) size:", len(G), "| class counts:", count_by_label(G))
    print("Client pool (D) size:", len(D), "| class counts:", count_by_label(D))
    print("Test size:", len(T))
