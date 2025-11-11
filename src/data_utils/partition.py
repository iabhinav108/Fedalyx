# src/data_utils/partition.py
import os
import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from torchvision import datasets, transforms

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_transforms(img_size: int, mean, std):
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    tfm = transforms.Compose(base)
    return tfm, tfm  # (train_tfm, test_tfm)

def _gather_items_from_imagefolder(dataset: datasets.ImageFolder) -> List[dict]:
    """
    Convert an ImageFolder to a list of dicts:
    {"path": str, "label": int, "id": str}
    id is the file path relative to dataset.root (stable, used later to match logits).
    """
    items = []
    for path, label in dataset.samples:
        rel = os.path.relpath(path, dataset.root).replace(os.sep, "/")
        items.append({"path": path, "label": int(label), "id": rel})
    return items

def _stratified_iid_subset(
    items: List[dict],
    teacher_pct: float,
    seed: int
) -> Tuple[List[dict], List[dict]]:
    """
    Create a small IID (class-proportional) subset for the teacher (~teacher_pct of total).
    Returns (teacher_items, remaining_items).
    """
    set_seed(seed)
    by_class = defaultdict(list)
    for e in items:
        by_class[e["label"]].append(e)

    total = len(items)
    target = max(1, int(round(teacher_pct * total)))

    # Proportional quotas per class (rounding-safe)
    counts = {c: len(v) for c, v in by_class.items()}
    total_c = sum(counts.values())
    quotas = {c: int(round(target * (counts[c] / total_c))) for c in counts}

    # Fix rounding difference
    diff = target - sum(quotas.values())
    if diff != 0:
        classes = list(quotas.keys())
        i = 0
        while diff != 0:
            c = classes[i % len(classes)]
            if diff > 0:
                quotas[c] += 1
                diff -= 1
            else:
                if quotas[c] > 0:
                    quotas[c] -= 1
                    diff += 1
            i += 1

    teacher, rest = [], []
    for c, lst in by_class.items():
        random.shuffle(lst)
        k = min(quotas[c], len(lst))
        teacher.extend(lst[:k])
        rest.extend(lst[k:])

    random.shuffle(teacher)
    random.shuffle(rest)
    return teacher, rest

def load_chestxray_teacher_client(
    data_root: str,
    img_size: int,
    mean,
    std,
    teacher_pct: float,
    seed: int,
):
    """
    Returns:
      (train_tfm, test_tfm),
      teacher_items: 5% IID subset from train/,
      client_pool_items: remaining 95% from train/,
      test_items: items from official test/
    """
    train_tfm, test_tfm = build_transforms(img_size, mean, std)

    train_dir = os.path.join(data_root, "train")
    test_dir  = os.path.join(data_root, "test")

    train_ds = datasets.ImageFolder(train_dir)
    test_ds  = datasets.ImageFolder(test_dir)

    train_items = _gather_items_from_imagefolder(train_ds)
    test_items  = _gather_items_from_imagefolder(test_ds)

    teacher_items, client_pool_items = _stratified_iid_subset(
        train_items, teacher_pct=teacher_pct, seed=seed
    )

    return (train_tfm, test_tfm), teacher_items, client_pool_items, test_items
