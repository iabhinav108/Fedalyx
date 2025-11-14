#!/usr/bin/env python3
"""
partition.py

Usage:
    python partition.py \
        --train_dir /path/to/data/train \
        --out_dir /path/to/outputs/partitions \
        --global_frac 0.05 \
        --num_clients 8 \
        --dir_alpha 0.5 \
        --seed 42

Outputs:
    out_dir/
      global_teacher/CLASS/*.png
      clients/
        client_0/CLASS/*.png
        client_1/...
      partitions.json   # metadata: global list + per-client lists
"""
import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List

import numpy as np


def gather_class_files(train_dir: str) -> Dict[str, List[str]]:
    """
    Assumes ImageFolder layout: train_dir/<class>/*.*
    Returns dict class_name -> list of absolute file paths
    """
    classes = {}
    for cls in sorted(os.listdir(train_dir)):
        cls_path = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        files = []
        for fname in sorted(os.listdir(cls_path)):
            fpath = os.path.join(cls_path, fname)
            if os.path.isfile(fpath):
                files.append(fpath)
        classes[cls] = files
    return classes


def make_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_files(file_list: List[str], dst_dir: str):
    make_dir(dst_dir)
    for src in file_list:
        basename = os.path.basename(src)
        dst = os.path.join(dst_dir, basename)
        # avoid copying onto itself if same path
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        shutil.copy2(src, dst)


def sample_global_teacher(classes_files: Dict[str, List[str]], global_frac: float, seed: int):
    """
    Create a balanced global teacher set ~global_frac of total, but balanced across classes.
    Strategy:
      - desired_total = max(1, round(total_images * global_frac))
      - per_class = desired_total // num_classes, distribute remainder
      - if a class has fewer images than per_class, take all and reduce allocation for others.
    Returns:
      global_selection: dict class -> list(files)
      remaining_pool: dict class -> list(files)
    """
    rng = random.Random(seed)

    total = sum(len(v) for v in classes_files.values())
    num_classes = len(classes_files)
    desired_total = max(1, int(round(total * global_frac)))
    # target per class (balanced)
    base = desired_total // num_classes
    remainder = desired_total - base * num_classes

    per_class_target = {cls: base for cls in classes_files}
    # distribute remainder to first classes deterministically (sorted order)
    cls_names = sorted(classes_files.keys())
    for i in range(remainder):
        per_class_target[cls_names[i]] += 1

    global_selection = {}
    remaining_pool = {}

    # sample
    for cls, files in classes_files.items():
        n_avail = len(files)
        n_take = min(per_class_target[cls], n_avail)
        files_shuffled = files[:]  # copy
        rng.shuffle(files_shuffled)
        take = files_shuffled[:n_take]
        rest = files_shuffled[n_take:]
        global_selection[cls] = take
        remaining_pool[cls] = rest

    # If for some reason we undersampled (e.g., many classes had <target), try to fill remaining slots
    cur_total = sum(len(v) for v in global_selection.values())
    if cur_total < desired_total:
        needed = desired_total - cur_total
        # flatten remaining across classes and sample extra
        flat_remaining = []
        for cls, files in remaining_pool.items():
            flat_remaining.extend(files)
        rng.shuffle(flat_remaining)
        extra = flat_remaining[:needed]
        # add extras to classes by their actual class
        for f in extra:
            cls = os.path.basename(os.path.dirname(f))
            global_selection[cls].append(f)
            remaining_pool[cls].remove(f)

    return global_selection, remaining_pool


def dirichlet_partition(remaining_pool: Dict[str, List[str]], num_clients: int, alpha: float, min_size: int = 1, seed: int = 42):
    """
    Standard Dirichlet non-IID partition.
    For each class c, draw p ~ Dir(alpha) over clients; assign class c's samples to clients proportionally.
    Guarantees at least min_size examples per client by resampling if necessary.
    Returns: dict client_id -> dict class -> list(files)
    """
    rng = np.random.RandomState(seed)
    classes = sorted(remaining_pool.keys())
    client_data = {i: defaultdict(list) for i in range(num_clients)}

    # loop/resample until every client has >= min_size samples (or until max tries)
    max_tries = 100
    for attempt in range(max_tries):
        # clear assignment
        for i in range(num_clients):
            client_data[i] = defaultdict(list)

        for cls in classes:
            files = remaining_pool[cls][:]
            n = len(files)
            if n == 0:
                continue
            # sample a Dirichlet distribution
            proportions = rng.dirichlet([alpha] * num_clients)
            # convert proportions to counts (at least zero)
            counts = (proportions * n).astype(int)
            # Fix rounding: distribute remaining counts starting from largest frac parts
            remainder = n - counts.sum()
            if remainder > 0:
                # compute fractional parts
                frac = proportions * n - counts
                order = np.argsort(-frac)  # descending fractional parts
                for idx in order[:remainder]:
                    counts[idx] += 1
            # shuffle files and split
            rng.shuffle(files)
            start = 0
            for client_idx in range(num_clients):
                cnt = int(counts[client_idx])
                if cnt > 0:
                    assigned = files[start:start + cnt]
                    client_data[client_idx][cls].extend(assigned)
                    start += cnt
                # else nothing assigned

        # check sizes
        client_sizes = [sum(len(v) for v in client_data[i].values()) for i in range(num_clients)]
        if min(client_sizes) >= min_size:
            # OK
            return client_data

        # otherwise retry with different random seed sample
        rng = np.random.RandomState(seed + attempt + 1)

    raise RuntimeError("Dirichlet partition failed to produce all clients with min_size >= {}".format(min_size))


def write_partitions(out_dir: str, global_selection: Dict[str, List[str]], client_data: Dict[int, Dict[str, List[str]]]):
    """
    Create directories and copy files.
    Structure:
      out_dir/global_teacher/<class>/*
      out_dir/clients/client_{i}/<class>/*
    Also write partitions.json summarizing lists.
    """
    global_dir = os.path.join(out_dir, "global_teacher")
    clients_root = os.path.join(out_dir, "clients")
    make_dir(global_dir)
    make_dir(clients_root)

    # copy global
    partitions = {"global_teacher": {}, "clients": {}}
    for cls, files in global_selection.items():
        dst_cls_dir = os.path.join(global_dir, cls)
        make_dir(dst_cls_dir)
        copy_files(files, dst_cls_dir)
        partitions["global_teacher"][cls] = [os.path.join(dst_cls_dir, os.path.basename(f)) for f in files]

    # copy clients
    for client_idx, cls_map in client_data.items():
        client_dir = os.path.join(clients_root, f"client_{client_idx}")
        make_dir(client_dir)
        partitions["clients"][str(client_idx)] = {}
        for cls, files in cls_map.items():
            dst_cls_dir = os.path.join(client_dir, cls)
            make_dir(dst_cls_dir)
            copy_files(files, dst_cls_dir)
            partitions["clients"][str(client_idx)][cls] = [os.path.join(dst_cls_dir, os.path.basename(f)) for f in files]

    # save json (paths are the new copied paths)
    json_path = os.path.join(out_dir, "partitions.json")
    with open(json_path, "w") as f:
        json.dump(partitions, f, indent=2)
    print(f"Wrote partitions metadata to {json_path}")


def main(args):
    # Validate input train dir
    if not os.path.isdir(args.train_dir):
        raise FileNotFoundError(f"train_dir not found: {args.train_dir}")

    classes_files = gather_class_files(args.train_dir)
    if len(classes_files) == 0:
        raise RuntimeError("No class directories found under train_dir")

    print("Found classes:", list(classes_files.keys()))
    total = sum(len(v) for v in classes_files.values())
    print(f"Total images in train_dir: {total}")

    global_selection, remaining_pool = sample_global_teacher(classes_files, args.global_frac, seed=args.seed)
    g_total = sum(len(v) for v in global_selection.values())
    remaining_total = sum(len(v) for v in remaining_pool.values())
    print(f"Selected {g_total} images for Global Teacher set (target frac={args.global_frac})")
    print(f"Remaining for clients pool: {remaining_total}")

    # run Dirichlet partition on remaining pool
    client_data = dirichlet_partition(remaining_pool, args.num_clients, args.dir_alpha, min_size=args.min_client_size, seed=args.seed)
    client_sizes = [sum(len(v) for v in client_data[i].values()) for i in range(args.num_clients)]
    for i, s in enumerate(client_sizes):
        print(f"Client {i} size: {s}")

    # write partitions (copy files into out_dir)
    make_dir(args.out_dir)
    write_partitions(args.out_dir, global_selection, client_data)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition dataset into Global Teacher set (balanced small) "
                                                 "and Dirichlet-partitioned Client sets.")
    parser.add_argument("--train_dir", type=str, required=True, help="ImageFolder-style train directory (train/<class>/*).")
    parser.add_argument("--out_dir", type=str, required=True, help="Where to write global_teacher/ and clients/ directories.")
    parser.add_argument("--global_frac", type=float, default=0.05, help="Fraction of total images to allocate to Global Teacher set (default 0.05).")
    parser.add_argument("--num_clients", type=int, default=8, help="Number of simulated clients.")
    parser.add_argument("--dir_alpha", type=float, default=0.5, help="Dirichlet concentration parameter (smaller => more heterogeneity).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--min_client_size", type=int, default=1, help="Minimum samples per client required (resample if violated).")
    args = parser.parse_args()
    main(args)
