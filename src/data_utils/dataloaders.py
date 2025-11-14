# src/data_utils/dataloaders.py
from pathlib import Path
import json
from typing import Tuple, List
from PIL import Image, ImageOps
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import FileListDataset, load_partitions_json

# convert PIL to RGB (top-level function to be picklable on Windows)
def to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")

# simple CLAHE/equalize helper (top-level)
def pil_clahe(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    try:
        import cv2
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        out = clahe.apply(arr)
        return Image.fromarray(out).convert("RGB")
    except Exception:
        try:
            return ImageOps.equalize(img).convert("RGB")
        except Exception:
            return img.convert("RGB")

class ResizePad:
    def __init__(self, target_size: int = 256, interpolation=Image.BILINEAR, pad_value=0):
        self.target_size = int(target_size)
        self.interpolation = interpolation
        self.pad_value = pad_value

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError("Image has zero width or height.")
        scale = min(self.target_size / w, self.target_size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_resized = img.resize((new_w, new_h), resample=self.interpolation)
        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_bottom = self.target_size - new_h - pad_top
        img_padded = ImageOps.expand(img_resized, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.pad_value)
        return img_padded

def build_transforms(img_size: int = 256, is_train: bool = True, use_clahe: bool = True):
    tf_list = []
    tf_list.append(transforms.Lambda(to_rgb))
    tf_list.append(ResizePad(target_size=img_size))
    if use_clahe:
        tf_list.append(transforms.Lambda(pil_clahe))
    if is_train:
        tf_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])
    tf_list.append(transforms.ToTensor())  # keep in [0,1], no normalization
    return transforms.Compose(tf_list)

def get_global_dataloader(json_path: str = "outputs/partitions/partitions.json",
                          batch_size: int = 32,
                          img_size: int = 256,
                          num_workers: int = 0,
                          shuffle: bool = True,
                          use_clahe: bool = True) -> Tuple[DataLoader, FileListDataset]:
    global_list, _, _ = load_partitions_json(json_path)
    tf = build_transforms(img_size=img_size, is_train=False, use_clahe=use_clahe)
    ds = FileListDataset(global_list, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader, ds

def get_client_dataloader(client_id: int,
                          json_path: str = "outputs/partitions/partitions.json",
                          batch_size: int = 16,
                          img_size: int = 256,
                          num_workers: int = 0,
                          shuffle: bool = True,
                          is_train: bool = True,
                          use_clahe: bool = True) -> Tuple[DataLoader, FileListDataset]:
    _, clients, _ = load_partitions_json(json_path)
    if client_id not in clients:
        raise ValueError(f"client_id {client_id} not found in partitions.json")
    file_list = clients[client_id]
    tf = build_transforms(img_size=img_size, is_train=is_train, use_clahe=use_clahe)
    ds = FileListDataset(file_list, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle if is_train else False, num_workers=num_workers, pin_memory=True)
    return loader, ds

if __name__ == "__main__":
    g_loader, g_ds = get_global_dataloader(batch_size=16, img_size=256, num_workers=0)
    print("Global dataset size:", len(g_ds))
    xb, yb, paths = next(iter(g_loader))
    print("Global batch:", xb.shape, yb.shape)

    c_loader, c_ds = get_client_dataloader(client_id=0, batch_size=16, img_size=256, num_workers=0)
    print("Client 0 dataset size:", len(c_ds))
    xb, yb, paths = next(iter(c_loader))
    print("Client batch:", xb.shape, yb.shape)
