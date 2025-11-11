# src/data_utils/datasets.py
from typing import Callable, Dict, Optional, List
from PIL import Image
import torch
from torch.utils.data import Dataset

class ClientDataset(Dataset):
    """
    Returns (image_tensor, hard_label:int, teacher_logits:Tensor|None)
    For Task 1, pass logit_lookup=None so logits are None.
    """
    def __init__(
        self,
        items: List[dict],
        transform: Optional[Callable],
        logit_lookup: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.items = items
        self.tfm = transform
        self.lut = logit_lookup

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        e = self.items[idx]
        img = Image.open(e["path"]).convert("RGB")
        if self.tfm is not None:
            img = self.tfm(img)
        label = int(e["label"])
        tlog = None
        if self.lut is not None:
            t = self.lut.get(e["id"])
            if t is not None:
                tlog = t.clone().detach()
        return img, label, tlog
