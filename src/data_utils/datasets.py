import json
from pathlib import Path
from typing import List
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class FileListDataset(Dataset):
    """
    file_list: list of (abs_path, label_int)
    transform: torchvision transform
    """
    def __init__(self, file_list: List[tuple], transform=None):
        self.files = file_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label, path  # return path for mapping to logits later

def load_partitions_json(json_path: str = "outputs/partitions/partitions.json"):
    obj = json.loads(Path(json_path).read_text())
    # returns: global_list (list of (path,label)), clients dict -> client_id -> list of (path,label)
    global_list = []
    clients = {}
    class_to_idx = {}  # determine class->idx by sorting class names
    classes = sorted(obj["global_teacher"].keys())
    class_to_idx = {c:i for i,c in enumerate(classes)}
    for cls, paths in obj["global_teacher"].items():
        for p in paths:
            global_list.append((p, class_to_idx[cls]))
    for cid, clsmap in obj["clients"].items():
        lst = []
        for cls, paths in clsmap.items():
            idx = class_to_idx.get(cls, None)
            for p in paths:
                lst.append((p, idx))
        clients[int(cid)] = lst
    return global_list, clients, class_to_idx