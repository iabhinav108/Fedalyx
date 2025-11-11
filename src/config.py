# config.py
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "paths": {
        "data_root": os.path.join(ROOT_DIR, "data", "chest_xray"),  # expects train/ and test/
        "outputs_root": os.path.join(ROOT_DIR, "outputs"),
        "teacher_ckpt": os.path.join(ROOT_DIR, "outputs", "models", "teacher_model.pth"),
    },
    "data": {
        "num_classes": 2,              # NORMAL (0), PNEUMONIA (1)
        "img_size": 224,
        "teacher_split_pct": 0.05,     # 5% IID for teacher
        "workers": 4,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std":  [0.229, 0.224, 0.225],
    },
    "optim": {
        "batch_size": 32,
        "epochs": 10,
        "lr": 3e-4,
        "weight_decay": 1e-4,
    },
    "seed": 1337,
}
