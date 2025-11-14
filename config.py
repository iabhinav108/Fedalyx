# config.py
PROJECT_ROOT = r"E:/Semester-7/AI In Healthcare/Project/Fedalyx"
DATA_ROOT = f"{PROJECT_ROOT}/data"
ORIG_TRAIN = f"{DATA_ROOT}/original/train"
ORIG_TEST  = f"{DATA_ROOT}/original/test"
PARTITIONS_OUT = f"{PROJECT_ROOT}/outputs/partitions"

# federated hyperparams
NUM_CLIENTS = 10
GLOBAL_FRAC = 0.05
DIR_ALPHA = 0.5
SEED = 22
