# scripts/run_partition.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
import config
from src.data_utils.partition import main as partition_main

if __name__ == "__main__":
    args = SimpleNamespace(
        train_dir = config.ORIG_TRAIN,
        out_dir = config.PARTITIONS_OUT,
        global_frac = config.GLOBAL_FRAC,
        num_clients = config.NUM_CLIENTS,
        dir_alpha = config.DIR_ALPHA,
        seed = config.SEED,
        min_client_size = 1,
    )
    partition_main(args)
