"""Small helpers shared by all scripts: reproducible seeds and YAML configs."""
import os
import random

import numpy as np
import yaml


def set_seed(seed: int):
    """Seed every random number generator we use, so a rerun gives the same results."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is not needed for the feature baselines
        pass


def load_config(path) -> dict:
    """Read one experiment's settings from a YAML file in configs/."""
    with open(path) as f:
        return yaml.safe_load(f)
