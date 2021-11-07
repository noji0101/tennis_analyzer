"""Load model checkpoint"""
from pathlib import Path

import torch

def load_ckpt(resume: str) -> object:
    """Loads Checkpoint"""
    if not Path(resume).exists():
        raise ValueError(' No checkpoint found !')
    ckpt = torch.load(resume)
    return ckpt