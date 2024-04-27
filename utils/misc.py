import random
import numpy as np
import torch
import json
import os


class Params:
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)
        torch.set_deterministic(True)


class AverageMeter:
    """
    Return total and average of some number
    """

    def __init__(self, name="Metric"):
        self.name = name
        self.steps = 0
        self.value = 0

    def update(self, val):
        self.value += val
        self.steps += 1

    def __repr__(self):
        return f"{self.name}: {self.value / float(self.steps)}"

    def __call__(self):
        return self.value / float(self.steps)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def save_dict_to_json(d, json_path):
    with open(json_path, "w") as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    file_path = os.path.join(checkpoint, "last.pth")
