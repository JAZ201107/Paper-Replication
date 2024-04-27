from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import tqdm as tqdm


from misc import *


def train_one_epoch(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    lr_scheduler,
    device: torch.device,
):
    model.train()
    pass


@torch.no_grad()
def validation(
    model: torch.nn.Module,
    valid_dataloader: torch.utils.data.DataLoader,
    metrics: Dict[str, function],
    device: torch.device,
):
    model.eval()

    pass


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    lr_scheduler,
    device: torch.device,
):
    pass


@torch.no_grad()
def run(
    model: torch.nn.Module,
    model_weight_path: str,
    valid_dataloader: torch.utils.data.DataLoader,
    metrics: Dict[str, function],
    device: torch.device,
):
    model.eval()

    state = torch.load(model_weight_path)
    
