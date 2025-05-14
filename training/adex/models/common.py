from __future__ import annotations
from typing import Callable

import torch
import torchvision
from gymnasium import spaces
from torch import nn
from torchvision import models


class DictForwardModule(nn.Module):

    def __init__(self, module_dict: dict[str, nn.Module] | nn.ModuleDict) -> None:
        super().__init__()
        if isinstance(module_dict, dict):
            module_dict = nn.ModuleDict(module_dict)
        self.module_dict = module_dict

    def forward(self, input: dict[str]) -> dict[str, torch.Tensor]:
        outputs = {}
        for key, mod in self.module_dict.items():
            features = mod(input[key])
            outputs[key] = features
        return outputs



class LambdaLayer(nn.Module):

    def __init__(self, fn: Callable) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, input):
        return self.fn(input)