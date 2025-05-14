from __future__ import annotations

from typing import Callable

import gymnasium
import torch
from torch import nn
from torch.nn.functional import softplus


class IndependentNormalDistributionHead(nn.Module):

    def __init__(self, min_std=0.001):
        super().__init__()
        self.min_std = min_std

    def forward(self, x):
        mean, log_std = torch.split(x, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        std = torch.clamp(std, min=self.min_std)
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=std),
            reinterpreted_batch_ndims=1
        )


class MaskedCategoricalDistributionHead(nn.Module):

    def forward(self, x, mask):
        mask = mask.to(torch.bool)
        logits = torch.where(mask, x, torch.tensor(-1e8, device=x.device))
        return torch.distributions.Independent(
            torch.distributions.Categorical(logits=logits),
            reinterpreted_batch_ndims=1
        )


class BetaDistributionHead(nn.Module):

    def forward(self, params):
        params = 1 + softplus(params)
        alpha, beta = torch.split(params, 2, dim=-1)
        return torch.distributions.Independent(
            torch.distributions.Beta(alpha, beta),
            reinterpreted_batch_ndims=1
        )

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


class BirdViewEncoder(nn.Module):

    def __init__(self, input_channels: int):
        super().__init__()
        self.bv_encoder = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(input_channels, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
            nn.Flatten(),
        )

    def forward(self, x):
        return self.bv_encoder(x)


class ObservationModel(nn.Module):

    def __init__(self, models: dict[str, nn.Module], obs_space: gymnasium.spaces.Dict):
        super().__init__()
        self.models = nn.ModuleDict(models)

        dummy_obs = {
            k: torch.tensor(o).unsqueeze(0)
            for k, o in obs_space.sample().items()
        }
        dummy_embeds = self(dummy_obs)
        self.embedding_dim = dummy_embeds.shape[-1]

    def forward(self, obs_dict: dict):
        embeddings = {}
        for k in self.models.keys():
            assert k in obs_dict.keys(), f"key {k} not in obs"
            embeddings[k] = self.models[k](obs_dict[k])
        embeddings = [embeddings[k] for k in sorted(embeddings.keys())]
        return torch.cat(embeddings, dim=-1)
