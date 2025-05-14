from __future__ import annotations
import gymnasium
import numpy as np
import torch
import torchvision
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.data import TensorSpec
from torchrl.modules import MLP, SafeModule, SafeSequential

from training.adex.models.common import DictForwardModule, LambdaLayer


class DictObservationEncoder(nn.Module):
    def __init__(
        self,
        obs_space: gymnasium.spaces.Dict,
        keys: list[str] = None,
        hidden_dims: int = 256,
        mlp_depth: int = 2,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if keys is None:
            keys = sorted(list(obs_space.spaces.keys()))
        self.keys = keys
        modules = {}

        for obs in keys:
            space = obs_space[obs]
            if obs == "birdview":
                modules[obs] = BirdViewObservationEncoder(
                    num_input_channels=space.shape[0],
                    embed_dim=embed_dim
                )
            else:
                assert isinstance(space, gymnasium.spaces.Box) or isinstance(
                    space, gymnasium.spaces.Discrete
                )
                modules[obs] = MLP(
                    in_features=np.prod(space.shape).astype(int)
                    if isinstance(space, gymnasium.spaces.Box)
                    else space.n,
                    out_features=embed_dim,
                    num_cells=hidden_dims,
                    depth=mlp_depth,
                    activation_class=nn.ELU
                )
        self.enocders = nn.ModuleDict(modules)
        self.merger = (
            MLP(
                in_features=len(modules) * embed_dim,
                depth=mlp_depth,
                num_cells=hidden_dims,
                out_features=embed_dim,
                activation_class=nn.ELU,
            )
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = []
        assert set(self.keys).issubset(obs.keys())
        for key in sorted(self.keys):
            embed = self.enocders[key](obs[key])
            embeddings.append(embed)
        embeddings = torch.cat(embeddings, dim=-1)
        return self.merger(embeddings)



class BirdViewObservationEncoder(nn.Module):
    def __init__(self, num_input_channels: int, embed_dim: int) -> None:
        super().__init__()
        resnet = torchvision.models.resnet18()
        resnet.conv1 = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=False,
        )
        self.resnet = resnet
        self.projection = nn.Linear(1000, embed_dim)

    def forward(self, birdview: torch.Tensor) -> torch.Tensor:
        shape = birdview.shape
        if birdview.dim() > 4:
            birdview = birdview.reshape(-1, *shape[-3:])
        features = self.resnet(birdview)
        features = self.projection(features)
        return features.view(*shape[:-3], -1)
