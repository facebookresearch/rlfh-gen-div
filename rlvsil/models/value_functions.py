"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from torch import nn
import torch


class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPT2 that returns a scalar for each output token."""

    def __init__(self, hidden_size: int, normalisation: float, normalisation_std: float, config):
        super().__init__()
        self.detach_head = False
        self.linear = nn.Linear(hidden_size, 1)
        self.activation = nn.Identity()
        if config.value_head_activation:
            self.activation = nn.Tanh()

        self.flatten = nn.Flatten()
        self.normalisation = nn.Parameter(torch.tensor(float(normalisation)), requires_grad=False)
        self.normalisation_std = nn.Parameter(torch.tensor(float(normalisation_std)), requires_grad=False)

    def forward(self, hidden_states, cls_index=None):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states
        output = self.linear(output)
        output = self.activation(output)
        output = self.flatten(output)
        output = (output + self.normalisation) / self.normalisation_std

        return output
