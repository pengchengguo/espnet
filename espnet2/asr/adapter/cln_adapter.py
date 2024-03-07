#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  2024, Northwestern Polytechnical University, Pengcheng Guo
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from espnet2.asr.adapter.film_adapter import FiLM


class ConditionalLayerNorm(nn.Module):
    """Conditional layer normalization layer.

    URL: https://openreview.net/pdf?id=de11dbHzAMF
         https://github.com/HuangZiliAndy/fairseq/tree/multispk
    """

    def __init__(
        self,
        enroll_size: int,
        normalized_shape: int,
        eps: float = 1e-5,
        modulate_bias: bool = False,
        init_weight: torch.Tensor = None,
        init_bias: torch.Tensor = None,
    ):
        super().__init__()

        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        self.init_weight = init_weight
        self.init_bias = init_bias

        self.weight = Parameter(torch.empty(self.normalized_shape))
        self.bias = Parameter(torch.empty(self.normalized_shape))

        self.ln_weight_modulation = FiLM(enroll_size, self.normalized_shape[0])
        if modulate_bias:
            self.ln_bias_modulation = FiLM(enroll_size, self.normalized_shape[0])
        else:
            self.ln_bias_modulation = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weight is not None:
            self.weight.data.copy_(self.init_weight)
        else:
            torch.nn.init.ones_(self.weight)

        if self.init_bias is not None:
            self.bias.data.copy_(self.init_bias)
        else:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, enroll: torch.Tensor):
        # mean.shape: (B, T, 1), var.shape: (B, T, 1)
        mean = torch.mean(x, -1, keepdim=True)
        var = torch.var(x, -1, unbiased=False, keepdim=True)

        # weight.shape: (B, D)
        weight = self.ln_weight_modulation(
            self.weight.expand(enroll.size(0), -1), enroll
        )
        # weight.shape: (B, T, D)
        weight = weight.unsqueeze(1).expand(-1, x.size(1), -1)

        if self.ln_bias_modulation is None:
            bias = self.bias
        else:
            # bias.shape: (B, D)
            bias = self.ln_bias_modulation(self.bias.expand(enroll.size(0), -1), enroll)
            # bias.shape: (B, T, D)
            bias = bias.unsqueeze(1).expand(-1, x.size(1), -1)

        result = (x - mean) / torch.sqrt(var + self.eps) * weight + bias

        return result

