#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  2024, Northwestern Polytechnical University, Pengcheng Guo
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) layer.

    URL: https://arxiv.org/pdf/1709.07871.pdf,
         https://github.com/HuangZiliAndy/fairseq/tree/multispk
    """

    def __init__(
        self,
        enroll_size: torch.Tensor,
        hidden_size: torch.Tensor,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        self.enroll_size = enroll_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        gamma_lst, beta_lst = [], []
        for i in range(num_layers):
            if i == 0:
                gamma_lst.append(nn.Linear(enroll_size, hidden_size))
                beta_lst.append(nn.Linear(enroll_size, hidden_size))
            else:
                gamma_lst.append(nn.Linear(hidden_size, hidden_size))
                beta_lst.append(nn.Linear(hidden_size, hidden_size))
        self.gamma_lst = nn.ModuleList(gamma_lst)
        self.beta_lst = nn.ModuleList(beta_lst)

        self.init_weights()

    def init_weights(self):
        for layer in self.gamma_lst + self.beta_lst:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, enroll: torch.Tensor):
        for i in range(self.num_layers):
            if i == 0:
                gamma = self.gamma_lst[i](enroll)
                beta = self.beta_lst[i](enroll)
            else:
                gamma = self.gamma_lst[i](gamma)
                beta = self.beta_lst[i](beta)

        x = (1 + gamma) * x + beta

        return x

