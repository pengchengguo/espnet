#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  2024, Northwestern Polytechnical University, Pengcheng Guo
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Feedforward post encoder."""

import torch
from typeguard import check_argument_types

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder


class FeedForwardPostEncoder(AbsPostEncoder):
    """Feedforward PostEncoder."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_rate: float = 0.1,
    ):
        assert check_argument_types()
        super().__init__()
        self.w_1 = torch.nn.Linear(input_size, hidden_size)
        self.w_2 = torch.nn.Linear(hidden_size, input_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = torch.nn.GELU()
        self.out_sz = input_size

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor):
        x = self.w_1(input)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x, input_lengths

    def output_size(self):
        """Get the output size."""
        return self.out_sz
