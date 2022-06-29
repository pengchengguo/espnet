# Copyright 2020 Pengcheng Guo
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""" Various Featurizer Class"""

from typeguard import check_argument_types

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class WeightedSumFeaturizer(torch.nn.Module):
    """Weighted summation featurizer (transparent attention) definition.
    
    Combining the latent features of encoder by a learnalbe weights.
    
    Paper: https://arxiv.org/abs/1808.07561.
    """

    def __init__(
        self,
        feat_dim: int,
        num_feat_in: int,
        num_feat_out: int,
        dropout_rate: float = 0.0,
        dropconnect_rate: float = 0.0,
    ):
        assert check_argument_types()
        super().__init__()

        self.num_feat_in = num_feat_in
        self.num_feat_out = num_feat_out

        # init the fusion weight
        self.weights = torch.nn.Parameter(
            torch.Tensor(self.num_feat_out, self.num_feat_in)
        )
        scale = (2.0 / (self.num_feat_out + self.num_feat_in)) ** 0.5
        torch.nn.init.uniform_(self.weights, -scale, scale)

        self.dropout = dropout_rate
        self.dropconnect = dropconnect_rate
        self.norm = LayerNorm(feat_dim)

    def forward(self, encoder_out_lst):
        assert isinstance(encoder_out_lst, list), type(encoder_out_lst)
        assert len(encoder_out_lst) == self.weights.shape[1], (
            len(encoder_out_lst),
            self.weights.shape[1],
        )

        encoder_out_fusion = []
        for widx in range(self.weights.shape[0]):
            cur_weight = F.softmax(self.weights[widx], dim=0)
            cur_weight = F.dropout(cur_weight, self.dropout, training=self.training)
            cur_encoder_out = [
                cur_weight[hidx] * encoder_out_lst[hidx]
                for hidx in range(len(encoder_out_lst))
            ]
            cur_encoder_out = torch.stack(cur_encoder_out).sum(0)
            encoder_out_fusion.append(self.norm(cur_encoder_out))

        return encoder_out_fusion


class IdentityFeaturizer(torch.nn.Module):
    """Do anything to the input."""
    def __init__():
        super().__init__()
        
    def forward(self, encoder_out_lst):
        assert isinstance(encoder_out_lst, list), type(encoder_out_lst)
        
        return encoder_out_lst
