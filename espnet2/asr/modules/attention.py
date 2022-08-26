#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Pengcheng Guo
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math
from random import sample

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class WeightedSumMultiHeadedAttention(MultiHeadedAttention):
    """Multi-headed attention with a learnable weights to combine multi
    encoder sequences.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        n_enc_seq (int): The number of encoder sequences
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, n_enc_seq, dropout_rate):
        super().__init__(n_head, n_feat, dropout_rate)
        self.n_enc_seq = n_enc_seq

        # init the fusion weight
        self.fusion_weight = nn.Parameter(torch.Tensor(self.n_enc_seq, 1))
        nn.init.xavier_uniform_(self.fusion_weight)

        # self.norm = LayerNorm(n_feat)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (List[torch.Tensor]): Key tensors from multiple inter layers of
                encoder, List[(#batch, time2, size)].
            value (List[torch.Tensor]): Value tensors from multiple inter layers of
                encoder, List[(#batch, time2, size)].
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        # fusion encoder sequences
        assert isinstance(key, list) and isinstance(value, list)
        assert self.n_enc_seq == len(key) == len(value)
        cur_weight = F.softmax(self.fusion_weight, dim=0)
        # cur_weight = self.dropout(cur_weight)
        key_combine = torch.stack(
            [cur_weight[idx] * key[idx] for idx in range(self.n_enc_seq)]
        ).sum(0)
        # key_combine = self.norm(key_combine)
        value_combine = torch.stack(
            [cur_weight[idx] * value[idx] for idx in range(self.n_enc_seq)]
        ).sum(0)
        # value_combine = self.norm(value_combine)

        q, k, v = self.forward_qkv(query, key_combine, value_combine)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)


class GumbelSoftmaxMultiHeadedAttention(nn.Module):
    """Multi-headed attention with Gumbel-Softmax to select attention heads

    Automatically sampling H heads from H' head candidates for different
    query-key vector pairs. Most used in the encoder-decoder attention. The
    query vector is text representation, while the key/value vectors are
    encoded speech representations from different intermediate layeres of encoder.

    Here, H' head candidates will be seperated into H groups and each group consists
    H' / H condidate heads. For each intermediate speech representation, we will
    select one head from each group, resulting H heads totally.

    Paper: https://arxiv.org/abs/2106.10840
    """

    def __init__(
        self, n_head, n_head_cand, n_feat, n_inters, temp, combine_type, dropout_rate
    ):
        """Construct a GumbelSoftmaxMultiHeadedAttention object."""
        super(GumbelSoftmaxMultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        assert n_head_cand % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head  # dim for each head
        # for attention select module
        self.h_cand = n_head_cand
        self.n_group = n_head
        self.h_per_group = n_head_cand // self.n_group  # num of head in each group

        self.linear_q = nn.Linear(n_feat, self.d_k * n_head_cand)
        self.linear_k = nn.Linear(n_feat, self.d_k * n_head_cand)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.n_inters = n_inters
        self.h_posterior = nn.Parameter(
            torch.FloatTensor(self.n_inters, self.n_group, self.h_per_group)
        )
        nn.init.xavier_uniform_(self.h_posterior)

        assert len(temp) == 3
        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp

        self.combine_type = combine_type
        if self.combine_type == "cat":
            self.combine = nn.Linear(self.n_inters * n_feat, n_feat)
        elif self.combine_type == "gated":
            self.sigmoid = nn.Sigmoid()
            self.combine = nn.Linear(2 * n_feat, n_feat)
        elif self.combine_type == "gated_v2":
            self.sigmoid = nn.Sigmoid()
            self.combine = nn.Linear(2 * n_feat, n_feat)
        else:
            self.combine = None

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay**num_updates, self.min_temp
        )

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (List[torch.Tensor]): Key tensors List[(#batch, time2, size)].
            value (List[torch.Tensor]): Value tensors List[(#batch, time2, size)].

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h_cand, self.d_k)
        q = q.transpose(1, 2)  # (batch, head_cand, time1, d_k)
        k_lst = []
        for k in key:
            k_trans = self.linear_k(k).view(n_batch, -1, self.h_cand, self.d_k)
            k_lst.append(k_trans.transpose(1, 2))  # (batch, head_cand, time2, d_k)
        v_lst = []
        for v in value:
            v_trans = self.linear_v(v).view(n_batch, -1, self.n_group, self.d_k)
            v_lst.append(v_trans.transpose(1, 2))  # (batch, head, time2, d_k)

        return q, k_lst, v_lst

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.n_group * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward_combine(self, context_lst):
        """Combine attention context vectors from different enc-dec sequence pairs.

        Args:
            context_lst (List[torch.Tensor]): list of attention context vectors
        """
        if self.combine_type == "sum":
            context_comb = torch.stack(context_lst, dim=0)
            context = torch.sum(context_comb, dim=0) / len(context_lst)
        elif self.combine_type == "cat":
            context_comb = torch.cat(context_lst, dim=-1)
            context = self.combine(context_comb)
        elif self.combine_type == "gated":
            context = context_lst[0]
            for idx in range(1, len(context_lst)):
                context_comb = torch.cat([context, context_lst[idx]], dim=-1)
                scale = self.sigmoid(self.combine(context_comb))
                context = scale * context + (1 - scale) * context_lst[idx]
        elif self.combine_type == "gated_v2":
            context = context_lst[0]
            for idx in range(1, len(context_lst)):
                context_comb = torch.cat([context, context_lst[idx]], dim=-1)
                scale = self.sigmoid(self.combine(context_comb))
                context = context + scale * context_lst[idx]
        else:
            raise NotImplementedError

        return context

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (List[torch.Tensor]): Key tensors from multiple inter layers of
                encoder, List[(#batch, time2, size)].
            value (List[torch.Tensor]): Value tensors from multiple inter layers of
                encoder, List[(#batch, time2, size)].
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        assert isinstance(key, list) and isinstance(value, list)
        assert self.n_inters == len(key) == len(value)

        q, k_lst, v_lst = self.forward_qkv(query, key, value)

        # select heads by Gumbel-Softmax
        if self.training:
            # (num_inter, n_group, h_per_group)
            h_select = F.gumbel_softmax(self.h_posterior, tau=self.curr_temp, hard=True)
        else:
            _, k = self.h_posterior.max(-1, keepdim=True)
            # (num_inter, n_group, h_per_group)
            h_select = self.h_posterior.new_zeros(*self.h_posterior.shape).scatter_(
                -1, k, 1.0
            )

        context_lst = []
        for idx in range(self.n_inters):
            # (batch, head_cand, time1, time2)
            scores = torch.matmul(q, k_lst[idx].transpose(-2, -1)) / math.sqrt(self.d_k)
            n_batch, _, len_q, len_k = scores.shape
            scores = scores.view(n_batch, self.n_group, self.h_per_group, len_q, len_k)
            # (batch, time1, time2, n_group, h_per_group)
            scores = scores.permute(0, 3, 4, 1, 2)
            # (batch, time1, time2, n_group) -> (batch, n_group, time1, time2)
            scores_select = (scores * h_select[idx]).sum(-1).permute(0, 3, 1, 2)
            context = self.forward_attention(v_lst[idx], scores_select, mask)
            context_lst.append(context)

        return self.forward_combine(context_lst)


class GumbelSoftmaxMultiHeadedAttention_V2(nn.Module):
    """Multi-headed attention with Gumbel-Softmax to select attention heads

    Automatically sampling H heads from H' head candidates for different
    query-key vector pairs. Most used in the encoder-decoder attention. The
    query vector is text representation, while the key/value vectors are
    encoded speech representations from different intermediate layeres of encoder.

    Here, H' head candidates will be seperated into H groups and each group consists
    H' / H condidate heads. For each intermediate speech representation, we will
    select one head from each group, resulting H heads totally.

    Paper: https://arxiv.org/abs/2106.10840
    """

    def __init__(self, n_head, n_head_cand, n_feat, n_inters, temp, dropout_rate):
        """Construct a GumbelSoftmaxMultiHeadedAttention object."""
        super(GumbelSoftmaxMultiHeadedAttention_V2, self).__init__()
        assert n_feat % n_head == 0
        assert n_head_cand % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head  # dim for each head
        # for attention select module
        self.h_cand = n_head_cand
        self.n_group = n_head
        self.h_per_group = n_head_cand // self.n_group  # num of head in each group

        self.linear_q = nn.Linear(n_feat, n_head_cand * self.d_k)
        self.linear_k = nn.Linear(n_feat, n_head_cand * self.d_k)
        self.linear_v = nn.Linear(n_feat, n_head_cand * self.d_k)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.n_inters = n_inters
        self.h_posterior = nn.Parameter(
            torch.FloatTensor(self.n_inters, self.n_group, self.h_per_group)
        )
        nn.init.xavier_uniform_(self.h_posterior)
        self.linear_out = nn.Linear(self.n_inters * n_feat, n_feat)

        assert len(temp) == 3
        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay**num_updates, self.min_temp
        )

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (List[torch.Tensor]): Key tensors List[(#batch, time2, size)].
            value (List[torch.Tensor]): Value tensors List[(#batch, time2, size)].

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h_cand, self.d_k)
        q = q.transpose(1, 2)  # (batch, head_cand, time1, d_k)
        k_lst = []
        for k in key:
            k_trans = self.linear_k(k).view(n_batch, -1, self.h_cand, self.d_k)
            k_lst.append(k_trans.transpose(1, 2))  # (batch, head_cand, time2, d_k)
        v_lst = []
        for v in value:
            v_trans = self.linear_v(v).view(n_batch, -1, self.h_cand, self.d_k)
            v_lst.append(v_trans.transpose(1, 2))  # (batch, head_cand, time2, d_k)

        return q, k_lst, v_lst

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head_cand, time1, time2)
        else:
            self.attn = torch.softmax(
                scores, dim=-1
            )  # (batch, head_cand, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head_cand, time1, d_k)

        return x.transpose(1, 2).contiguous()  # (batch, time1, head_cand, d_k)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (List[torch.Tensor]): Key tensors from multiple inter layers of
                encoder, List[(#batch, time2, size)].
            value (List[torch.Tensor]): Value tensors from multiple inter layers of
                encoder, List[(#batch, time2, size)].
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        assert isinstance(key, list) and isinstance(value, list)
        assert self.n_inters == len(key) == len(value)

        q, k_lst, v_lst = self.forward_qkv(query, key, value)

        # select heads by Gumbel-Softmax
        if self.training:
            # (num_inter, n_group, h_per_group)
            h_mask = F.gumbel_softmax(self.h_posterior, tau=self.curr_temp, hard=True)
        else:
            _, k = self.h_posterior.max(-1, keepdim=True)
            # (num_inter, n_group, h_per_group)
            h_mask = self.h_posterior.new_zeros(*self.h_posterior.shape).scatter_(
                -1, k, 1.0
            )

        n_batch = q.shape[0]
        context_lst = []
        for idx in range(self.n_inters):
            # (batch, head_cand, time1, time2)
            scores = torch.matmul(q, k_lst[idx].transpose(-2, -1)) / math.sqrt(self.d_k)
            # (batch, time1, head_cand, d_k)
            context = self.forward_attention(v_lst[idx], scores, mask)
            # (batch, time1, n_group, h_per_group, d_k)
            context = context.view(
                n_batch, -1, self.n_group, self.h_per_group, self.d_k
            )
            # (batch, time1, n_group, d_k)
            context_select = (context * h_mask[idx].unsqueeze(-1)).sum(-2)
            # (batch, time1, d_model)
            context_select = context_select.view(n_batch, -1, self.n_group * self.d_k)
            context_lst.append(context_select)

        return self.linear_out(torch.cat(context_lst, dim=-1))


class GumbelSoftmaxMultiHeadedAttention_V3(nn.Module):
    """Multi-headed attention with Gumbel-Softmax to select attention heads

    Automatically sampling H heads from H' head candidates for different
    query-key vector pairs. Most used in the encoder-decoder attention. The
    query vector is text representation, while the key/value vectors are
    encoded speech representations from different intermediate layeres of encoder.

    Two selection strategies can be choosen.
    1. Group selection
    Here, H' head candidates will be seperated into H groups and each group consists
    H' / H condidate heads. For each intermediate speech representation, we will
    select one head from each group, resulting H heads totally.

    2. Subset selection

    Paper: https://arxiv.org/abs/2106.10840
    """

    def __init__(
        self,
        n_head,
        n_head_cand,
        n_feat,
        n_inters,
        temp,
        dropout_rate,
        select_type="group",
    ):
        """Construct a GumbelSoftmaxMultiHeadedAttention object."""
        super(GumbelSoftmaxMultiHeadedAttention_V3, self).__init__()
        assert n_feat % n_head == 0
        assert n_head_cand % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head  # dim for each head

        # for attention select module
        self.select_type = select_type
        self.h_cand = n_head_cand
        self.n_group = n_head
        self.n_inters = n_inters

        self.linear_q = nn.Linear(n_feat, n_head_cand * self.d_k)
        self.linear_k = nn.Linear(n_feat, n_head_cand * self.d_k)
        self.linear_v = nn.Linear(n_feat, n_head_cand * self.d_k)
        self.linear_out = nn.Linear(self.n_inters * n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        if self.select_type == "group":
            self.h_per_group = n_head_cand // self.n_group  # num of head in each group
            self.h_posterior = nn.Parameter(
                torch.FloatTensor(self.n_inters, self.n_group, self.h_per_group, 2)
            )
            self.h_prior = torch.Tensor([1 / self.h_per_group])
        else:
            raise NotImplementedError

        nn.init.xavier_uniform_(self.h_posterior)

        assert len(temp) == 3
        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay**num_updates, self.min_temp
        )

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (List[torch.Tensor]): Key tensors List[(#batch, time2, size)].
            value (List[torch.Tensor]): Value tensors List[(#batch, time2, size)].

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h_cand, self.d_k)
        q = q.transpose(1, 2)  # (batch, head_cand, time1, d_k)
        k_lst = []
        for k in key:
            k_trans = self.linear_k(k).view(n_batch, -1, self.h_cand, self.d_k)
            k_lst.append(k_trans.transpose(1, 2))  # (batch, head_cand, time2, d_k)
        v_lst = []
        for v in value:
            v_trans = self.linear_v(v).view(n_batch, -1, self.h_cand, self.d_k)
            v_lst.append(v_trans.transpose(1, 2))  # (batch, head_cand, time2, d_k)

        return q, k_lst, v_lst

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head_cand, time1, time2)
        else:
            self.attn = torch.softmax(
                scores, dim=-1
            )  # (batch, head_cand, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head_cand, time1, d_k)

        return x.transpose(1, 2).contiguous()  # (batch, time1, head_cand, d_k)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (List[torch.Tensor]): Key tensors from multiple inter layers of
                encoder, List[(#batch, time2, size)].
            value (List[torch.Tensor]): Value tensors from multiple inter layers of
                encoder, List[(#batch, time2, size)].
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        assert isinstance(key, list) and isinstance(value, list)
        assert self.n_inters == len(key) == len(value)

        q, k_lst, v_lst = self.forward_qkv(query, key, value)

        if self.select_type == "group":
            if self.training:
                # Obtain the select posterior for each head using Gumbel-Softmax
                # for self.h_posterior, dim0 means skip and dim1 means select
                # h_select.shape = (num_inter, n_group, h_per_group)
                self.h_select = F.gumbel_softmax(
                    self.h_posterior, tau=self.curr_temp, hard=False
                )[:, :, :, 1]
            else:
                # for inference, we use hard selection decisions
                select_posterior = self.h_posterior[:, :, :, 1]
                _, k = select_posterior.max(-1, keepdim=True)
                # (num_inter, n_group, h_per_group)
                self.h_select = select_posterior.new_zeros(
                    *select_posterior.shape
                ).scatter_(-1, k, 1.0)
        else:
            raise NotImplementedError

        n_batch = q.shape[0]
        context_lst = []
        for idx in range(self.n_inters):
            # (batch, head_cand, time1, time2)
            scores = torch.matmul(q, k_lst[idx].transpose(-2, -1)) / math.sqrt(self.d_k)
            # (batch, time1, head_cand, d_k)
            context = self.forward_attention(v_lst[idx], scores, mask)
            # (batch, time1, n_group, h_per_group, d_k)
            context = context.view(
                n_batch, -1, self.n_group, self.h_per_group, self.d_k
            )
            # (batch, time1, n_group, d_k)
            context_select = (context * self.h_select[idx].unsqueeze(-1)).sum(-2)
            # (batch, time1, d_model)
            context_select = context_select.view(n_batch, -1, self.n_group * self.d_k)
            context_lst.append(context_select)

        return self.linear_out(torch.cat(context_lst, dim=-1))


class SelectedMultiHeadedAttention(nn.Module):
    """Multi-headed attention with Gumbel-Softmax to select attention heads

    Automatically sampling H heads from H' head candidates for different
    query-key vector pairs. Most used in the encoder-decoder attention. The
    query vector is text representation, while the key/value vectors are
    encoded speech representations from different intermediate layeres of encoder.

    Two selection strategies can be choosen.
    1. Group selection
    Here, H' head candidates will be seperated into H groups and each group consists
    H' / H condidate heads. For each intermediate speech representation, we will
    select one head from each group, resulting H heads totally.

    2. Subset selection

    Paper: https://arxiv.org/abs/2106.10840
    """

    def __init__(
        self,
        n_head,
        n_head_cand,
        n_feat,
        n_inter,
        temp,
        dropout_rate,
        select_type="group",
    ):
        """Construct a GumbelSoftmaxMultiHeadedAttention object."""
        super(SelectedMultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        assert n_head_cand % n_head == 0  # just for group selection
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head  # dim for each head
        self.h = n_head

        self.linear_q = nn.Linear(n_feat, n_head_cand * self.d_k)
        self.linear_k = nn.Linear(n_feat, n_head_cand * self.d_k)
        self.linear_v = nn.Linear(n_feat, n_head_cand * self.d_k)
        self.linear_out = nn.Linear(n_inter * n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        # for attention head selection
        self.select_type = select_type
        self.h_cand = n_head_cand
        self.n_inter = n_inter

        self.h_logit = nn.Parameter(
            torch.Tensor(self.n_inter, self.h_cand), requires_grad=True
        )
        self.h_prior = float(self.h) / self.h_cand
        nn.init.uniform_(self.h_logit, a=math.log(0.01), b=math.log(1.0))

        assert len(temp) == 3
        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay**num_updates, self.min_temp
        )

    def gumbel_sample(self, logit, tau):
        """Sampling discrete variables by Gumbel-Sigmoid.

        Add two gumbel noises (from a norm gumbel distribution, F(x) = e^{-e^{-x}})
        to the logits and then conduct sigmoid fucntion for a 2-class classification.

        Paper: https://aclanthology.org/2020.acl-main.269.pdf

        Args:
            logits (torch.Tensor): logits to do sample
            tau (float): temperature parameter
        """
        noise1 = (
            -torch.empty_like(logit, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        noise2 = (
            -torch.empty_like(logit, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        samples = (logit + noise1 - noise2) / tau
        samples_soft = samples.sigmoid()

        return samples_soft

    def subset_select(self, h_samples, topk):
        """Select head by subset type.

        Select H heads from H' head candiates based on the head posteriors.

        Args:
            h_samples (torch.tensor): head posteriors sampled by Gumbel-Sigmoid
            topk (int): select the largest topk heads
        """
        top_vals, top_idxs = torch.topk(h_samples, k=topk, dim=-1)
        top_weights = 1.0 - top_vals.detach() + top_vals
        return top_idxs.detach(), top_weights

    def group_select(self, h_samples, topk):
        """Select head by gourp type.

        Here, H' head candidates will be seperated into H groups and each group consists
        H' / H condidate heads. For each intermediate speech representation, we will
        select one head from each group, resulting in H heads totally.

        Args:
            h_samples (torch.tensor): head posteriors sampled by Gumbel-Sigmoid
            topk (int): select the largest topk heads
        """
        device = h_samples.device
        # top_vals.shape == top_idxs.shape == (n_inter, n_head)
        top_vals, top_idxs = torch.max(h_samples.view(self.n_inter, -1, topk), dim=1)
        base_idxs = torch.arange(topk, device=device).unsqueeze(0)
        top_idxs = top_idxs * topk + base_idxs
        top_weights = 1.0 - top_vals.detach() + top_vals
        return top_idxs.detach(), top_weights

    def head_select(self, logit, n_select, temp):
        """Select heads for each encoder intermediate sequence."""
        # obtain the Gumbel-Sigmoid output for each head candidate, (n_iter, n_head_cand)
        self.h_samples = self.gumbel_sample(logit, tau=temp)

        if self.select_type == "subset":
            h_select_idxs, h_select_weights = self.subset_select(
                self.h_samples, topk=n_select
            )
        elif self.select_type == "group":
            # h_idx_select: (n_iter, n_head), detached from computation graph
            # h_weight_select: (n_iter, n_head)
            h_select_idxs, h_select_weights = self.group_select(
                self.h_samples, topk=n_select
            )
        else:
            raise ValueError("{} is not supported.".format(self.select_type))

        return h_select_idxs, h_select_weights

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (List[torch.Tensor]): Key tensors List[(#batch, time2, size)].
            value (List[torch.Tensor]): Value tensors List[(#batch, time2, size)].

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h_cand, self.d_k)
        q = q.transpose(1, 2)  # (batch, head_cand, time1, d_k)
        k_lst = []
        for k in key:
            k_trans = self.linear_k(k).view(n_batch, -1, self.h_cand, self.d_k)
            k_lst.append(k_trans.transpose(1, 2))  # (batch, head_cand, time2, d_k)
        v_lst = []
        for v in value:
            v_trans = self.linear_v(v).view(n_batch, -1, self.h_cand, self.d_k)
            v_lst.append(v_trans.transpose(1, 2))  # (batch, head_cand, time2, d_k)

        return q, k_lst, v_lst

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head_cand, time1, time2)
        else:
            self.attn = torch.softmax(
                scores, dim=-1
            )  # (batch, head_cand, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head_cand, time1, d_k)

        return x.transpose(1, 2).contiguous()  # (batch, time1, head_cand, d_k)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (List[torch.Tensor]): Key tensors from multiple inter layers of
                encoder, List[(#batch, time2, size)].
            value (List[torch.Tensor]): Value tensors from multiple inter layers of
                encoder, List[(#batch, time2, size)].
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        assert isinstance(key, list) and isinstance(value, list)
        assert self.n_inter == len(key) == len(value)

        q, k_lst, v_lst = self.forward_qkv(query, key, value)

        # h_select_idx: (n_iter, n_head), h_select_weight: (n_iter, n_head)
        h_select_idxs, h_select_weights = self.head_select(
            logit=self.h_logit, n_select=self.h, temp=self.curr_temp
        )

        n_batch = q.shape[0]
        context_lst = []
        for idx in range(self.n_inter):
            # (batch, head_cand, time1, time2)
            scores = torch.matmul(q, k_lst[idx].transpose(-2, -1)) / math.sqrt(self.d_k)
            # (batch, time1, head_cand, d_k)
            mixed_context = self.forward_attention(v_lst[idx], scores, mask)
            # (batch, time1, head, d_k)
            context = mixed_context[:, :, h_select_idxs[idx], :]
            context = context * h_select_weights[idx].unsqueeze(-1)
            context = context.contiguous().view(n_batch, -1, self.h * self.d_k)
            context_lst.append(context)

            # select attention scores
            # (batch, head_cand, time1, time2) -> (batch, head, time1, time2)
            self.attn = self.attn[:, h_select_idxs[idx], :, :]

        return self.linear_out(torch.cat(context_lst, dim=-1))
