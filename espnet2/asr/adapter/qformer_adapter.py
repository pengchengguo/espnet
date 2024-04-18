#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  2024, Northwestern Polytechnical University, Pengcheng Guo
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Definition of the Q-Former module.

Reference:
    Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image 
                Encoders and Large Language Models"
    https://arxiv.org/abs/2301.12597
    https://github.com/salesforce/LAVIS/tree/main
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.asr.adapter.Qformer import BertConfig, BertLMHeadModel
from espnet2.torch_utils.model_summary import model_summary


class QFormerAdapter(nn.Module):
    def __init__(
        self,
        encoder_width: int,
        num_query_tokens: int = 1,
        num_hidden_layers: int = 2,
    ):
        super().__init__()

        config = BertConfig()
        config.num_hidden_layers = num_hidden_layers
        config.encoder_width = encoder_width
        config.add_cross_attention = True
        config.cross_attention_freq = 1
        config.query_length = num_query_tokens
        config.max_position_embeddings = 1500  # same as whisper encoder

        # init qformer module
        self.qformer = BertLMHeadModel(config=config)

        # init query_tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.query_length, config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=config.initializer_range)

        # logging.info(f"Qformer Summary: \n{model_summary(self)}")

    def output_size(self):
        """Get the output size."""
        return self.qformer.config.hidden_size

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        enroll_feats: torch.Tensor,
        enroll_feats_lens: torch.Tensor,
    ):
        query_tokens = self.query_tokens.expand(
            encoder_out.size(0), -1, -1
        ).contiguous()

        query_attn_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            encoder_out.device
        )
        enroll_attn_mask = ~make_pad_mask(enroll_feats_lens).to(encoder_out.device)
        attention_mask = torch.cat([query_attn_mask, enroll_attn_mask], dim=1)

        encoder_attention_mask = ~make_pad_mask(encoder_out_lens).to(encoder_out.device)

        qformer_output = self.qformer.bert(
            enroll_feats,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_out,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        query_length = query_tokens.size(1)
        query_embeddings = qformer_output.last_hidden_state[
            :, :query_length, :
        ].contiguous()
        enroll_embeddings = qformer_output.last_hidden_state[
            :, query_length:, :
        ].contiguous()

        return query_embeddings, enroll_embeddings
