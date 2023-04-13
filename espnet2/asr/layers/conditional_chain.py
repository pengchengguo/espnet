# Copyright 2022 Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Definition of Conditional Chain Module"""

from typing import List
from typeguard import check_argument_types

import torch
import torch.nn as nn

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device


class ConditionalChain(nn.Module):
    """Conditional Chain Module.
    Args:
        input_size (int): Input dimension.
        hidden_size (int): Hidden dimension of RNN.
        num_layers (int): Number of RNN layers.
        rnn_type (str): Type of RNN cell. Only support lstm and gru.
        droput_rate (float): Dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 3,
        rnn_type: str = "lstm",
        dropout_rate: float = 0.1,
    ):
        assert check_argument_types()
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self._output_size = hidden_size

        if rnn_type not in ["lstm", "gru"]:
            raise ValueError(f"Not supported rnn_type={rnn_type}")

        # TODO (GPC): support using CTC alignment as conditions
        self.embed = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Dropout(dropout_rate),
        )

        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.layers += [
            nn.LSTMCell(2 * input_size, hidden_size)
            if rnn_type == "lstm"
            else nn.GRUCell(2 * input_size, hidden_size)
        ]
        self.dropout_layers += [nn.Dropout(dropout_rate)]

        for _ in range(1, num_layers):
            self.layers += [
                nn.LSTMCell(hidden_size, hidden_size)
                if rnn_type == "lstm"
                else nn.GRUCell(hidden_size, hidden_size)
            ]
            self.dropout_layers += [nn.Dropout(dropout_rate)]

    def zero_state(self, hs_pad: torch.Tensor):
        """Initialize the hidden state or cell state.

        Args:
            hs_pad (torch.Tensor): encoder output, (B * T, D_enc)
        """
        return hs_pad.new_zeros(hs_pad.size(0), self.hidden_size)

    def rnn_forward(
        self,
        hs_pad: torch.Tensor,
        h_list: List[torch.Tensor],
        c_list: List[torch.Tensor],
        h_prev: List[torch.Tensor],
        c_prev: List[torch.Tensor],
    ):
        """RNN forward operation.

        Args:
            hs_pad (torch.Tensor): Encoder output.
            h_list (torch.Tensor): Store the output hidden states of LSTM layers
            c_list (torch.Tensor): Store the output cell states of LSTM layers
            h_prev (torch.Tensor): Preivous hidden states of LSTM layers
            c_prev (torch.Tensor): Previous cell states of LSTM layers
        """
        if self.rnn_type == "lstm":
            # LSTM network
            h_list[0], c_list[0] = self.layers[0](hs_pad, (h_prev[0], c_prev[0]))
            for i in range(1, self.num_layers):
                h_list[i], c_list[i] = self.layers[i](
                    self.dropout_layers[i - 1](h_list[i - 1]), (h_prev[i], c_prev[i])
                )
        else:
            # GRU network
            h_list[0] = self.layers[0](hs_pad, h_prev[0])
            for i in range(1, self.num_layers):
                h_list[i] = self.decoder[i](
                    self.dropout_layers[i - 1](h_list[i - 1]), h_prev[i]
                )

        return h_list, c_list

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        enc_out: torch.Tensor,
        enc_out_lens: torch.Tensor,
        condition: torch.Tensor,
        prev_states: torch.Tensor = None,
    ):
        """Definition of the forward operation.

        Args:
            enc_out (torch.Tensor): Output of Mixture Encoder, (B, T, D_enc)
            enc_out_lens (torch.Tensor): Length of Mixture Encoder output, (B, T)
            condition (torch.Tensor): Conditions for current step, (B, T, D_enc)
        """
        # (B, T, D_enc)
        # here, we use the latent representation of pre-softmax layer as condition
        condition_emb = self.embed(condition)

        # (B, T, 2 * D_enc)
        enc_out = torch.cat((enc_out, condition_emb), dim=-1)
        nbatch, tmax, dunits = enc_out.size()
        enc_out = enc_out.view(nbatch * tmax, dunits)

        # Init hidden state and cell state for RNN network
        if prev_states is None:
            h_list = [self.zero_state(enc_out)]
            c_list = [self.zero_state(enc_out)]
            for _ in range(1, self.num_layers):
                h_list.append(self.zero_state(enc_out))
                c_list.append(self.zero_state(enc_out))
        else:
            h_list, c_list = prev_states

        h_list, c_list = self.rnn_forward(enc_out, h_list, c_list, h_list, c_list)

        condchain_out = self.dropout_layers[-1](h_list[-1])
        condchain_out = condchain_out.view(nbatch, tmax, -1)
        # mask to remove bias value of padded part
        mask = to_device(self, make_pad_mask(enc_out_lens).unsqueeze(-1))

        return condchain_out.masked_fill(mask, 0.0), (h_list, c_list)
