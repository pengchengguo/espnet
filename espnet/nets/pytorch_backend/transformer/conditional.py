# Copyright 2019 Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ConditionalModule definition."""

import logging
import six
import torch

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device


class ConditionalModule(torch.nn.Module):
    """Conditional module for conditional chain model.

    Args:
        eprojs (int): number of projection units of encoder network
        odim (int): number of characters
        ctype (int): layer type of conditional module (LSTM or GRU)
        clayers (int): nubmer of layers of conditional module
        cunits (int): number of units of each conditional layer
        eos (int): end-of-sentence symbol
        use_ctc_alignment (bool): whether to use ctc alignment as condition or
                                  pre-softmax as condition
        sampling_probability (float):
        dropout (float):
        num_spkr (int):
    """

    def __init__(
        self,
        eprojs,
        odim,
        ctype,
        clayers,
        cunits,
        eos,
        use_ctc_alignment=True,
        dropout_rate=0.0,
        ignore_id=-1,
        num_spkrs=2,
    ):
        super(ConditionalModule, self).__init__()
        self.ctype = ctype
        self.cunits = cunits
        self.clayers = clayers
        self.use_ctc_alignment = use_ctc_alignment
        self.ignore_id = ignore_id
        self.eos = eos
        self.dropout_rate = dropout_rate
        self.num_spkrs = num_spkrs

        if self.use_ctc_alignment:
            logging.warning("Using CTC alignments as conditions.")
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(odim, cunits),
                torch.nn.Dropout(dropout_rate),
            )
        else:
            logging.warning("Using latent representations as conditions.")
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(eprojs, cunits),
                torch.nn.Dropout(dropout_rate),
            )

        self.layers = torch.nn.ModuleList()
        self.layers += [
            torch.nn.LSTMCell(cunits + eprojs, cunits)
            if self.ctype == "lstm"
            else torch.nn.GRUCell(cunits + eprojs, cunits)
        ]
        self.dropout_layers = torch.nn.ModuleList()
        self.dropout_layers += [torch.nn.Dropout(dropout_rate)]

        for _ in six.moves.range(1, self.clayers):
            self.layers += [
                torch.nn.LSTMCell(cunits, cunits)
                if self.ctype == "lstm"
                else torch.nn.GRUCell(cunits, cunits)
            ]
            self.dropout_layers += [torch.nn.Dropout(dropout_rate)]

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.cunits)

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        if self.ctype == "lstm":
            z_list[0], c_list[0] = self.layers[0](ey, (z_prev[0], c_prev[0]))
            for l in six.moves.range(1, self.clayers):
                z_list[l], c_list[l] = self.layers[l](
                    self.dropout_layers[l - 1](z_list[l - 1]), (z_prev[l], c_prev[l])
                )
        else:
            z_list[0] = self.layers[0](ey, z_prev[0])
            for l in six.moves.range(1, self.clayers):
                z_list[l] = self.layers[l](
                    self.dropout_layers[l - 1](z_list[l - 1]), z_prev[l]
                )
        return z_list, c_list

    def forward(self, xs_pad, ys_pad, ilens, prev_states=None, pad_compensation=True):
        """Definition of the forward operation.

        Args:
            xs_pad ([type]): [description]
            ys_pad ([type]): [description]
            ilens ([type]): [description]
            prev_states ([type], optional): [description]. Defaults to None.
            pad_compensation (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        if self.use_ctc_alignment:
            # ys_pad means ctc alignements, (B, Tmax)
            ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
            if pad_compensation:
                ys_in_pad = pad_list2(ys, ilens, self.eos)
            else:
                ys_in_pad = pad_list(ys, self.eos)

            # pre-computation of embedding
            ys_pad_emb = self.embed(ys_in_pad)
        else:
            # ys_pad means pre-softmax layer latent representations
            ys_pad_emb = self.embed(ys_pad)

        xs_pad = torch.cat((xs_pad, ys_pad_emb), dim=2)  # (B, Tmax, dim)
        n_batch, tmax, dunits = xs_pad.size()
        xs_pad = xs_pad.view(n_batch * tmax, dunits)

        if prev_states is None:
            z_list = [self.zero_state(xs_pad)]
            c_list = [self.zero_state(xs_pad)]
            for _ in six.moves.range(1, self.clayers):
                z_list.append(self.zero_state(xs_pad))
                c_list.append(self.zero_state(xs_pad))
        else:
            z_list, c_list = prev_states

        z_list, c_list = self.rnn_forward(xs_pad, z_list, c_list, z_list, c_list)

        xs_pad = self.dropout_layers[-1](z_list[-1])  # utt x (zdim)

        xs_pad = xs_pad.view(n_batch, tmax, xs_pad.size(1))
        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens).unsqueeze(-1))

        return xs_pad.masked_fill(mask, 0.0), (z_list, c_list)


def pad_list2(xs, ilens, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        ilens: torch.Tensor ilens batch of lengths of input sequences (B)
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    max_ilens = torch.max(ilens)
    pad = (
        xs[0].new(n_batch, max(max_len, max_ilens), *xs[0].size()[1:]).fill_(pad_value)
    )

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]
        if ilens[i] > xs[i].size(0):
            pad[i, : xs[i].size(0)] = xs[i][-1]

    return pad