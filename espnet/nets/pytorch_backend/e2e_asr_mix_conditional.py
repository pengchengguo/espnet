#!/usr/bin/env python3

"""
This script is used to construct End-to-End models of multi-speaker ASR.

Copyright 2017 Johns Hopkins University (Shinji Watanabe)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import argparse
import logging
import math
import os
import sys

import chainer
from chainer import reporter
import editdistance
import numpy as np
import six
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.e2e_asr import E2E as E2E_ASR
from espnet.nets.pytorch_backend.frontends.feature_transform import (
    feature_transform_for,  # noqa: H301
)
from espnet.nets.pytorch_backend.frontends.frontend import frontend_for

from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for as encoder_for_single
from espnet.nets.pytorch_backend.rnn.encoders import Encoder as EncoderSingle
from espnet.nets.pytorch_backend.rnn.encoders import RNNP
from espnet.nets.pytorch_backend.rnn.encoders import VGG2L

CTC_LOSS_THRESHOLD = 10000


class PIT(object):
    """Permutation Invariant Training (PIT) module.

    :parameter int num_spkrs: number of speakers for PIT process (2 or 3)
    """

    def __init__(self, num_spkrs):
        """Initialize PIT module."""
        self.num_spkrs = num_spkrs
        perms = []
        self.permutationDFS(np.linspace(0, num_spkrs-1, num_spkrs, dtype=np.int64), 0, perms)
        # [[0, 1], [1, 0]] or [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]]
        self.perm_choices = perms   # torch.tensor(self.generate_permutation_schemes())
        # [[0, 3], [1, 2]] or [[0, 4, 8], [0, 5, 7], [1, 3, 8], [1, 5, 6], [2, 4, 6], [2, 3, 7]]
        self.loss_perm_idx = torch.linspace(0, num_spkrs*(num_spkrs-1), num_spkrs).long().unsqueeze(0)
        self.loss_perm_idx = (self.loss_perm_idx + torch.tensor(self.perm_choices)).tolist()

    def min_pit_sample(self, loss):
        """Compute the PIT loss for each sample.

        :param 1-D torch.Tensor loss: list of losses for one sample,
            including [h1r1, h1r2, h2r1, h2r2] or
            [h1r1, h1r2, h1r3, h2r1, h2r2, h2r3, h3r1, h3r2, h3r3]
        :return minimum loss of best permutation
        :rtype torch.Tensor (1)
        :return the best permutation
        :rtype List: len=2

        """
        score_perms = torch.stack([torch.sum(loss[loss_perm_idx]) 
                                   for loss_perm_idx in self.loss_perm_idx]
                                 ) / self.num_spkrs
        perm_loss, min_idx = torch.min(score_perms, 0)
        permutation = self.perm_choices[min_idx]

        return perm_loss, permutation

    def pit_process(self, losses):
        """Compute the PIT loss for a batch.

        :param torch.Tensor losses: losses (B, 1|4|9)
        :return minimum losses of a batch with best permutation
        :rtype torch.Tensor (B)
        :return the best permutation
        :rtype torch.LongTensor (B, 1|2|3)

        """
        bs = losses.size(0)
        ret = [self.min_pit_sample(losses[i]) for i in range(bs)]

        loss_perm = torch.stack([r[0] for r in ret], dim=0).to(losses.device)  # (B)
        permutation = torch.tensor([r[1] for r in ret]).long().to(losses.device)

        return torch.mean(loss_perm), permutation
    
    def permutationDFS(self, source, start, res):
        # get permutations with DFS
        # given the source as [1, 2, ..., N]
        # return order in [[1, 2], [2, 1]] or
        # [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 2, 1], [3, 1, 2]]
        if start == len(source) - 1:  # reach final state
            res.append(source.tolist())
        for i in range(start, len(source)):
            # swap values at position start and i
            source[start], source[i] = source[i], source[start]
            self.permutationDFS(source, start + 1, res)
            # reverse the swap
            source[start], source[i] = source[i], source[start]


class E2E(E2E_ASR, ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2E.encoder_add_arguments(parser)
        E2E.encoder_mix_add_arguments(parser)
        E2E.attention_add_arguments(parser)
        E2E.decoder_add_arguments(parser)
        return parser

    @staticmethod
    def encoder_mix_add_arguments(parser):
        """Add arguments for multi-speaker encoder."""
        group = parser.add_argument_group("E2E encoder setting for multi-speaker")
        # asr-mix encoder
        group.add_argument(
            "--elayers-sd",
            default=4,
            type=int,
            help="Number of speaker differentiate encoder layers"
            "for multi-speaker speech recognition task.",
        )
        return parser

    def __init__(self, idim, odim, args):
        """Initialize multi-speaker E2E module."""
        torch.nn.Module.__init__(self)
        self.mtlalpha = args.mtlalpha
        assert 0.0 <= self.mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.reporter = Reporter()
        self.num_spkrs = args.num_spkrs
        self.pit = PIT(self.num_spkrs)

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        self.subsample = get_subsample(args, mode="asr", arch="rnn_mix")

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(
                odim, args.lsm_type, transcript=args.train_json
            )
        else:
            labeldist = None

        if getattr(args, "use_frontend", False):  # use getattr to keep compatibility
            self.frontend = frontend_for(args, idim)
            self.feature_transform = feature_transform_for(args, (idim - 1) * 2)
            idim = args.n_mels
        else:
            self.frontend = None

        # encoder
        self.enc = encoder_for(args, idim, self.subsample)
        self.enc_common = DecoderConditional(
            args.eunits*2, odim, args.etype, args.elayers, args.eunits, args.eprojs, 
            args.eprojs, self.eos, args.dropout_rate, args.num_spkrs
        )
        # ctc
        self.ctc = ctc_for(args, odim, reduce=False)
        # attention
        num_att = 1
        self.att = att_for(args, num_att)
        # decoder
        self.dec = decoder_for(args, odim, self.sos, self.eos, self.att, labeldist)

        # weight initialization
        self.init_like_chainer()

        # options for beam search
        if "report_cer" in vars(args) and (args.report_cer or args.report_wer):
            recog_args = {
                "beam_size": args.beam_size,
                "penalty": args.penalty,
                "ctc_weight": args.ctc_weight,
                "maxlenratio": args.maxlenratio,
                "minlenratio": args.minlenratio,
                "lm_weight": args.lm_weight,
                "rnnlm": args.rnnlm,
                "nbest": args.nbest,
                "space": args.sym_space,
                "blank": args.sym_blank,
            }

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False
        self.rnnlm = None

        self.logzero = -10000000000.0
        self.loss = None
        self.acc = None

    def init_like_chainer(self):
        """Initialize weight like chainer.

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """

        def lecun_normal_init_parameters(module):
            for p in module.parameters():
                data = p.data
                if data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1.0 / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 4:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1.0 / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.0)

        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[l].bias_ih)

    def min_ctc_loss_and_perm(self, hs_pad, hs_len, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor hs_len: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, num_spkrs', Lmax)
        :rtype: torch.Tensor
        :return: attention loss value
        """
        n_batch, n_left_spkrs, _ = ys_pad.size()
        loss_stack = torch.stack(
            [self.ctc(hs_pad, hs_len, ys_pad[:, i]) for i in range(n_left_spkrs)]
        ) # (N, B, 1)
        min_loss, min_idx = torch.min(loss_stack, 0)
        if n_left_spkrs > 1:
            for i in range(n_batch):
                tmp = ys_pad[i][0]
                ys_pad[i][0] = ys_pad[i][min_idx[i]]
                ys_pad[i][min_idx[i]] = tmp
        return min_loss

    def forward(self, xs_pad, ilens, ys_pad, ys_ctc_align_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad:
            batch of padded character id sequence tensor (B, num_spkrs, Lmax)
        :param torch.Tensor ys_ctc_align_pad:
            batch of padded forced alignment sequence tensor (B, num_spkrs, Tmax')
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. Encoder
        hs_pad, hlens, _ = self.enc(xs_pad, ilens)
        
        hs_pad_sd, hs_len_sd, loss_ctc = [None] * self.num_spkrs, [None] * self.num_spkrs, [None] * self.num_spkrs
        align_ctc_state = ys_ctc_align_pad[:, 0].new_zeros(ys_ctc_align_pad[:, 0].size())  # (B, Tmax)
        for i in range(self.num_spkrs):
            hs_pad_sd[i], hs_len_sd[i], _ = self.enc_common(hs_pad, align_ctc_state, hlens)
            loss_ctc[i] = self.min_ctc_loss_and_perm(hs_pad_sd[i], hs_len_sd[i], ys_pad[:, i:])
            align_ctc_state = ys_ctc_align_pad[:, i]

        self.hs_pad_sd = hs_pad_sd
        loss_ctc = torch.stack(loss_ctc, dim=0).mean()  # (num_spkrs, B)

        # 2. CTC loss
        if self.mtlalpha == 0:
            loss_ctc, min_perm = None, None
        else:
            ys_pad = ys_pad.transpose(0, 1)  # (num_spkrs, B, Lmax)
            logging.info("ctc loss:" + str(float(loss_ctc)))

        # 3. attention loss
        if self.mtlalpha == 1:
            loss_att = None
            acc = None
        else:
            if not isinstance(hs_pad, list):  # single-speaker input xs_pad
                loss_att, acc, _ = self.dec(hs_pad, hlens, ys_pad)
            else:
                for i in range(ys_pad.size(1)):  # B
                    ys_pad[:, i] = ys_pad[min_perm[i], i]
                rslt = [
                    self.dec(hs_pad[i], hlens[i], ys_pad[i], strm_idx=i)
                    for i in range(self.num_spkrs)
                ]
                loss_att = sum([r[0] for r in rslt]) / float(len(rslt))
                acc = sum([r[1] for r in rslt]) / float(len(rslt))
        self.acc = acc

        # 4. compute cer without beam search
        if self.mtlalpha == 0 or self.char_list is None:
            cer_ctc = None
        else:
            cers = []
            for ns in range(self.num_spkrs):
                y_hats = self.ctc.argmax(hs_pad[ns]).data
                for i, y in enumerate(y_hats):
                    y_hat = [x[0] for x in groupby(y)]
                    y_true = ys_pad[ns][i]

                    seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                    seq_true = [
                        self.char_list[int(idx)] for idx in y_true if int(idx) != -1
                    ]
                    seq_hat_text = "".join(seq_hat).replace(self.space, " ")
                    seq_hat_text = seq_hat_text.replace(self.blank, "")
                    seq_true_text = "".join(seq_true).replace(self.space, " ")

                    hyp_chars = seq_hat_text.replace(" ", "")
                    ref_chars = seq_true_text.replace(" ", "")
                    if len(ref_chars) > 0:
                        cers.append(
                            editdistance.eval(hyp_chars, ref_chars) / len(ref_chars)
                        )

            cer_ctc = sum(cers) / len(cers) if cers else None

        # 5. compute cer/wer
        if (
            self.training
            or not (self.report_cer or self.report_wer)
            or not isinstance(hs_pad, list)
        ):
            cer, wer = 0.0, 0.0
            # oracle_cer, oracle_wer = 0.0, 0.0
        else:
            if self.recog_args.ctc_weight > 0.0:
                lpz = [
                    self.ctc.log_softmax(hs_pad[i]).data for i in range(self.num_spkrs)
                ]
            else:
                lpz = None

            word_eds, char_eds, word_ref_lens, char_ref_lens = [], [], [], []
            nbest_hyps = [
                self.dec.recognize_beam_batch(
                    hs_pad[i],
                    torch.tensor(hlens[i]),
                    lpz[i],
                    self.recog_args,
                    self.char_list,
                    self.rnnlm,
                    strm_idx=i,
                )
                for i in range(self.num_spkrs)
            ]
            # remove <sos> and <eos>
            y_hats = [
                [nbest_hyp[0]["yseq"][1:-1] for nbest_hyp in nbest_hyps[i]]
                for i in range(self.num_spkrs)
            ]
            for i in range(len(y_hats[0])):
                hyp_words = []
                hyp_chars = []
                ref_words = []
                ref_chars = []
                for ns in range(self.num_spkrs):
                    y_hat = y_hats[ns][i]
                    y_true = ys_pad[ns][i]

                    seq_hat = [
                        self.char_list[int(idx)] for idx in y_hat if int(idx) != -1
                    ]
                    seq_true = [
                        self.char_list[int(idx)] for idx in y_true if int(idx) != -1
                    ]
                    seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, " ")
                    seq_hat_text = seq_hat_text.replace(self.recog_args.blank, "")
                    seq_true_text = "".join(seq_true).replace(
                        self.recog_args.space, " "
                    )

                    hyp_words.append(seq_hat_text.split())
                    ref_words.append(seq_true_text.split())
                    hyp_chars.append(seq_hat_text.replace(" ", ""))
                    ref_chars.append(seq_true_text.replace(" ", ""))

                tmp_word_ed = [
                    editdistance.eval(
                        hyp_words[ns // self.num_spkrs], ref_words[ns % self.num_spkrs]
                    )
                    for ns in range(self.num_spkrs ** 2)
                ]  # h1r1,h1r2,h2r1,h2r2
                tmp_char_ed = [
                    editdistance.eval(
                        hyp_chars[ns // self.num_spkrs], ref_chars[ns % self.num_spkrs]
                    )
                    for ns in range(self.num_spkrs ** 2)
                ]  # h1r1,h1r2,h2r1,h2r2

                word_eds.append(self.pit.min_pit_sample(torch.tensor(tmp_word_ed))[0])
                word_ref_lens.append(len(sum(ref_words, [])))
                char_eds.append(self.pit.min_pit_sample(torch.tensor(tmp_char_ed))[0])
                char_ref_lens.append(len("".join(ref_chars)))

            wer = (
                0.0
                if not self.report_wer
                else float(sum(word_eds)) / sum(word_ref_lens)
            )
            cer = (
                0.0
                if not self.report_cer
                else float(sum(char_eds)) / sum(char_ref_lens)
            )

        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_ctc_data, loss_att_data, self.acc, ctc_cer, cer, wer, loss_data)
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        """E2E beam search.

        :param ndarray x: input acoustic feature (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        ilens = [x.shape[0]]

        # subsample frame
        x = x[:: self.subsample[0], :]
        h = to_device(self, to_torch_tensor(x).float())
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        # 0. Frontend
        if self.frontend is not None:
            hs, hlens, mask = self.frontend(hs, ilens)
            hlens_n = [None] * self.num_spkrs
            for i in range(self.num_spkrs):
                hs[i], hlens_n[i] = self.feature_transform(hs[i], hlens)
            hlens = hlens_n
        else:
            hs, hlens = hs, ilens

        # 1. Encoder
        if not isinstance(hs, list):  # single-channel multi-speaker input x
            hs, hlens, _ = self.enc(hs, hlens)
        else:  # multi-channel multi-speaker input x
            for i in range(self.num_spkrs):
                hs[i], hlens[i], _ = self.enc(hs[i], hlens[i])

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = [self.ctc.log_softmax(i)[0] for i in hs]
        else:
            lpz = None

        # 2. decoder
        # decode the first utterance
        y = [
            self.dec.recognize_beam(
                hs[i][0], lpz[i], recog_args, char_list, rnnlm, strm_idx=i
            )
            for i in range(self.num_spkrs)
        ]

        if prev:
            self.train()
        return y

    def recognize_batch(self, xs, recog_args, char_list, rnnlm=None):
        """E2E beam search.

        :param ndarray xs: input acoustic feature (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[:: self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)

        # 0. Frontend
        if self.frontend is not None:
            hs_pad, hlens, mask = self.frontend(xs_pad, ilens)
            hlens_n = [None] * self.num_spkrs
            for i in range(self.num_spkrs):
                hs_pad[i], hlens_n[i] = self.feature_transform(hs_pad[i], hlens)
            hlens = hlens_n
        else:
            hs_pad, hlens = xs_pad, ilens

        # 1. Encoder
        if not isinstance(hs_pad, list):  # single-channel multi-speaker input x
            hs_pad, hlens, _ = self.enc(hs_pad, hlens)
        else:  # multi-channel multi-speaker input x
            for i in range(self.num_spkrs):
                hs_pad[i], hlens[i], _ = self.enc(hs_pad[i], hlens[i])

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = [self.ctc.log_softmax(hs_pad[i]) for i in range(self.num_spkrs)]
            normalize_score = False
        else:
            lpz = None
            normalize_score = True

        # 2. decoder
        y = [
            self.dec.recognize_beam_batch(
                hs_pad[i],
                hlens[i],
                lpz[i],
                recog_args,
                char_list,
                rnnlm,
                normalize_score=normalize_score,
                strm_idx=i,
            )
            for i in range(self.num_spkrs)
        ]

        if prev:
            self.train()
        return y

    def enhance(self, xs):
        """Forward only the frontend stage.

        :param ndarray xs: input acoustic feature (T, C, F)
        """
        if self.frontend is None:
            raise RuntimeError("Frontend doesn't exist")
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[:: self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)
        enhanced, hlensm, mask = self.frontend(xs_pad, ilens)
        if prev:
            self.train()

        if isinstance(enhanced, (tuple, list)):
            enhanced = list(enhanced)
            mask = list(mask)
            for idx in range(len(enhanced)):  # number of speakers
                enhanced[idx] = enhanced[idx].cpu().numpy()
                mask[idx] = mask[idx].cpu().numpy()
            return enhanced, mask, ilens
        return enhanced.cpu().numpy(), mask.cpu().numpy(), ilens

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad:
            batch of padded character id sequence tensor (B, num_spkrs, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            # 0. Frontend
            if self.frontend is not None:
                hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
                hlens_n = [None] * self.num_spkrs
                for i in range(self.num_spkrs):
                    hs_pad[i], hlens_n[i] = self.feature_transform(hs_pad[i], hlens)
                hlens = hlens_n
            else:
                hs_pad, hlens = xs_pad, ilens

            # 1. Encoder
            if not isinstance(hs_pad, list):  # single-channel multi-speaker input x
                hs_pad, hlens, _ = self.enc(hs_pad, hlens)
            else:  # multi-channel multi-speaker input x
                for i in range(self.num_spkrs):
                    hs_pad[i], hlens[i], _ = self.enc(hs_pad[i], hlens[i])

            # Permutation
            ys_pad = ys_pad.transpose(0, 1)  # (num_spkrs, B, Lmax)
            if self.num_spkrs <= 3:
                loss_ctc = torch.stack(
                    [
                        self.ctc(
                            hs_pad[i // self.num_spkrs],
                            hlens[i // self.num_spkrs],
                            ys_pad[i % self.num_spkrs],
                        )
                        for i in range(self.num_spkrs ** 2)
                    ],
                    1,
                )  # (B, num_spkrs^2)
                loss_ctc, min_perm = self.pit.pit_process(loss_ctc)
            for i in range(ys_pad.size(1)):  # B
                ys_pad[:, i] = ys_pad[min_perm[i], i]

            # 2. Decoder
            att_ws = [
                self.dec.calculate_all_attentions(
                    hs_pad[i], hlens[i], ys_pad[i], strm_idx=i
                )
                for i in range(self.num_spkrs)
            ]

        return att_ws


class EncoderSpeech(torch.nn.Module):
    """Encoder module for the case of multi-speaker mixture speech.

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers_sd:
        number of layers of speaker differentiate part in encoder network
    :param int elayers_rec:
        number of layers of shared recognition part in encoder network
    :param int eunits: number of lstm units of encoder network
    :param int eprojs: number of projection units of encoder network
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
    :param int num_spkrs: number of number of speakers
    """

    def __init__(
        self,
        etype,
        idim,
        elayers,
        eunits,
        eprojs,
        subsample,
        dropout,
        in_channel=1,
    ):
        """Initialize the encoder of single-channel multi-speaker ASR."""
        super(EncoderSpeech, self).__init__()
        typ = etype.lstrip("vgg").rstrip("p")
        if typ not in ["lstm", "gru", "blstm", "bgru"]:
            logging.error("Error: need to specify an appropriate encoder architecture")
        if etype.startswith("vgg"):
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList(
                    [
                        VGG2L(in_channel),
                        RNNP(
                            get_vgg2l_odim(idim, in_channel=in_channel),
                            elayers,
                            eunits,
                            eprojs,
                            subsample,
                            dropout,
                            typ=typ,
                        )
                    ]
                )
                logging.info("Use CNN-VGG + B" + typ.upper() + "P for encoder")
            else:
                logging.error(
                    f"Error: need to specify an appropriate encoder architecture. "
                    f"Illegal name {etype}"
                )
                sys.exit()
        else:
            logging.error(
                f"Error: need to specify an appropriate encoder architecture. "
                f"Illegal name {etype}"
            )
            sys.exit()

    def forward(self, xs_pad, ilens):
        """Encodermix forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: list: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        for module in self.enc:
            xs_pad, ilens, _ = module(xs_pad, ilens)

        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens).unsqueeze(-1))

        return xs_pad.masked_fill(mask, 0.0), ilens, None


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
    pad = xs[0].new(n_batch, max(max_len, max_ilens), *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]
        if ilens[i] > xs[i].size(0):
            pad[i, : xs[i].size(0)] = xs[i][-1]

    return pad



from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
class DecoderConditional(torch.nn.Module):
    """Encoder module for the case of multi-speaker mixture speech.

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers_sd:
        number of layers of speaker differentiate part in encoder network
    :param int elayers_rec:
        number of layers of shared recognition part in encoder network
    :param int eunits: number of lstm units of encoder network
    :param int eprojs: number of projection units of encoder network
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
    :param int num_spkrs: number of number of speakers
    """

    def __init__(
        self,
        eprojs,
        odim,
        dtype,
        dlayers,
        dunits,
        eos,
        verbose=0,
        sampling_probability=0.0,
        dropout=0.0,
        num_spkrs=2,
    ):
        """Initialize the encoder of single-channel multi-speaker ASR."""
        super(DecoderConditional, self).__init__()
        self.dtype = dtype
        self.dunits = dunits
        self.dlayers = dlayers
        self.embed = torch.nn.Embedding(odim, dunits)
        self.dropout_emb = torch.nn.Dropout(p=dropout)

        self.ignore_id = -1

        self.eos = eos
        self.odim = odim
        self.verbose = verbose
        self.sampling_probability = sampling_probability
        self.dropout = dropout
        self.num_spkrs = num_spkrs

        self.decoder = torch.nn.ModuleList()
        self.dropout_dec = torch.nn.ModuleList()
        self.decoder += [
            torch.nn.LSTMCell(dunits + eprojs, dunits)
            if self.dtype == "lstm"
            else torch.nn.GRUCell(dunits + eprojs, dunits)
        ]
        self.dropout_dec += [torch.nn.Dropout(p=dropout)]
        for _ in six.moves.range(1, self.dlayers):
            self.decoder += [
                torch.nn.LSTMCell(dunits, dunits)
                if self.dtype == "lstm"
                else torch.nn.GRUCell(dunits, dunits)
            ]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        if self.dtype == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]), (z_prev[l], c_prev[l])
                )
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])
            for l in six.moves.range(1, self.dlayers):
                z_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]), z_prev[l]
                )
        return z_list, c_list

    def forward(self, xs_pad, ys_pad, ilens, prev_states=None, pad_compensation=True):
        """EncoderCommon forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ys_pad: batch of padded output sequences of previous speaker (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: list: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        if pad_compensation:
            ys_in_pad = pad_list2(ys, ilens, self.eos)
        else:
            ys_in_pad = pad_list(ys, self.eos)

        # pre-computation of embedding
        ys_pad_emb = self.dropout_emb(self.embed(ys_in_pad))

        xs_pad = torch.cat((xs_pad, ys_pad_emb), dim=2)  # (B, Tmax, dim)
        n_batch, tmax, dunits = xs_pad.size()
        xs_pad = xs_pad.view(n_batch * tmax, dunits)

        if prev_states is None:
            z_list = [self.zero_state(xs_pad)]
            c_list = [self.zero_state(xs_pad)]
            for _ in six.moves.range(1, self.dlayers):
                z_list.append(self.zero_state(xs_pad))
                c_list.append(self.zero_state(xs_pad))
        else:
            z_list, c_list = prev_states

        z_list, c_list = self.rnn_forward(xs_pad, z_list, c_list, z_list, c_list)

        xs_pad = self.dropout_dec[-1](z_list[-1])  # utt x (zdim)

        xs_pad = xs_pad.view(n_batch, tmax, xs_pad.size(1))
        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens).unsqueeze(-1))

        return xs_pad.masked_fill(mask, 0.0), (z_list, c_list)


def encoder_for(args, idim, subsample):
    """Construct the encoder."""
    return EncoderSingle(
        args.etype,
        idim,
        args.elayers_sd,
        args.eunits,
        args.eprojs,
        subsample[:args.elayers_sd+1],
        args.dropout_rate,
    )
