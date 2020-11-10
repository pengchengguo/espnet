#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from distutils.util import strtobool
import logging
import math
import random
import six

import torch

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator as ASRErrorCalculator
from espnet.nets.e2e_mt_common import ErrorCalculator as MTErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_st import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.st_interface import STInterface
from espnet.utils.fill_missing_args import fill_missing_args


class E2E(STInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument(
            "--transformer-init",
            type=str,
            default="pytorch",
            choices=[
                "pytorch",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
            ],
            help="how to initialize transformer parameters",
        )
        group.add_argument(
            "--transformer-input-layer",
            type=str,
            default="conv2d",
            choices=["conv2d", "linear", "embed"],
            help="transformer input layer type",
        )
        group.add_argument(
            "--transformer-attn-dropout-rate",
            default=None,
            type=float,
            help="dropout in transformer attention. use --dropout-rate if None is set",
        )
        group.add_argument(
            "--transformer-lr",
            default=10.0,
            type=float,
            help="Initial value of learning rate",
        )
        group.add_argument(
            "--transformer-warmup-steps",
            default=25000,
            type=int,
            help="optimizer warmup steps",
        )
        group.add_argument(
            "--transformer-length-normalized-loss",
            default=False,
            type=strtobool,
            help="normalize loss by length",
        )
        group.add_argument(
            "--transformer-encoder-selfattn-layer-type",
            type=str,
            default="selfattn",
            choices=[
                "selfattn",
                "lightconv",
                "lightconv2d",
                "dynamicconv",
                "dynamicconv2d",
                "light-dynamicconv2d",
            ],
            help="transformer encoder self-attention layer type",
        )
        group.add_argument(
            "--transformer-decoder-selfattn-layer-type",
            type=str,
            default="selfattn",
            choices=[
                "selfattn",
                "lightconv",
                "lightconv2d",
                "dynamicconv",
                "dynamicconv2d",
                "light-dynamicconv2d",
            ],
            help="transformer decoder self-attention layer type",
        )
        # Lightweight/Dynamic convolution related parameters.
        # See https://arxiv.org/abs/1912.11793v2
        # and https://arxiv.org/abs/1901.10430 for detail of the method.
        # Configurations used in the first paper are in
        # egs/{csj, librispeech}/asr1/conf/tuning/ld_conv/
        group.add_argument(
            "--wshare",
            default=4,
            type=int,
            help="Number of parameter shargin for lightweight convolution",
        )
        group.add_argument(
            "--ldconv-encoder-kernel-length",
            default="21_23_25_27_29_31_33_35_37_39_41_43",
            type=str,
            help="kernel size for lightweight/dynamic convolution: "
            'Encoder side. For example, "21_23_25" means kernel length 21 for '
            "First layer, 23 for Second layer and so on.",
        )
        group.add_argument(
            "--ldconv-decoder-kernel-length",
            default="11_13_15_17_19_21",
            type=str,
            help="kernel size for lightweight/dynamic convolution: "
            'Decoder side. For example, "21_23_25" means kernel length 21 for '
            "First layer, 23 for Second layer and so on.",
        )
        group.add_argument(
            "--ldconv-usebias",
            type=strtobool,
            default=False,
            help="use bias term in lightweight/dynamic convolution",
        )
        group.add_argument(
            "--dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for the encoder",
        )
        group.add_argument(
            "--sampling-probability",
            default=0.5,
            type=float,
            help="Ratio of predicted labels fed back to mt encoder",
        )
        # Encoder
        group.add_argument(
            "--elayers",
            default=4,
            type=int,
            help="Number of encoder layers",
        )
        group.add_argument(
            "--eunits",
            "-u",
            default=2048,
            type=int,
            help="Number of encoder hidden units",
        )
        # Attention
        group.add_argument(
            "--adim",
            default=256,
            type=int,
            help="Number of attention transformation dimensions",
        )
        group.add_argument(
            "--aheads",
            default=4,
            type=int,
            help="Number of heads for multi head attention",
        )
        # Decoder
        group.add_argument(
            "--dlayers", default=6, type=int, help="Number of decoder layers"
        )
        group.add_argument(
            "--dunits", default=2048, type=int, help="Number of decoder hidden units"
        )
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.encoder = Encoder(
            idim=idim,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_encoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
        )
        self.decoder = Decoder(
            odim=odim,
            selfattention_layer_type=args.transformer_decoder_selfattn_layer_type,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_decoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate,
        )
        self.pad = 0  # use <blank> for padding
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="st", arch="transformer")
        self.sampling_probability = args.sampling_probability
        self.reporter = Reporter()

        self.criterion = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )
        # submodule for ASR task
        self.mtlalpha = args.mtlalpha
        self.asr_weight = args.asr_weight

        # submodule for MT task
        self.mt_weight = args.mt_weight
        self.encoder_mt = Encoder(
            idim=odim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            input_layer="embed",
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            padding_idx=0,
        )
        self.reset_parameters(args)  # NOTE: place after the submodule initialization
        self.adim = args.adim  # used for CTC (equal to d_model)
        self.ctc = CTC(
            odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
        )
        self.fc = torch.nn.Linear(2 * self.adim, self.adim)

        # translation error calculator
        self.error_calculator = MTErrorCalculator(
            args.char_list, args.sym_space, args.sym_blank, args.report_bleu
        )

        # recognition error calculator
        self.error_calculator_asr = ASRErrorCalculator(
            args.char_list,
            args.sym_space,
            args.sym_blank,
            args.report_cer,
            args.report_wer,
        )
        self.rnnlm = None

        # multilingual E2E-ST related
        self.multilingual = getattr(args, "multilingual", False)
        self.replace_sos = getattr(args, "replace_sos", False)

    def reset_parameters(self, args):
        """Initialize parameters."""
        initialize(self, args.transformer_init)
        if self.mt_weight > 0:
            torch.nn.init.normal_(
                self.encoder_mt.embed[0].weight, mean=0, std=args.adim ** -0.5
            )
            torch.nn.init.constant_(self.encoder_mt.embed[0].weight[self.pad], 0)

    def forward(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :param torch.Tensor ys_pad_src: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 0. Extract target language ID
        tgt_lang_ids = None
        if self.multilingual:
            tgt_lang_ids = ys_pad[:, 0:1]
            ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining

        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)

        # 2. forward ctc
        loss_ctc, cer_ctc, ys_hat_ctc = self.forward_asr(hs_pad, hs_mask, ys_pad_src)

        # 3. forward mt encoder
        hs_mt_pad, _ = self.encoder_mt(ys_hat_ctc, hs_mask)

        # 4. fusion module
        hs_all_pad = torch.cat((hs_pad, hs_mt_pad), dim=2)  # (B, Tmax, dim*2)
        hs_all_pad = self.fc(hs_all_pad)

        # 5. forward decoder
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        # replace <sos> with target language ID
        if self.replace_sos:
            ys_in_pad = torch.cat([tgt_lang_ids, ys_in_pad[:, 1:]], dim=1)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

        # 6. compute ST loss
        loss_st = self.criterion(pred_pad, ys_out_pad)

        self.acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        # 8. compute corpus-level bleu in a mini-batch
        if self.training:
            self.bleu = None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            self.bleu = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        self.loss = loss_ctc + loss_st

        loss_asr_data = float(loss_ctc)
        loss_st_data = float(loss_st)
        loss_data = float(self.loss)

        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_asr_data,
                None,
                loss_st_data,
                None,
                None,
                self.acc,
                cer_ctc,
                None,
                None,
                self.bleu,
                loss_data,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    def forward_asr(self, hs_pad, hs_mask, ys_pad):
        """Forward pass in the ASR task.

        :param torch.Tensor hs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor hs_mask: batch of input token mask (B, Lmax)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        """
        # CTC
        batch_size = hs_pad.size(0)
        hs_len = hs_mask.view(batch_size, -1).sum(1)
        loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
        ys_hat_ctc = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
        cer_ctc = self.error_calculator_asr(ys_hat_ctc.cpu(), ys_pad.cpu(), is_ctc=True)
        # for visualization
        self.ctc.softmax(hs_pad)
        return loss_ctc, cer_ctc, ys_hat_ctc

    def forward_mt(self, xs_pad, ys_pad_src):
        """Forward pass in the auxiliary MT task.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        """
        # compute the real length of xs_pad
        ilens = torch.sum(ys_pad_src != self.ignore_id, dim=1).cpu().numpy()
        # NOTE: xs_pad is padded with -1
        xs_mask = ys_pad_src == self.ignore_id
        xs_zero_pad = xs_pad.masked_fill(xs_mask, 0.0)  # change pad value from -1 to 0
        xs_zero_pad = xs_zero_pad[:, : max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_zero_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder_mt(xs_zero_pad, src_mask)

        return hs_pad, hs_mask

    def forward_mt_encoder(self, xs_pad, ys_in_pad):
        """Forward pass in MT encoder.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ys_in_pad: batch of padded target sequences (B, Lmax)
        :param torch.Tensor ys_mask: batch of input token mask (B, Lmax)
        """
        ilens = torch.sum(xs_pad != self.ignore_id, dim=1).cpu().numpy()
        # NOTE: xs_pad is padded with -1
        xs = [x[x != self.ignore_id] for x in xs_pad]  # parse padded xs
        xs_zero_pad = pad_list(xs, self.pad)  # re-pad with zero
        xs_zero_pad = xs_zero_pad[:, : max(ilens)]  # for data parallel
        src_mask = (
            make_non_pad_mask(ilens.tolist()).to(xs_zero_pad.device).unsqueeze(-2)
        )
        hs_pad, hs_mask = self.encoder_mt(xs_zero_pad, src_mask)

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder)

    def encode(self, x):
        """Encode source acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)
        return enc_output.squeeze(0)

    def encode_mt(self, x):
        """Encode source linguistic features."""
        self.eval()
        x = torch.as_tensor(x).long().unsqueeze(0)
        enc_output, _ = self.encoder_mt(x, None)
        return enc_output.squeeze(0)

    def recognize_asr(self, h, recog_args, char_list=None, rnnlm=None, use_jit=False):
        logging.info("ASR: input lengths: " + str(h.size(1)))
        # search parms
        beam = recog_args.asr_beam_size
        penalty = recog_args.penalty

        # prepare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.size(1)
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(1)))
        minlen = int(recog_args.minlenratio * h.size(1))
        logging.info("ASR: max output length: " + str(maxlen))
        logging.info("ASR: min output length: " + str(minlen))

        hyp = {"score": 0.0, "yseq": [y]}
        hyps = [hyp]
        ended_hyps = []

        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("ASR: position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.decoder_asr.forward_one_step, (ys, ys_mask, h)
                        )
                    local_att_scores = traced_decoder(ys, ys_mask, h)[0]
                else:
                    local_att_scores = self.decoder_asr.forward_one_step(
                        ys, ys_mask, h
                    )[0]

                local_scores = local_att_scores

                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam, dim=1
                )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("ASR: number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "ASR: best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("ASR: adding <eos> in the last postion in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("ASR: end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("ASR: remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("ASR: no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "ASR: hypo: "
                        + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("ASR: number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]

        logging.info("ASR: total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "ASR: normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def recognize_mt(self, h, trans_args, char_list=None, rnnlm=None, use_jit=False):
        # preprate sos
        if getattr(trans_args, "tgt_lang", False):
            if self.replace_sos:
                y = char_list.index(trans_args.tgt_lang)
        else:
            y = self.sos
        logging.info("MT: <sos> index: " + str(y))
        logging.info("MT: <sos> mark: " + char_list[y])

        logging.info("MT: input lengths: " + str(h.size(1)))
        # search parms
        beam = trans_args.beam_size
        penalty = trans_args.penalty

        vy = h.new_zeros(1).long()

        if trans_args.maxlenratio == 0:
            maxlen = h.shape[1]
        else:
            # maxlen >= 1
            maxlen = max(1, int(trans_args.maxlenratio * h.size(1)))
        minlen = int(trans_args.minlenratio * h.size(0))
        logging.info("MT: max output length: " + str(maxlen))
        logging.info("MT: min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        hyps = [hyp]
        ended_hyps = []

        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("MT: position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.decoder.forward_one_step, (ys, ys_mask, h)
                        )
                    local_att_scores = traced_decoder(ys, ys_mask, h)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(ys, ys_mask, h)[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + trans_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam, dim=1
                )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("MT: number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "MT: best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("MT: adding <eos> in the last postion in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += trans_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and trans_args.maxlenratio == 0.0:
                logging.info("MT: end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("MT: remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("MT: no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "MT: hypo: "
                        + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("MT: number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), trans_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "MT: there is no N-best results, perform translation "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            trans_args = Namespace(**vars(trans_args))
            trans_args.minlenratio = max(0.0, trans_args.minlenratio - 0.1)
            return self.recognize_mt(h, trans_args, char_list, rnnlm)

        logging.info("MT: total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "MT: normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def translate(
        self,
        x,
        y,
        trans_args,
        char_list=None,
        rnnlm=None,
        use_jit=False,
    ):
        """Translate input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace trans_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        # 1. forward encoder
        enc_output = self.encode(x).unsqueeze(0)  # (n_batch, tmax, dunits)

        # 2. forward lstm
        n_batch, tmax, dunits = enc_output.size()
        enc_output = enc_output.view(n_batch * tmax, dunits)
        h_init = enc_output.new_zeros(enc_output.size(0), self.adim)
        c_init = enc_output.new_zeros(enc_output.size(0), self.adim)
        lstm_h, lstm_c = self.lstm(
            enc_output, (h_init, c_init)
        )  # (n_batch * tmax, dunits)

        h = lstm_h.view(n_batch, tmax, dunits)

        # 3. greedy search of asr
        best_src_hyp = self.recognize_asr(h, trans_args, char_list)

        # 4. forward encoder mt
        h_mt = self.encode_mt(best_src_hyp[0]["yseq"][1:-1]).unsqueeze(0)
        h_mt_attn = self.attn_mt(h, h_mt, h_mt, None)

        # 5. forward lstm
        n_batch, tmax, dunits = h_mt_attn.size()
        h_mt_attn = h_mt_attn.view(n_batch * tmax, dunits)
        lstm_h, lstm_c = self.lstm(
            h_mt_attn, (lstm_h, lstm_c)
        )  # (n_batch * tmax, dunits)

        h_mt_attn = lstm_h.view(n_batch, tmax, dunits)

        # 6. beam search for decoder
        best_mt_hyp = self.recognize_mt(h_mt_attn, trans_args, char_list)

        return best_src_hyp, best_mt_hyp

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_src:
            batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, ys_pad_src)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention) and m.attn is not None
            ):  # skip MHA for submodules
                ret[name] = m.attn.cpu().numpy()
        self.train()
        return ret

    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_src:
            batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.asr_weight == 0 or self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, ys_pad_src)
        ret = None
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret
