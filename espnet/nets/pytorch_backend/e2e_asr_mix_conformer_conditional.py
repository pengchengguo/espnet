# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Conformer speech recognition model (pytorch).

It is a fusion of `e2e_asr_transformer.py`
Refer to: https://arxiv.org/abs/2005.08100

"""

from argparse import Namespace
from distutils.util import strtobool
import logging
import math
import random

import numpy
import torch

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.conformer.argument import (
    verify_rel_pos_type,  # noqa: H301
)
from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E as E2EConformer
from espnet.nets.pytorch_backend.e2e_asr_mix import E2E as E2EASRMIX
from espnet.nets.pytorch_backend.e2e_asr_mix_transformer_conditional import StopBCELoss
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.conditional import ConditionalModule
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask


class E2E(E2EConformer):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2EConformer.add_arguments(parser)
        E2EASRMIX.encoder_mix_add_arguments(parser)
        E2E.encoder_argument(parser)
        E2E.loss_argument(parser)
        return parser

    @staticmethod
    def encoder_argument(parser):
        """Add arguments for the encoder."""
        group = parser.add_argument_group("E2E encoder setting")
        group.add_argument(
            "--elayers_cond",
            default=1,
            type=int,
            help="Number of encoder layers (for shared recognition part "
            "in multi-speaker asr mode)",
        )
        group.add_argument(
            "--eunits_cond",
            default=300,
            type=int,
            help="Number of encoder hidden units",
        )
        group.add_argument(
            "--elayers_rec",
            default=1,
            type=int,
            help="Number of encoder layers (for shared recognition part "
            "in multi-speaker asr mode)",
        )
        return parser

    @staticmethod
    def loss_argument(parser):
        """Add arguments for the loss."""
        group = parser.add_argument_group("E2E loss setting")
        group.add_argument(
            "--sampling-probability",
            default=0.0,
            type=float,
            help="Ratio of predicted labels fed back to decoder",
        )
        parser.add_argument(
            "--use-ctc-alignment",
            default=True,
            type=strtobool,
            help="Use CTC alignments as conditions or use pre-softmax hidden "
            "representations as conditions.",
        )
        group.add_argument(
            "--use-inter-ctc",
            default=False,
            type=strtobool,
            help="Whether to use intermediate CTC regularization loss.",
        )
        group.add_argument(
            "--inter-ctc-weight",
            default=0.3,
            type=float,
            help="Weight of the intermediate CTC regularization loss.",
        )
        group.add_argument(
            "--use-stop-sign-ctc",
            default=False,
            type=strtobool,
            help="Use an additional blank sequence as the last label for stop process.",
        )
        group.add_argument(
            "--use-stop-sign-bce",
            default=False,
            type=strtobool,
            help="Use an additional bce loss to predict the stop process.",
        )
        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super().__init__(idim, odim, args, ignore_id)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        # Check the relative positional encoding type
        args = verify_rel_pos_type(args)

        self.encoder_condition = ConditionalModule(
            eprojs=args.adim,
            odim=odim,
            ctype="lstm",
            clayers=args.elayers_cond,
            cunits=args.eunits_cond,
            eos=self.eos,
            use_ctc_alignment=args.use_ctc_alignment,
            dropout_rate=args.dropout_rate,
            num_spkrs=args.num_spkrs,
        )

        self.encoder_recognition = Encoder(
            idim=args.eunits_cond,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers_rec,
            input_layer="linear",
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            pos_enc_layer_type=args.transformer_encoder_pos_enc_layer_type,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            activation_type=args.transformer_encoder_activation_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            zero_triu=args.zero_triu,
            cnn_module_kernel=args.cnn_module_kernel,
        )

        self.reset_parameters(args)
        assert args.mtlalpha > 0.0
        self.ctc = CTC(
            odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=False
        )

        self.use_ctc_alignment = args.use_ctc_alignment
        self.use_inter_ctc = args.use_inter_ctc
        if self.use_inter_ctc:
            self.inter_ctc_weight = args.inter_ctc_weight
            self.project_linear = torch.nn.Linear(args.eunits_cond, args.adim)
        self.use_stop_sign_ctc = args.use_stop_sign_ctc
        self.use_stop_sign_bce = args.use_stop_sign_bce
        if self.use_stop_sign_bce:
            self.stop_sign_loss = StopBCELoss(args.adim, 1, nunits=args.adim)

        self.num_spkrs = args.num_spkrs
        self.blank = args.sym_blank
        self.sampling_probability = args.sampling_probability

    def min_ctc_loss_and_perm(self, hs_pad, hs_len, ys_pad):
        """E2E min ctc loss and permutation.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor hs_len: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, num_spkrs', Lmax)
        :rtype: torch.Tensor
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: minimum index
        """
        _, n_left_spkrs, _ = ys_pad.size()
        loss_stack = torch.stack(
            [self.ctc(hs_pad, hs_len, ys_pad[:, i]) for i in range(n_left_spkrs)]
        )  # (N, B, 1)
        min_loss, min_idx = torch.min(loss_stack, 0)
        return min_loss, min_idx

    def resort_sequence(self, xs_pad, min_idx, start):
        """E2E re-sort sequence according to min_idx.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, num_spkrs, Lmax)
        :param torch.Tensor min_idx: batch of min idx (B)
        :param int start: current head of sequence.
        :rtype: torch.Tensor
        :return: re-sorted sequence
        """
        n_batch = xs_pad.size(0)
        for i in range(n_batch):
            tmp = xs_pad[i, start].clone()
            xs_pad[i, start] = xs_pad[i, min_idx[i]]
            xs_pad[i, min_idx[i]] = tmp
        return xs_pad

    def forward(self, xs_pad, ilens, ys_pad, ys_ctc_align_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences
                                    (B, num_spkrs, Lmax)
        :param torch.Tensor ys_ctc_align_pad: batch of padded forced alignment sequences
                                    (B, num_spkrs, Tmax')
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        self.hs_pad = hs_pad

        # 2. ctc
        cer_ctc = None
        assert self.mtlalpha > 0.0
        batch_size = xs_pad.size(0)
        num_spkrs = ys_pad.size(1)
        hs_len = hs_mask.view(batch_size, -1).sum(1)
        prev_states = None
        hs_pad_sd, loss_ctc, loss_inter_ctc, loss_stop = (
            [None] * num_spkrs,
            [None] * num_spkrs,
            [None] * num_spkrs,
            [None] * num_spkrs,
        )

        if self.use_ctc_alignment:
            align_ctc_state = ys_ctc_align_pad.new_zeros(
                ys_ctc_align_pad[:, 0].size()
            )  # (B, Tmax)
        else:
            align_ctc_state = hs_pad.new_zeros(hs_pad.size())

        for i in range(num_spkrs):
            condition_out, prev_states = self.encoder_condition(
                hs_pad, align_ctc_state, hs_len, prev_states
            )
            hs_pad_sd[i], hs_mask = self.encoder_recognition(condition_out, hs_mask)
            loss_ctc[i], min_idx = self.min_ctc_loss_and_perm(
                hs_pad_sd[i], hs_len, ys_pad[:, i:]
            )
            min_idx = min_idx + i

            if i < num_spkrs - 1:
                ys_pad = self.resort_sequence(ys_pad, min_idx, i)
                if self.use_ctc_alignment:
                    ys_ctc_align_pad = self.resort_sequence(
                        ys_ctc_align_pad, min_idx, i
                    )

            if self.use_inter_ctc:
                project_out = self.project_linear(condition_out)
                loss_inter_ctc[i] = self.ctc(project_out, hs_len, ys_pad[:, i])

            if self.use_ctc_alignment:
                if random.random() < self.sampling_probability:
                    logging.info("shceduled sampling.")
                    align_ctc_state = self.ctc.argmax(hs_pad_sd[i]).data
                else:
                    align_ctc_state = ys_ctc_align_pad[:, i]
            else:
                logging.info("using latent representation as soft conditions.")
                align_ctc_state = hs_pad_sd[i].detach().data

            if self.use_stop_sign_bce:
                stop_label = hs_pad_sd[i].new_zeros((batch_size, 1))
                if i == num_spkrs - 1:
                    stop_label += 1
                loss_stop[i] = self.stop_sign_loss(hs_pad_sd[i], hs_len, stop_label)

        # The left information
        if self.use_stop_sign_ctc:
            # blank_ctc_label = ys_pad.new_zeros(ys_pad[:, 0].size())  # (B, Lmax)
            # blank_ctc_label = blank_ctc_label.masked_fill_(ys_pad[:, 0].lt(0), -1)
            blank_ctc_label = hs_pad_sd[-1].new_zeros(hs_pad_sd[-1].size())  # (B, Tmax)
            blank_ctc_label = blank_ctc_label.masked_fill_(
                hs_mask.view(batch_size, -1, 1).lt(0), -1
            )
            loss_ctc.append(self.ctc(hs_pad_sd[-1], hs_len, blank_ctc_label))
            if torch.isinf(torch.sum(torch.stack(loss_ctc))):
                print(loss_ctc)
                exit()

        loss_ctc = torch.stack(loss_ctc, dim=0).mean()  # (num_spkrs, B)
        logging.info("ctc loss:" + str(float(loss_ctc)))

        if self.use_inter_ctc:
            loss_inter_ctc = torch.stack(loss_inter_ctc, dim=0).mean()  # (num_spkrs, B)
            logging.info("inter ctc loss:" + str(float(loss_inter_ctc)))

        ys_pad = ys_pad.transpose(0, 1)  # (num_spkrs, B, Lmax)
        ys_out_len = [
            float(torch.sum(ys_pad[i] != self.ignore_id)) for i in range(num_spkrs)
        ]

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        if self.error_calculator is not None:
            cer_ctc = []
            for i in range(num_spkrs):
                ys_hat = self.ctc.argmax(hs_pad_sd[i]).data
                cer_ctc.append(
                    self.error_calculator(ys_hat.cpu(), ys_pad[i].cpu(), is_ctc=True)
                )
            cer_ctc = sum(map(lambda x: x[0] * x[1], zip(cer_ctc, ys_out_len))) / sum(
                ys_out_len
            )
        else:
            cer_ctc = None

        # 3. forward decoder
        if self.mtlalpha == 1.0:
            loss_att, self.acc, cer, wer = None, None, None, None
        else:
            pred_pad, pred_mask = [None] * self.num_spkrs, [None] * self.num_spkrs
            loss_att, acc = [None] * self.num_spkrs, [None] * self.num_spkrs
            for i in range(num_spkrs):
                (
                    pred_pad[i],
                    pred_mask[i],
                    loss_att[i],
                    acc[i],
                ) = self.decoder_and_attention(
                    hs_pad_sd[i], hs_mask, ys_pad[i], batch_size
                )

            # 4. compute attention loss
            # The following is just an approximation
            loss_att = sum(map(lambda x: x[0] * x[1], zip(loss_att, ys_out_len))) / sum(
                ys_out_len
            )
            self.acc = sum(map(lambda x: x[0] * x[1], zip(acc, ys_out_len))) / sum(
                ys_out_len
            )

            # 5. compute cer/wer
            if self.training or self.error_calculator is None:
                cer, wer = None, None
            else:
                ys_hat = pred_pad.argmax(dim=-1)
                cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copyied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
            loss_inter_ctc_data = None
        elif alpha == 1:
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
            if self.use_inter_ctc:
                self.loss = (
                    self.inter_ctc_weight * loss_inter_ctc
                    + (1 - self.inter_ctc_weight) * loss_ctc
                )
                loss_inter_ctc_data = float(loss_inter_ctc)
            else:
                self.loss = loss_ctc
                loss_inter_ctc_data = None
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)
            loss_inter_ctc_data = None

        if self.use_stop_sign_bce:
            self.loss += loss_stop

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            # TODO: support reporting for the loss_inter_ctc_data
            # For now, we only replace the attention loss with the loss_inter_ctc_loss
            # since the we only run CTC model.
            self.reporter.report(
                loss_ctc_data,
                loss_inter_ctc_data,
                self.acc,
                cer_ctc,
                cer,
                wer,
                loss_data,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    def decoder_and_attention(self, hs_pad, hs_mask, ys_pad):
        """Forward decoder and attention loss."""
        # forward decoder
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

        # compute attention loss
        loss_att = self.criterion(pred_pad, ys_out_pad)
        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )
        return pred_pad, pred_mask, loss_att, acc

    def recognize_for_one_spkr(
        self,
        enc_output,
        enc_len,
        alignment,
        recog_args,
        char_list=None,
        rnnlm=None,
        use_jit=False,
        prev_states=None,
    ):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        hs_pad, prev_states = self.encoder_condition(
            enc_output, alignment, enc_len, prev_states
        )
        hs_pad, _ = self.encoder_recognition(hs_pad, None)

        if self.mtlalpha == 1.0:
            recog_args.ctc_weight = 1.0
            logging.info("Set to pure CTC decoding mode.")

        if self.mtlalpha > 0 and recog_args.ctc_weight == 1.0:
            from itertools import groupby

            lpz = self.ctc.argmax(hs_pad)
            collapsed_indices = [x[0] for x in groupby(lpz[0])]
            blank_id = 0
            if char_list is not None:
                blank_id = char_list.index(self.blank)
            hyp = [x for x in filter(lambda x: x != blank_id, collapsed_indices)]
            nbest_hyps = [{"score": 0.0, "yseq": [self.sos] + hyp}]
            if recog_args.beam_size > 1:
                raise NotImplementedError("Pure CTC beam search is not implemented.")
            # TODO(hirofumi0810): Implement beam search

            if self.use_ctc_alignment:
                alignment = lpz
            else:
                alignment = hs_pad.detach().data
            return nbest_hyps, alignment, prev_states

        elif self.mtlalpha > 0 and recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs_pad)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        raise NotImplementedError("Attention beam search is not implemented.")
        h = enc_output.squeeze(0)

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six

        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

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
                            self.decoder.forward_one_step, (ys, ys_mask, enc_output)
                        )
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(
                        ys, ys_mask, enc_output
                    )[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                        :, local_best_ids[0]
                    ] + ctc_weight * torch.from_numpy(
                        ctc_scores - hyp["ctc_score_prev"]
                    )
                    if rnnlm:
                        local_scores += (
                            recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
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
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last postion in the loop")
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
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def predict_alignment(self, ctc_log_prob, best_seq, char_list=None):
        # print('ctc_log_prob:', ctc_log_prob.size())
        # print('best_seq', best_seq)
        # ret = torch.argmax(ctc_log_prob, dim=1)
        ret = torch.tensor(
            self.forward_process(ctc_log_prob, best_seq, char_list)
        ).unsqueeze(0)
        return ret

    def forward_process(self, lpz, y, char_list):
        """Forward process of getting alignments of CTC

        :param torch.Tensor lpz: log probabilities of CTC (T, odim)
        :param torch.Tensor y: id sequence tensor (L)
        :param list char_list: list of characters
        :return: best alignment results
        :rtype: list
        """

        def interpolate_blank(l, blank_id=0):
            l = numpy.expand_dims(l, 1)
            zero = numpy.zeros((l.shape[0], 1), dtype=numpy.int64)
            l = numpy.concatenate([zero, l], axis=1)
            l = l.reshape(-1)
            l = numpy.append(l, l[0])
            return l

        blank_id = 0
        if char_list is not None:
            blank_id = char_list.index(self.blank)
        y_interp = interpolate_blank(y, blank_id)

        logdelta = (
            numpy.zeros((lpz.size(0), len(y_interp))) - 100000000000.0
        )  # log of zero
        state_path = (
            numpy.zeros((lpz.size(0), len(y_interp)), dtype=numpy.int16) - 1
        )  # state path

        logdelta[0, 0] = lpz[0][y_interp[0]]
        logdelta[0, 1] = lpz[0][y_interp[1]]

        for t in range(1, lpz.size(0)):
            for s in range(len(y_interp)):
                if y_interp[s] == blank_id or s < 2 or y_interp[s] == y_interp[s - 2]:
                    candidates = numpy.array(
                        [logdelta[t - 1, s], logdelta[t - 1, s - 1]]
                    )
                    prev_state = [s, s - 1]
                else:
                    candidates = numpy.array(
                        [
                            logdelta[t - 1, s],
                            logdelta[t - 1, s - 1],
                            logdelta[t - 1, s - 2],
                        ]
                    )
                    prev_state = [s, s - 1, s - 2]
                logdelta[t, s] = numpy.max(candidates) + lpz[t][y_interp[s]]
                state_path[t, s] = prev_state[numpy.argmax(candidates)]

        state_seq = -1 * numpy.ones((lpz.size(0), 1), dtype=numpy.int16)

        candidates = numpy.array(
            [logdelta[-1, len(y_interp) - 1], logdelta[-1, len(y_interp) - 2]]
        )
        prev_state = [len(y_interp) - 1, len(y_interp) - 2]
        state_seq[-1] = prev_state[numpy.argmax(candidates)]
        for t in range(lpz.size(0) - 2, -1, -1):
            state_seq[t] = state_path[t + 1, state_seq[t + 1, 0]]

        output_state_seq = []
        for t in range(0, lpz.size(0)):
            output_state_seq.append(y_interp[state_seq[t, 0]])

        # orig_seq = []
        # for t in range(0, len(y)):
        #     orig_seq.append(char_list[y[t]])

        return output_state_seq

    def recognize_for_one_spkr_v2(
        self,
        enc_output,
        enc_len,
        alignment,
        recog_args,
        char_list=None,
        rnnlm=None,
        use_jit=False,
        prev_states=None,
    ):
        """Recognize input speech.

        :param ndnarray enc_output: encoder outputs (B, T, D) or (T, D)
        :param ndnarray enc_mask: encoder masks (B, T, D) or (T, D)
        :param torch.Tensor enc_len: batch of lengths of encoder output sequences (B)
        :param torch.Tensor alignment: alignment of the last speaker (B, T)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        hs_pad, prev_states = self.encoder_condition(
            enc_output, alignment, enc_len, prev_states
        )
        hs_pad, _ = self.encoder_recognition(hs_pad, None)

        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs_pad)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six

        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # # FIXME: jit does not match non-jit result
                # if use_jit:
                #     if traced_decoder is None:
                #         traced_decoder = torch.jit.trace(
                #             self.decoder.forward_one_step, (ys, ys_mask, enc_output)
                #         )
                #     local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                # else:
                #     local_att_scores = self.decoder.forward_one_step(
                #         ys, ys_mask, enc_output
                #     )[0]
                n_batch, _ = ys.size()
                local_att_scores = torch.zeros(n_batch, self.odim, device=ys.device)

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                        :, local_best_ids[0]
                    ] + ctc_weight * torch.from_numpy(
                        ctc_scores - hyp["ctc_score_prev"]
                    )
                    if rnnlm:
                        local_scores += (
                            recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
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
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last postion in the loop")
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
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection

            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )

        if self.use_ctc_alignment:
            alignment = self.predict_alignment(lpz, nbest_hyps[0]["yseq"], char_list)
        else:
            alignment = hs_pad.detach().data

        return nbest_hyps, alignment, prev_states

    def recognize(
        self, x, recog_args, char_list=None, rnnlm=None, use_jit=False, num_spkrs=2
    ):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        enc_output = self.encode(x).unsqueeze(0)

        if self.use_ctc_alignment:
            alignment = enc_output.new_zeros(enc_output.size()[0:2]).long()
        else:
            alignment = enc_output.new_zeros(enc_output.size())

        if recog_args.beam_size > 1:
            logging.warning("Using beam search and LM rescoring.")
            recog_func = self.recognize_for_one_spkr_v2
        else:
            logging.warning("Using greedy search.")
            recog_func = self.recognize_for_one_spkr

        enc_len = torch.tensor([enc_output.size(1)], device=enc_output.device).long()
        nbest_hyps = []
        prev_states = None
        for ns in range(num_spkrs):
            nbest_hyps_single, alignment, prev_states = recog_func(
                enc_output,
                enc_len,
                alignment,
                recog_args,
                char_list,
                rnnlm,
                use_jit,
                prev_states,
            )
            nbest_hyps.append(nbest_hyps_single)
        return nbest_hyps
