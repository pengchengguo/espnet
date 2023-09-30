from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.mt.espnet_model import ESPnetMTModel
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet.nets.e2e_asr_common import ErrorCalculator as ASRErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetDiscreteSTModel(ESPnetMTModel):
    """Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        hier_encoder: Optional[AbsEncoder],
        decoder: AbsDecoder,
        ctc: Optional[CTC],
        st_ctc: Optional[CTC],
        ctc_weight: float = 0.3,
        st_ctc_weight: float = 0.3,
        interctc_weight: float = 0.0,
        src_vocab_size: int = 0,
        src_token_list: Union[Tuple[str, ...], List[str]] = [],
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_bleu: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        share_decoder_input_output_embed: bool = False,
        share_encoder_decoder_input_embed: bool = False,
        loss_att_weight_asr: float = 1.0,
        loss_att_weight_st: float = 1.0,
        loss_att_weight_mt: float = 1.0,
        loss_ctc_st_weight_asr: float = 1.0,
        loss_ctc_st_weight_st: float = 1.0,
        loss_ctc_st_weight_mt: float = 1.0,
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        sym_na: Optional[str] = "<na>",  # not available
        sym_sop: Optional[str] = "<sop>",  # start of prev
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            src_vocab_size=src_vocab_size,
            src_token_list=src_token_list,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_bleu=report_bleu,
            sym_space=sym_space,
            sym_blank=sym_blank,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            share_decoder_input_output_embed=share_decoder_input_output_embed,
            share_encoder_decoder_input_embed=share_encoder_decoder_input_embed,
        )

        self.specaug = specaug
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.ctc_weight = ctc_weight
        self.st_ctc_weight = st_ctc_weight
        self.interctc_weight = interctc_weight

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        if st_ctc_weight == 0.0:
            self.st_ctc = None
        else:
            self.st_ctc = st_ctc

        if report_bleu:
            self.error_calculator = ASRErrorCalculator(
                token_list, sym_space, sym_blank, True, True
            )

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.loss_att_weight_asr = loss_att_weight_asr
        self.loss_att_weight_st = loss_att_weight_st
        self.loss_att_weight_mt = loss_att_weight_mt

        self.loss_ctc_st_weight_asr = loss_ctc_st_weight_asr
        self.loss_ctc_st_weight_mt = loss_ctc_st_weight_mt
        self.loss_ctc_st_weight_st = loss_ctc_st_weight_st

        self.hier_encoder = hier_encoder

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        src_text: torch.Tensor,
        src_text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
            src_text: (Batch, length)
            src_text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            text.shape[0]
            == text_lengths.shape[0]
            == src_text.shape[0]
            == src_text_lengths.shape[0]
        ), (text.shape, text_lengths.shape, src_text.shape, src_text_lengths.shape)

        batch_size = src_text.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]
        src_text = src_text[:, : src_text_lengths.max()]

        if "text_ctc" in kwargs:
            text_ctc = kwargs["text_ctc"]
            text_ctc_lengths = kwargs["text_ctc_lengths"]
            assert text_ctc_lengths.dim() == 1, text_ctc_lengths.shape
            assert text_ctc.shape[0] == text_ctc_lengths.shape[0] == text.shape[0], (
                text_ctc.shape,
                text_ctc_lengths.shape,
                text.shape,
            )

            text_ctc = text_ctc[:, : text_ctc_lengths.max()]
        else:
            text_ctc = None

        # 1. Encoder
        if self.hier_encoder is not None or self.postencoder is not None:
            (
                st_encoder_out,
                st_encoder_out_lens,
                asr_encoder_out,
                asr_encoder_out_lens,
            ) = self.encode(src_text, src_text_lengths, return_int_enc=True)
        else:
            st_encoder_out, st_encoder_out_lens = self.encode(
                src_text, src_text_lengths
            )
            asr_encoder_out, asr_encoder_out_lens = st_encoder_out, st_encoder_out_lens

        loss_ctc, cer_ctc = None, None
        stats = dict()

        # 2a. CTC branch
        if self.ctc_weight != 0.0 and text_ctc is not None:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                asr_encoder_out, asr_encoder_out_lens, text_ctc, text_ctc_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # for hier enc
        loss_ctc_st, cer_ctc_st = None, None
        if self.st_ctc_weight != 0.0:
            loss_ctc_st, cer_ctc_st = self._calc_ctc_loss(
                st_encoder_out,
                st_encoder_out_lens,
                text,
                text_lengths,
                st_ctc=True,
            )

            stats["loss_ctc_st"] = (
                loss_ctc_st.detach() if loss_ctc_st is not None else None
            )
            stats["cer_ctc_st"] = cer_ctc_st

        # 2b. Attention-decoder branch (MT)
        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            st_encoder_out, st_encoder_out_lens, text, text_lengths
        )

        if self.st_ctc_weight == 0.0:
            loss_st = loss_att
        else:
            loss_st = (
                self.st_ctc_weight * loss_ctc_st + (1 - self.st_ctc_weight) * loss_att
            )

        if self.ctc_weight > 0:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_st
        else:
            loss = loss_st

        stats["loss_st"] = loss_st.detach() if loss_att is not None else None
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        src_text: torch.Tensor,
        src_text_lengths: torch.Tensor,
        return_int_enc: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by mt_inference.py

        Args:
            src_text: (Batch, Length, ...)
            src_text_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(src_text, src_text_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        if return_int_enc:
            int_encoder_out, int_encoder_out_lens = encoder_out, encoder_out_lens

        if self.hier_encoder is not None:
            encoder_out, encoder_out_lens, _ = self.hier_encoder(
                encoder_out, encoder_out_lens
            )

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == src_text.size(0), (
            encoder_out.size(),
            src_text.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if return_int_enc:
            return encoder_out, encoder_out_lens, int_encoder_out, int_encoder_out_lens

        return encoder_out, encoder_out_lens

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_mt(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        st_ctc: bool = False,
    ):
        # Calc CTC loss
        if st_ctc:
            loss_ctc = self.st_ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc
