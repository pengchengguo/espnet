import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from packaging.version import parse as V

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel as BaseESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


def get_similarity_weight(utt_list: List[str]):
    """Get speaker similarity weight based on the utt_list."""
    weight = torch.zeros(len(utt_list), len(utt_list))
    for i, utt_i in enumerate(utt_list):
        for j, utt_j in enumerate(utt_list):
            idx_i = int(utt_i[-1]) - 1
            spk_i = utt_i.split("_")[idx_i].split("-")[0]

            idx_j = int(utt_j[-1]) - 1
            spk_j = utt_j.split("_")[idx_j].split("-")[0]

            weight[i, j] = int(spk_i == spk_j)

    return weight


def get_similarity_weight_wsj2mix(utt_list: List[str]):
    """Get speaker similarity weight based on the utt_list. (for wsj2mix data)"""
    weight = torch.zeros(len(utt_list), len(utt_list))
    for i, utt_i in enumerate(utt_list):
        for j, utt_j in enumerate(utt_list):
            spk_i = utt_i.split("_")[-1][:3]
            spk_j = utt_j.split("_")[-1][:3]

            weight[i, j] = int(spk_i == spk_j)

    return weight


def get_similarity_weight_ami(utt_list: List[str]):
    """Get speaker similarity weight based on the utt_list. (for ami data)"""
    weight = torch.zeros(len(utt_list), len(utt_list))
    for i, utt_i in enumerate(utt_list):
        for j, utt_j in enumerate(utt_list):
            spk_i = utt_i.split("_")[3]
            spk_j = utt_j.split("_")[3]

            weight[i, j] = int(spk_i == spk_j)

    return weight


class TgtSpkQformerESPnetASRModel(BaseESPnetASRModel):
    """CTC-attention hybrid Encoder-Decoder model
    Not Finished.
    """

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        spkencoder: Optional[torch.nn.Module] = None,
        adapter: Optional[torch.nn.Module] = None,
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
    ):
        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            aux_ctc=aux_ctc,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            sym_sos=sym_sos,
            sym_eos=sym_eos,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            lang_token_id=lang_token_id,
        )

        self.spkencoder = spkencoder
        self.adapter = adapter

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        enroll: torch.Tensor,
        enroll_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            enroll: (Batch, Length, ...)
            enroll_lengths: (Batch, )
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        assert speech.shape[0] == enroll.shape[0] == enroll_lengths.shape[0], (
            speech.shape,
            enroll.shape,
            enroll_lengths.shape,
        )

        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # 2. Speaker Encoder
        enroll_feats, enroll_feats_lens = self.encode_spk(enroll, enroll_lengths)

        # 3. Speaker Adaptation
        if self.adapter is not None:
            spk_prompt = self.adapter(
                encoder_out, encoder_out_lens, enroll_feats, enroll_feats_lens
            )

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        stats = dict()
        # 3. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # 4. Attention decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths, spk_prompt
            )

        # 5. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode_spk(self, enroll: torch.Tensor, enroll_lengths: torch.Tensor):
        """Forward speaker encoder and obtain speaker features.

        Args:
            enroll (torch.Tensor): enrollment speech or speaker embedding.
            enroll_lengths (torch.Tensor): length of enrollment speech or
                speaker embedding
        """
        with autocast(False):
            # 1. Extract feats
            enroll_feats, enroll_feats_lengths = self._extract_feats(
                enroll, enroll_lengths
            )

        with torch.no_grad():
            if self.spkencoder is not None:
                enroll_feats, enroll_feats_lengths, _ = self.spkencoder(
                    enroll_feats, enroll_feats_lengths
                )
            else:
                # use speech encoder as speaker encoder
                enroll_feats, enroll_feats_lengths, _ = self.encoder(
                    enroll_feats, enroll_feats_lengths
                )

        return enroll_feats, enroll_feats_lengths

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        spk_prompt: torch.Tensor,
    ):
        if hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            ys_pad = torch.cat(
                [
                    self.lang_token_id.repeat(ys_pad.size(0), 1).to(ys_pad.device),
                    ys_pad,
                ],
                dim=1,
            )
            ys_pad_lens += 1

        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
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


class TgtSpkQformerESPnetASRModel_V2(BaseESPnetASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            aux_ctc=aux_ctc,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            sym_sos=sym_sos,
            sym_eos=sym_eos,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            lang_token_id=lang_token_id,
        )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        enroll: torch.Tensor,
        enroll_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            enroll: (Batch, Length, ...)
            enroll_lengths: (Batch, )
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
            == enroll.shape[0]
            == enroll_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
            enroll.shape,
            enroll_lengths.shape,
        )

        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens, spk_prompt, enroll_embedding = self.encode(
            speech, speech_lengths, enroll, enroll_lengths
        )

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        stats = dict()

        # 2a. CTC branch
        if self.ctc_weight != 0.0:
            prompt_lens = spk_prompt.size(1)
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out[:, prompt_lens:],
                encoder_out_lens - prompt_lens,
                text,
                text_lengths,
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # 2b. Attention decoder branch
        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths, spk_prompt
        )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        enroll: torch.Tensor,
        enroll_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            enroll: (Batch, Dim) for embedding enrollment
                    or (Batch, Length) for speech enrollment
            enroll_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

            # 4. Extract feats for enrollment
            enroll_feats, enroll_feats_lengths = self._extract_feats(
                enroll, enroll_lengths
            )

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 5. Forward encoder
        encoder_out, encoder_out_lens, spk_prompt, enroll_embedding = self.encoder(
            feats, feats_lengths, enroll_feats, enroll_feats_lengths
        )

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        return encoder_out, encoder_out_lens, spk_prompt, enroll_embedding

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        spk_prompt: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, spk_prompt
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
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


class TgtSpkQformerESPnetASRModel_V3(TgtSpkQformerESPnetASRModel_V2):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        sim_weight: float = 1.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            aux_ctc=aux_ctc,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            sym_sos=sym_sos,
            sym_eos=sym_eos,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            lang_token_id=lang_token_id,
        )

        self.sim_weight = sim_weight
        if self.sim_weight > 0.0:
            self.sim_head = torch.nn.Linear(self.encoder.output_size(), 2)
            self.temp = torch.nn.Parameter(0.07 * torch.ones([]))

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        enroll: torch.Tensor,
        enroll_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            enroll: (Batch, Length, ...)
            enroll_lengths: (Batch, )
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
            == enroll.shape[0]
            == enroll_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
            enroll.shape,
            enroll_lengths.shape,
        )

        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 0. Select negative samples for computing similarity loss
        # compute negtive sample probability, (batch, batch)
        sim_weight = get_similarity_weight(kwargs["utt_id"])
        neg_weight = torch.ones_like(sim_weight).masked_fill_(sim_weight == 1, -10000)
        neg_weight = F.softmax(neg_weight, dim=1)

        # select a negative speech sample for each enrollment sample
        speech_neg = []
        speech_lengths_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(neg_weight[b], 1).item()
            speech_neg.append(speech[neg_idx])
            speech_lengths_neg.append(speech_lengths[neg_idx])
        speech_neg = torch.stack(speech_neg, dim=0)
        speech_lengths_neg = torch.stack(speech_lengths_neg, dim=0)

        # select a negative enrollment sample for each speech sample
        enroll_neg = []
        enroll_lengths_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(neg_weight[b], 1).item()
            enroll_neg.append(enroll[neg_idx])
            enroll_lengths_neg.append(enroll_lengths[neg_idx])
        enroll_neg = torch.stack(enroll_neg, dim=0)
        enroll_lengths_neg = torch.stack(enroll_lengths_neg, dim=0)

        # [pos, neg, pos] for speech
        speech_all = torch.cat([speech, speech_neg, speech], dim=0)
        speech_lengths_all = torch.cat(
            [speech_lengths, speech_lengths_neg, speech_lengths], dim=0
        )

        # [pos, pos, neg] for enrollment
        enroll_all = torch.cat([enroll, enroll, enroll_neg], dim=0)
        enroll_lengths_all = torch.cat(
            [enroll_lengths, enroll_lengths, enroll_lengths_neg], dim=0
        )

        # 1. Encoder
        encoder_out, encoder_out_lens, spk_prompt, enroll_embedding = self.encode(
            speech_all, speech_lengths_all, enroll_all, enroll_lengths_all
        )

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_sim, acc_sim = None, None
        stats = dict()

        # 2a. Similarity Loss
        if self.sim_weight > 0.0:
            sim_output = self.sim_head(spk_prompt)
            # average across the query length
            sim_logit = sim_output.mean(dim=1)
            sim_label = torch.cat(
                [
                    torch.ones(batch_size, dtype=torch.long),
                    torch.zeros(2 * batch_size, dtype=torch.long),
                ],
                dim=0,
            ).to(speech.device)

            loss_sim = F.cross_entropy(sim_logit, sim_label)
            sim_pred = sim_logit.argmax(dim=-1)
            sim_acc = float(torch.sum(sim_pred == sim_label)) / float(sim_label.size(0))

            stats["loss_sim"] = loss_sim.detach() if loss_sim is not None else None
            stats["sim_acc"] = sim_acc

        # only keep the pos samples
        encoder_out = encoder_out[:batch_size]
        encoder_out_lens = encoder_out_lens[:batch_size]
        spk_prompt = spk_prompt[:batch_size]

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            prompt_lens = spk_prompt.size(1)
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out[:, prompt_lens:],
                encoder_out_lens - prompt_lens,
                text,
                text_lengths,
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # 2c. Attention decoder branch
        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths, spk_prompt
        )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        if self.sim_weight != 0.0:
            loss = loss + self.sim_weight * loss_sim

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight


class TgtSpkQformerESPnetASRModel_V4(TgtSpkQformerESPnetASRModel_V2):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        contrastive_type: str = "w2v2",
        contrastive_weight: float = 1.0,
        contrastive_temp: float = 0.1,
        num_negatives: int = 10,
        is_wsj2mix: bool = False,
        is_ami: bool = False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            aux_ctc=aux_ctc,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            sym_sos=sym_sos,
            sym_eos=sym_eos,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            lang_token_id=lang_token_id,
        )

        self.contrastive_type = contrastive_type
        self.contrastive_weight = contrastive_weight
        self.contrastive_temp = contrastive_temp
        self.num_negatives = num_negatives
        self.is_wsj2mix = is_wsj2mix
        self.is_ami = is_ami

        logging.info(f"Speaker prompt for encoder: {self.encoder.use_spk_prompt}")
        logging.info(f"Speaker prompt for decoder: {self.decoder.use_spk_prompt}")

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        enroll: torch.Tensor,
        enroll_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            enroll: (Batch, Length, ...)
            enroll_lengths: (Batch, )
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
            == enroll.shape[0]
            == enroll_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
            enroll.shape,
            enroll_lengths.shape,
        )

        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # compute negtive sample probability, (batch, batch)
        if self.is_wsj2mix:
            sim_weight = get_similarity_weight_wsj2mix(kwargs["utt_id"])
        elif self.is_ami:
            sim_weight = get_similarity_weight_ami(kwargs["utt_id"])
        else:
            sim_weight = get_similarity_weight(kwargs["utt_id"])
        neg_weight = torch.ones_like(sim_weight).masked_fill_(sim_weight == 1, -10000)
        neg_weight = F.softmax(neg_weight, dim=1)

        # 1. Encoder
        encoder_out, encoder_out_lens, spk_prompt, enroll_embedding = self.encode(
            speech, speech_lengths, enroll, enroll_lengths
        )

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_con, acc_con = None, None
        stats = dict()

        # 2a. contrastive Loss
        if self.contrastive_weight > 0.0:
            if self.contrastive_type == "w2v2":
                loss_con, acc_con = self._calc_w2v2_contrastive_loss(
                    spk_prompt, enroll_embedding, neg_weight
                )
            else:
                raise NotImplementedError(f"contrastive_type={self.contrastive_type}")

            # collect contrastive branch stats
            stats["loss_con"] = loss_con.detach() if loss_con is not None else None
            stats["acc_con"] = acc_con if acc_con is not None else None

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            prompt_encoder = self.encoder.use_spk_prompt
            prompt_lens = spk_prompt.size(1)
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out[:, prompt_lens:] if prompt_encoder else encoder_out,
                encoder_out_lens - prompt_lens if prompt_encoder else encoder_out_lens,
                text,
                text_lengths,
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # 2c. Attention decoder branch
        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths, spk_prompt
        )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        if self.contrastive_weight > 0.0:
            loss = loss + self.contrastive_weight * loss_con

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_w2v2_contrastive_loss(
        self,
        spk_prompt: torch.Tensor,
        enroll_emb: torch.Tensor,
        neg_weight: torch.Tensor,
    ):
        # (batch_size, dim)
        spk_prompt = spk_prompt.mean(dim=1)
        enroll_emb = enroll_emb.mean(dim=1)

        # get negative enrollment embeddings
        batch_size = spk_prompt.size(0)
        neg_enroll_emb = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(
                neg_weight[b], self.num_negatives, replacement=True
            )
            neg_enroll_emb.append(enroll_emb[neg_idx])
        # (num_negatives, batch_size, dim)
        neg_enroll_emb = torch.stack(neg_enroll_emb, dim=1)

        # (1 + num_negatives, batch_size, dim)
        target_enroll_emb = torch.cat([enroll_emb.unsqueeze(0), neg_enroll_emb], dim=0)

        # compute cosine similarity, (1 + num_negatives, batch_size)
        logits = torch.cosine_similarity(
            spk_prompt.float(), target_enroll_emb.float(), dim=-1
        )
        logits = logits / self.contrastive_temp
        logits = logits.type_as(spk_prompt)
        # (batch_size, 1 + num_negatives)
        logits = logits.transpose(0, 1).contiguous()
        # generate targets, (batch_size)
        target = logits.new_zeros(logits.size(0), dtype=torch.long)

        # compute contrastive loss
        loss_con = F.cross_entropy(logits, target)
        # compute contrastive accuracy
        pred_con = logits.argmax(dim=-1)
        acc_con = float(torch.sum(pred_con == target)) / float(target.size(0))

        return loss_con, acc_con
