import copy
from typing import Any, List, Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.scorer_interface import BatchScorerInterface


class ExpandedTokenEmbedding(torch.nn.Module):
    def __init__(self, ori_emebedding, additional_size):
        super().__init__()
        self.ori_emb = ori_emebedding

        orig_emb_std, orig_emb_mean = torch.std_mean(ori_emebedding.weight)
        self.add_emb = torch.nn.Embedding(additional_size, ori_emebedding.embedding_dim)
        torch.nn.init.normal_(
            self.add_emb.weight,
            orig_emb_mean.item(),
            orig_emb_std.item(),
        )
        self.num_embeddings = ori_emebedding.num_embeddings + additional_size

    @property
    def weight(self):
        return torch.cat([self.ori_emb.weight, self.add_emb.weight], dim=0)

    def forward(self, input):
        return torch.nn.functional.embedding(
            input,
            self.weight,
            self.ori_emb.padding_idx,
            self.ori_emb.max_norm,
            self.ori_emb.norm_type,
            self.ori_emb.scale_grad_by_freq,
            self.ori_emb.sparse,
        )


class OpenAIWhisperDecoder(AbsDecoder, BatchScorerInterface):
    """Transformer-based Speech-to-Text Decoder from OpenAI's Whisper Model:

    URL: https://github.com/openai/whisper
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: str = None,
        load_origin_token_embedding=False,
    ):
        try:
            import whisper
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools && "
                "./installers/install_whisper.sh"
            )
            raise e

        assert check_argument_types()
        super().__init__()

        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(
            whisper_model, download_root=download_dir, device="cpu"
        )
        self.decoders = copy.deepcopy(_model.decoder)
        attention_dim = self.decoders.token_embedding.embedding_dim

        # note that originally Whisper doesn't use dropouts
        self.dropout = torch.nn.Dropout(dropout_rate)

        # load the original token_embeddings, if the vocabulary is expanded
        self.load_origin_token_embedding = load_origin_token_embedding

        # vocab size mismatch -> reinitialize embedding
        # orig vocab size (multilingual): 51865
        # orig vocab size (english): 51864
        if vocab_size != self.decoders.token_embedding.num_embeddings:
            if self.load_origin_token_embedding:
                assert (
                    vocab_size > self.decoders.token_embedding.num_embeddings
                ), "expanded vocab_size should be larged than the origin"
                self.decoders.token_embedding = ExpandedTokenEmbedding(
                    self.decoders.token_embedding,
                    vocab_size - self.decoders.token_embedding.num_embeddings,
                )
            else:
                orig_emb_std, orig_emb_mean = torch.std_mean(
                    self.decoders.token_embedding.weight
                )
                self.decoders.token_embedding = torch.nn.Embedding(
                    vocab_size, attention_dim
                )
                torch.nn.init.normal_(
                    self.decoders.token_embedding.weight,
                    orig_emb_mean.item(),
                    orig_emb_std.item(),
                )

        self.decoders.train()
        del _model

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt, memory = ys_in_pad, hs_pad
        tgt = (
            self.decoders.token_embedding(tgt)
            + self.decoders.positional_embedding[: tgt.size(1)]
        )
        tgt = self.dropout(tgt)

        x = tgt.to(memory.dtype)

        for layer, block in enumerate(self.decoders.blocks):
            x = block(x, memory, mask=self.decoders.mask)
            if layer < len(self.decoders.blocks) - 1:
                x = self.dropout(x)

        x = self.decoders.ln(x)
        x = (
            x @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return x, ys_in_lens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        NOTE (Shih-Lun):
            cache implementation is ignored for now
            for simplicity & correctness
        """
        x = (
            self.decoders.token_embedding(tgt)
            + self.decoders.positional_embedding[: tgt.size(1)]
        )
        x = self.dropout(x)
        x = x.to(memory.dtype)

        for layer, block in enumerate(self.decoders.blocks):
            x = block(x, memory, mask=self.decoders.mask)
            if layer < len(self.decoders.blocks) - 1:
                x = self.dropout(x)

        x = self.decoders.ln(x)
        y = x[:, -1]
        y = (
            y @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        y = torch.log_softmax(y, dim=-1)

        return y, None

    def score(self, ys, state, x):
        """Score."""
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), torch.empty(0), x.unsqueeze(0), cache=state  # dummy mask
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # batch decoding, dummy mask is passed
        logp, states = self.forward_one_step(ys, torch.empty(0), xs, cache=None)

        return logp, None


class QFormerTgtSpkWhisperDecoder_V2(OpenAIWhisperDecoder):
    """QFormer based target speaker Whisper Decoder (V2)"""

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: str = None,
        load_origin_token_embedding=False,
        startofprev_token: int = 50361,
        use_spk_prompt: bool = True,
    ):
        super().__init__(
            vocab_size,
            encoder_output_size,
            dropout_rate,
            whisper_model,
            download_dir,
            load_origin_token_embedding,
        )

        self.startofprev_token = startofprev_token
        self.use_spk_prompt = use_spk_prompt

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        spk_prompt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder."""

        tgt, memory = ys_in_pad, hs_pad
        # (batch, 1)
        startofprev = (
            tgt.new([self.startofprev_token]).unsqueeze(1).expand(tgt.size(0), -1)
        ).contiguous()

        tgt = self.decoders.token_embedding(tgt)
        startofprev = self.decoders.token_embedding(startofprev)
        # concat startofprev tokens, speaker prompts, and target tokens
        if self.use_spk_prompt:
            tgt = torch.cat([startofprev, spk_prompt, tgt], dim=1)

        tgt = tgt + self.decoders.positional_embedding[: tgt.size(1)]
        tgt = self.dropout(tgt)
        x = tgt.to(memory.dtype)

        for layer, block in enumerate(self.decoders.blocks):
            x = block(x, memory, mask=self.decoders.mask)
            if layer < len(self.decoders.blocks) - 1:
                x = self.dropout(x)

        x = self.decoders.ln(x)
        x = (
            x @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        # only compute loss for the part of target tokens
        if self.use_spk_prompt:
            x = x[:, 1 + spk_prompt.size(1) :].contiguous()

        return x, ys_in_lens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        spk_prompt: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
            spe
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        NOTE (Shih-Lun):
            cache implementation is ignored for now
            for simplicity & correctness
        """
        startofprev = (
            tgt.new([self.startofprev_token]).unsqueeze(1).expand(tgt.size(0), -1)
        )

        tgt = self.decoders.token_embedding(tgt)
        startofprev = self.decoders.token_embedding(startofprev)
        # concat startofprev tokens, speaker prompts, and target tokens
        if self.use_spk_prompt:
            if spk_prompt.size(0) != tgt.size(0):
                # for beam size > 1
                spk_prompt = spk_prompt.expand(tgt.size(0), -1, -1)

            tgt = torch.cat([startofprev, spk_prompt, tgt], dim=1)

        x = tgt + self.decoders.positional_embedding[: tgt.size(1)]
        x = self.dropout(x)
        x = x.to(memory.dtype)

        for layer, block in enumerate(self.decoders.blocks):
            x = block(x, memory, mask=self.decoders.mask)
            if layer < len(self.decoders.blocks) - 1:
                x = self.dropout(x)

        x = self.decoders.ln(x)
        y = x[:, -1]
        y = (
            y @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        y = torch.log_softmax(y, dim=-1)

        return y, None

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        speech_prompt: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # batch decoding, dummy mask is passed
        logp, states = self.forward_one_step(
            ys, torch.empty(0), xs, speech_prompt, cache=None
        )

        return logp, None
