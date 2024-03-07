import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typeguard import check_argument_types

from espnet2.asr.adapter.film_adapter import FiLM
from espnet2.asr.adapter.cln_adapter import ConditionalLayerNorm
from espnet2.asr.adapter.qformer_adapter import QFormerAdapter
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.specaug.specaug import SpecAug


class OpenAIWhisperEncoder(AbsEncoder):
    """Transformer-based Speech Encoder from OpenAI's Whisper Model:

    URL: https://github.com/openai/whisper
    """

    def __init__(
        self,
        input_size: int = 1,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: str = None,
        use_specaug: bool = False,
        specaug_conf: Union[dict, None] = None,
        do_pad_trim: bool = False,
    ):
        try:
            import whisper
            from whisper.audio import HOP_LENGTH, N_FFT, N_MELS, N_SAMPLES
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools &&",
                "./installers/install_whisper.sh",
            )
            raise e

        assert check_argument_types()
        super().__init__()

        self.n_fft = N_FFT
        self.win_length = N_FFT
        self.hop_length = HOP_LENGTH
        self.n_mels = N_MELS

        self.mel_filters = whisper.audio.mel_filters

        # note that originally Whisper doesn't use dropouts
        self.dropout = torch.nn.Dropout(dropout_rate)

        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(
            whisper_model, download_root=download_dir, device="cpu"
        )
        self.encoders = copy.deepcopy(_model.encoder)
        self.encoders.train()

        del _model

        if use_specaug:
            self.specaug = SpecAug(**specaug_conf)
        else:
            self.specaug = None

        self.do_pad_trim = do_pad_trim
        self.pad_samples = N_SAMPLES

    def output_size(self) -> int:
        return self.encoders.ln_post.normalized_shape[-1]

    def pad_or_trim(
        self,
        array: torch.Tensor,
        length: int,
        axis: int = -1,
    ) -> torch.Tensor:
        """Pad or trim the audio array to N_SAMPLES.

        Used in zero-shot inference cases.
        """
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length).to(array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])

        return array

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Use log-mel spectrogram computation native to Whisper training"""
        window = torch.hann_window(self.win_length).to(audio.device)
        stft = torch.stft(
            audio, self.n_fft, self.hop_length, window=window, return_complex=True
        )

        # whisper deletes the last frame by default (Shih-Lun)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = self.mel_filters(audio.device, self.n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        if ilens is not None:
            olens = ilens // self.hop_length
        else:
            olens = None

        log_spec = torch.maximum(
            log_spec,
            log_spec.view(audio.size(0), -1).max(dim=-1)[0][:, None, None] - 8.0,
        )
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec, olens

    def whisper_encode(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        x = F.gelu(self.encoders.conv1(input))
        x = F.gelu(self.encoders.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = self.encoders.positional_embedding.size(0)
        if n_frames <= max_pos:
            x = (x + self.encoders.positional_embedding[: x.size(1), :]).to(x.dtype)
        else:
            # due to positional encoding, audios >30 sec won't be accepted
            x = x[:, :max_pos, :] + self.encoders.positional_embedding

        x = self.dropout(x)

        for layer, block in enumerate(self.encoders.blocks):
            x = block(x)
            if layer < len(self.encoders.blocks) - 1:
                x = self.dropout(x)

        x = self.encoders.ln_post(x)

        if ilens is not None:
            olens = (
                1
                + (
                    ilens
                    - self.encoders.conv2.kernel_size[0]
                    + 2 * self.encoders.conv2.padding[0]
                )
                // self.encoders.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        return x, olens

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.do_pad_trim:
            xs_pad = self.pad_or_trim(xs_pad, self.pad_samples)

        feats, feats_lens = self.log_mel_spectrogram(xs_pad, ilens)

        if self.specaug is not None and self.encoders.training:
            feats = torch.transpose(feats, 1, 2)
            feats, feats_lens = self.specaug(feats, feats_lens)
            feats = torch.transpose(feats, 1, 2)

        xs_pad, olens = self.whisper_encode(feats, feats_lens)

        return xs_pad, olens, None


class TgtSpkWhisperEncoder(OpenAIWhisperEncoder):
    """Target speaker adapted Whisper Encoder."""

    def __init__(
        self,
        input_size: int = 1,
        dropout_rate: float = 0,
        whisper_model: str = "small",
        download_dir: str = None,
        use_specaug: bool = False,
        specaug_conf: dict | None = None,
        do_pad_trim: bool = False,
        enroll_size: int = 512,
        adapter_method: str = "cat",
        adapter_normalize: bool = True,
    ):
        super().__init__(
            input_size,
            dropout_rate,
            whisper_model,
            download_dir,
            use_specaug,
            specaug_conf,
            do_pad_trim,
        )

        hidden_size = self.encoders.conv2.out_channels
        self.adapter_method = adapter_method
        if adapter_method in ["cat", "additive", "film"]:
            self.adapter = SpkAdapter(
                enroll_size,
                hidden_size,
                adapter_method,
                adapter_normalize,
            )
        elif adapter_method == "cln":
            # init conditional layernorm layers, only for the first encoder layer
            attn_ln = ConditionalLayerNorm(
                enroll_size,
                hidden_size,
                init_weight=self.encoders.blocks[0].attn_ln.weight.data,
                init_bias=self.encoders.blocks[0].attn_ln.bias.data,
            )
            mlp_ln = ConditionalLayerNorm(
                enroll_size,
                hidden_size,
                init_weight=self.encoders.blocks[0].mlp_ln.weight.data,
                init_bias=self.encoders.blocks[0].mlp_ln.bias.data,
            )
            # change the original layernorm to conditional layernorm
            setattr(self.encoders.blocks[0], "attn_ln", attn_ln)
            setattr(self.encoders.blocks[0], "mlp_ln", mlp_ln)
        else:
            raise ValueError(f"Not supported adapter: {adapter_method}")

    def whisper_encode(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        enroll: torch.Tensor,
    ) -> torch.Tensor:
        x = F.gelu(self.encoders.conv1(input))
        x = F.gelu(self.encoders.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = self.encoders.positional_embedding.size(0)
        if n_frames <= max_pos:
            x = (x + self.encoders.positional_embedding[: x.size(1), :]).to(x.dtype)
        else:
            # due to positional encoding, audios >30 sec won't be accepted
            x = x[:, :max_pos, :] + self.encoders.positional_embedding

        x = self.dropout(x)

        for layer, block in enumerate(self.encoders.blocks):
            if layer == 0:
                # only do speaker adaptation in the first layer
                if self.adapter_method in ["cat", "additive", "film"]:
                    x = self.adapter(x, enroll)
                    x = block(x)
                elif self.adapter_method == "cln":
                    # forward attention layer
                    x = x + block.attn(block.attn_ln(x, enroll))[0]
                    # forward mlp layer
                    x = x + block.mlp(block.mlp_ln(x, enroll))
                else:
                    raise ValueError(f"Not supported adapter: {self.adapter_method}")
            else:
                x = block(x)

            if layer < len(self.encoders.blocks) - 1:
                x = self.dropout(x)

        x = self.encoders.ln_post(x)

        if ilens is not None:
            olens = (
                1
                + (
                    ilens
                    - self.encoders.conv2.kernel_size[0]
                    + 2 * self.encoders.conv2.padding[0]
                )
                // self.encoders.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        return x, olens

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        enroll: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.do_pad_trim:
            xs_pad = self.pad_or_trim(xs_pad, self.pad_samples)

        feats, feats_lens = self.log_mel_spectrogram(xs_pad, ilens)

        if self.specaug is not None and self.encoders.training:
            feats = torch.transpose(feats, 1, 2)
            feats, feats_lens = self.specaug(feats, feats_lens)
            feats = torch.transpose(feats, 1, 2)

        xs_pad, olens = self.whisper_encode(feats, feats_lens, enroll)

        return xs_pad, olens, None


class SpkAdapter(torch.nn.Module):
    """Target speaker adaptation for Whisper encoder."""

    def __init__(
        self,
        enroll_size: int,
        hidden_size: int,
        adapter_method: str = "cat",
        adapter_normalize: bool = True,
    ):
        super().__init__()

        assert adapter_method in ["cat", "additive", "film"]
        self.adapter_method = adapter_method
        if adapter_method == "cat":
            self.adapter = torch.nn.Sequential(
                torch.nn.Linear(hidden_size + enroll_size, hidden_size),
            )
        elif adapter_method == "additive":
            linear_size = 2 * enroll_size
            self.adapter = torch.nn.Sequential(
                torch.nn.Linear(enroll_size, linear_size),
                torch.nn.GELU(),
                torch.nn.Linear(linear_size, hidden_size),
            )
        elif adapter_method == "film":
            self.adapter = FiLM(enroll_size, hidden_size)
        else:
            raise NotImplementedError(f"Not supported adapter: {adapter_method}")

        if adapter_normalize:
            self.adapter_norm = torch.nn.LayerNorm(hidden_size)
        else:
            self.adapter_norm = None

    def forward(self, x: torch.Tensor, enroll: torch.Tensor):
        enroll = enroll.unsqueeze(1).expand(-1, x.size(1), -1)

        if self.adapter_method == "cat":
            fused_emb = torch.cat([x, enroll], dim=-1)
            x = x + self.adapter(fused_emb)
        elif self.adapter_method == "additive":
            x = x + self.adapter(enroll)
        elif self.adapter_method == "film":
            x = self.adapter(x, enroll)
        else:
            raise NotImplementedError(f"Not supported adapter: {self.adapter_method}")

        if self.adapter_norm is not None:
            x = self.adapter_norm(x)

        return x
