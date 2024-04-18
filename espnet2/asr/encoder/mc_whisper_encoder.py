import copy
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet2.asr.adapter.film_adapter import FiLM
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.specaug.specaug import SpecAug


class LayerNorm(torch.nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(torch.nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(torch.nn.Conv1d):
    def _conv_forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultichannelWhisperEncoder(AbsEncoder):
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
        use_fbank: bool = True,
        in_channel: int = 1,
        adapter_method: str = "additive",
        adapter_normalize: bool = True,
        emb_dim: int = 24,
        eps: float = 1.0e-5,
        use_attn_beamforming: bool = False,
        use_attn_beamforming2: bool = False,
        use_attn_beamforming3: bool = False,
        attn_dim: int = 256,
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

        self.use_fbank = use_fbank
        self.use_attn_beamforming = use_attn_beamforming  # beamforming method 1
        self.use_attn_beamforming2 = use_attn_beamforming2  # beamforming method 2
        self.use_attn_beamforming3 = use_attn_beamforming3  # beamforming method 3
        if not use_fbank:
            from espnet2.enh.encoder.stft_encoder import STFTEncoder

            self.stft_encoder = STFTEncoder(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                window="hann",
            )

            if use_attn_beamforming:
                self.stft_conv = Conv2dEmbedding(self.n_fft + 2, emb_dim, dropout_rate)
                self.cross_attn = MultiHeadedAttention(1, attn_dim, dropout_rate)
                self.norm = torch.nn.LayerNorm(attn_dim)
            elif use_attn_beamforming2:
                assert attn_dim // emb_dim == 2
                self.stft_conv = Conv2dEmbedding(
                    self.n_fft // 2 + 1, emb_dim, dropout_rate
                )
                self.cross_attn = MultiHeadedAttention(1, attn_dim, dropout_rate)
                self.norm = torch.nn.LayerNorm(attn_dim)
            elif use_attn_beamforming3:
                assert attn_dim == emb_dim
                self.stft_conv = Conv2dEmbedding1(
                    self.n_fft // 2 + 1, emb_dim, dropout_rate
                )
                self.self_attn = MultiHeadedAttention(1, attn_dim, dropout_rate)
                self.norm = torch.nn.LayerNorm(attn_dim)
            else:
                t_ksize = 3
                ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
                self.stft_conv = torch.nn.Sequential(
                    torch.nn.Conv2d(2 * in_channel, emb_dim, ks, padding=padding),
                    torch.nn.GroupNorm(1, emb_dim, eps=eps),
                )

        if use_fbank:
            # x2 because of downsampling
            adapter_input_size = self.n_mels * in_channel * 2
        else:
            if use_attn_beamforming or use_attn_beamforming2 or use_attn_beamforming3:
                adapter_input_size = attn_dim
            else:
                adapter_input_size = (self.n_fft // 2 + 1) * emb_dim * 2
        hidden_size = self.encoders.conv2.out_channels
        self.adapter_method = adapter_method
        if adapter_method in ["cat", "additive", "film"]:
            self.adapter = SimpleAdapter(
                adapter_input_size,
                hidden_size,
                adapter_method,
                adapter_normalize,
            )
        # elif adapter_method == "cln":
        #     # init conditional layernorm layers, only for the first encoder layer
        #     attn_ln = ConditionalLayerNorm(
        #         self.n_mels,,
        #         hidden_size,
        #         init_weight=self.encoders.blocks[0].attn_ln.weight.data,
        #         init_bias=self.encoders.blocks[0].attn_ln.bias.data,
        #     )
        #     mlp_ln = ConditionalLayerNorm(
        #         self.n_mels,,
        #         hidden_size,
        #         init_weight=self.encoders.blocks[0].mlp_ln.weight.data,
        #         init_bias=self.encoders.blocks[0].mlp_ln.bias.data,
        #     )
        #     # change the original layernorm to conditional layernorm
        #     setattr(self.encoders.blocks[0], "attn_ln", attn_ln)
        #     setattr(self.encoders.blocks[0], "mlp_ln", mlp_ln)
        else:
            raise ValueError(f"Not supported adapter: {adapter_method}")

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
        input: torch.Tensor,  # (batch_size, [num_channel,] feat_dim, feat_lens), multi-channel version
        ilens: torch.Tensor = None,
        stft_input: torch.Tensor = None,  # (batch_size, 2*num_channel, feat_lens, feat_dim) or (batch_size, num_channel, feat_lens, feat_dim*2) or (batch_size, num_channel, 2, feat_lens, feat_dim), multi-channel version
        stft_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.use_fbank:
            x = F.gelu(self.encoders.conv1(input[:, 0]))
        else:
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
            # x = block(x)
            if layer == 0:
                # only do speaker adaptation in the first layer
                if self.adapter_method in ["cat", "additive", "film"]:
                    if self.use_fbank:
                        # (batch_size, feat_lens, feat_dim, num_channel)
                        input = torch.permute(input, (0, 3, 2, 1))
                        # (batch_size, feat_lens, feat_dim)
                        input = torch.reshape(input, (x.shape[0], x.shape[1], -1))
                        x = self.adapter(x, input)
                    elif self.use_attn_beamforming:
                        # (batch_size, 2*num_channel, feat_lens, feat_dim)
                        bs, n_channel, _, _ = stft_input.shape
                        # (bs * n_channel, feat_lens, feat_dim)
                        stft_input = stft_input.reshape(
                            bs * n_channel,
                            stft_input.shape[2],
                            stft_input.shape[3],
                        )
                        # (bs*n_channel, 1, maxlen)
                        stft_masks = (
                            ~make_pad_mask(
                                stft_lens.unsqueeze(1)
                                .expand(bs, n_channel)
                                .reshape(bs * n_channel),
                                maxlen=stft_input.shape[1],
                            )[:, None, :]
                        ).to(stft_input.device)
                        # different conv from lower (bs * n_channel, feat_lens, attn_dim)
                        stft_input, stft_masks = self.stft_conv(stft_input, stft_masks)
                        stft_input = stft_input.reshape(
                            bs, n_channel, stft_input.shape[1], stft_input.shape[2]
                        )
                        stft_masks = stft_masks.reshape(
                            bs, n_channel, stft_masks.shape[1], stft_masks.shape[2]
                        )
                        attn_output = [
                            self.cross_attn(
                                stft_input[:, 0],
                                stft_input[:, i],
                                stft_input[:, i],
                                stft_masks[:, 0],
                            )
                            for i in range(1, n_channel)
                        ]
                        attn_output = self.norm(sum(attn_output))
                        if self.pad_or_trim:
                            attn_output = self.pad_or_trim(
                                attn_output, x.size(1), axis=1
                            )
                        x = self.adapter(x, attn_output)
                    elif self.use_attn_beamforming2:
                        # (batch_size, num_channel, feat_lens, feat_dim*2)
                        bs, n_channel, _, _ = stft_input.shape
                        # (bs * n_channel, feat_lens, feat_dim)
                        stft_input = stft_input.reshape(
                            bs * n_channel,
                            stft_input.shape[2],
                            stft_input.shape[3],
                        )
                        stft_masks = (
                            ~make_pad_mask(
                                stft_lens.unsqueeze(1)
                                .expand(bs, n_channel)
                                .reshape(bs * n_channel),
                                maxlen=stft_input.shape[1],
                            )[:, None, :]
                        ).to(stft_input.device)
                        # different conv from lower (bs * n_channel, feat_lens, attn_dim)
                        stft_input, stft_masks = self.stft_conv(stft_input, stft_masks)
                        # (bs, n_channel, feat_lens, attn_dim)
                        stft_input = stft_input.reshape(
                            bs, n_channel, stft_input.shape[1], stft_input.shape[2]
                        )
                        # (bs, n_channel/2, feat_lens, attn_dim*2)
                        stft_input = torch.cat(
                            torch.split(stft_input, n_channel // 2, dim=1), dim=-1
                        )
                        stft_masks = stft_masks.reshape(
                            bs, n_channel, stft_masks.shape[1], stft_masks.shape[2]
                        )
                        attn_output = [
                            self.cross_attn(
                                stft_input[:, 0],
                                stft_input[:, i],
                                stft_input[:, i],
                                stft_masks[:, 0],
                            )
                            for i in range(1, n_channel // 2)
                        ]
                        attn_output = self.norm(sum(attn_output))
                        if self.pad_or_trim:
                            attn_output = self.pad_or_trim(
                                attn_output, x.size(1), axis=1
                            )
                        x = self.adapter(x, attn_output)
                    elif self.use_attn_beamforming3:
                        # (batch_size, num_channel, 2, feat_lens, feat_dim)
                        bs, n_channel, _, s_len, _ = stft_input.shape
                        # (bs * n_channel, 2, feat_lens, feat_dim)
                        stft_input = stft_input.reshape(
                            bs * n_channel,
                            stft_input.shape[2],
                            stft_input.shape[3],
                            stft_input.shape[4],
                        )
                        # (bs*n_channel, 1, maxlen)
                        stft_masks = (
                            ~make_pad_mask(
                                stft_lens.unsqueeze(1)
                                .expand(bs, n_channel)
                                .reshape(bs * n_channel),
                                maxlen=stft_input.shape[2],
                            )[:, None, :]
                        ).to(stft_input.device)
                        # different conv from lower (bs * n_channel, feat_lens, attn_dim)
                        stft_input, stft_masks = self.stft_conv(stft_input, stft_masks)
                        # (bs, n_channel, feat_lens, attn_dim)
                        stft_input = stft_input.reshape(
                            bs, n_channel, stft_input.shape[1], stft_input.shape[2]
                        )
                        # (bs, feat_lens, n_channel, attn_dim)
                        stft_input = stft_input.permute(0, 2, 1, 3)
                        # (bs * feat_lens, n_channel, attn_dim * out_ch)
                        stft_input = stft_input.reshape(
                            bs * stft_input.shape[1], n_channel, -1
                        )
                        stft_masks = stft_masks.reshape(
                            bs, n_channel, stft_masks.shape[1], stft_masks.shape[2]
                        )
                        # (bs * feat_lens, n_channel, attn_dim)
                        attn_output = self.self_attn(
                            stft_input, stft_input, stft_input, mask=None
                        )
                        # (bs * feat_len, attn_dim)
                        attn_output = self.norm(sum(attn_output.split(1, dim=1)))
                        # (bs, feat_len, attn_dim)
                        attn_output = attn_output.reshape(bs, s_len, -1)
                        if self.pad_or_trim:
                            attn_output = self.pad_or_trim(
                                attn_output, x.size(1), axis=1
                            )
                        x = self.adapter(x, attn_output)
                    else:
                        # (batch_size, emb_dim, feat_lens, feat_dim)
                        stft_input = self.stft_conv(stft_input)
                        # (batch_size, feat_lens, emb_dim, feat_dim)
                        stft_input = torch.permute(stft_input, (0, 2, 1, 3))
                        if self.pad_or_trim:
                            stft_input = self.pad_or_trim(
                                stft_input, 2 * x.size(1), axis=1
                            )
                        # (batch_size, feat_lens, feat_dim)
                        stft_input = torch.reshape(
                            stft_input, (x.shape[0], x.shape[1], -1)
                        )
                        x = self.adapter(x, stft_input)
                    x = block(x)
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
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # xs_pad = xs_pad[:, :, 0]  # select the first channel
        bs, seq_len, n_ch = xs_pad.shape

        if getattr(self, "use_fbank", True):
            # Use FBANK as input
            if self.do_pad_trim:
                # (bs, T, n_channel)
                xs_pad = self.pad_or_trim(xs_pad, self.pad_samples, axis=1)

            # (num_channel, batch_size, seq_len)
            xs_pad = torch.permute(xs_pad, (2, 0, 1))
            # (num_channel x batch_size, seq_len)
            xs_pad = xs_pad.reshape(n_ch * bs, seq_len)

            feats, feats_lens = self.log_mel_spectrogram(xs_pad, ilens)
            _, fdim, feat_seq_len = feats.shape
            feats = feats.reshape(n_ch, bs, fdim, feat_seq_len)

            if self.specaug is not None and self.encoders.training:
                feats = torch.transpose(feats, 1, 2)
                feats, feats_lens = self.specaug(feats, feats_lens)
                feats = torch.transpose(feats, 1, 2)

            xs_pad, olens = self.whisper_encode(feats, feats_lens)
        else:
            # Use STFT as input
            if self.do_pad_trim:
                # (bs, T)
                xs_pad0 = self.pad_or_trim(xs_pad[:, :, 0], self.pad_samples, axis=-1)

            feats, feats_lens = self.log_mel_spectrogram(xs_pad0, ilens)

            mix_std_ = torch.std(xs_pad, dim=(1, 2), keepdim=True)  # [bs, 1, 1]
            norm_xs_pad = xs_pad / mix_std_  # RMS normalization
            # [bs, T, n_channel, F]
            stft_feats = self.stft_encoder(norm_xs_pad, ilens)[0][:, : feats.shape[2]]
            stft_feats = stft_feats.transpose(1, 2)  # [bs, n_channel, T, F]
            if self.use_attn_beamforming:
                # [B, n_channel, T, 2*F]
                stft_feats = torch.cat((stft_feats.real, stft_feats.imag), dim=-1)
            elif self.use_attn_beamforming2:
                # [B, 2*n_channel, T, F]
                stft_feats = torch.cat((stft_feats.real, stft_feats.imag), dim=1)
            elif self.use_attn_beamforming3:
                # [B, n_channel, 2, T, F]
                stft_feats = torch.stack((stft_feats.real, stft_feats.imag), dim=2)
            else:
                # [B, 2*n_channel, T, F]
                stft_feats = torch.cat((stft_feats.real, stft_feats.imag), dim=1)
            stft_lens = ilens // self.hop_length

            if self.specaug is not None and self.encoders.training:
                feats = torch.transpose(feats, 1, 2)
                feats, feats_lens = self.specaug(feats, feats_lens)
                feats = torch.transpose(feats, 1, 2)

            xs_pad, olens = self.whisper_encode(
                feats, feats_lens, stft_feats, stft_lens
            )

        return xs_pad, olens, None


class SimpleAdapter(torch.nn.Module):
    """adaptation for Whisper encoder."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        adapter_method: str = "cat",
        adapter_normalize: bool = True,
    ):
        super().__init__()

        assert adapter_method in ["cat", "additive", "film", "cln"]
        self.adapter_method = adapter_method
        if adapter_method == "cat":
            self.adapter = torch.nn.Sequential(
                torch.nn.Linear(hidden_size + input_size, hidden_size),
            )
        elif adapter_method == "additive":
            linear_size = 2 * input_size
            self.adapter = torch.nn.Sequential(
                torch.nn.Linear(input_size, linear_size),
                torch.nn.GELU(),
                torch.nn.Linear(linear_size, hidden_size),
            )
        elif adapter_method == "film":
            self.adapter = FiLM(input_size, hidden_size)
        else:
            raise NotImplementedError(f"Not supported adapter: {adapter_method}")

        if adapter_normalize:
            self.adapter_norm = torch.nn.LayerNorm(hidden_size)
        else:
            self.adapter_norm = None

    def forward(self, x: torch.Tensor, enroll: torch.Tensor):
        # enroll = enroll.unsqueeze(1).expand(-1, x.size(1), -1)

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


class Conv2dEmbedding(torch.nn.Module):
    """Conv2dEmbedding module.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dEmbedding object."""
        super(Conv2dEmbedding, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1, padding=1),
            torch.nn.ReLU(),
        )
        # self.out = torch.nn.Sequential(
        #     torch.nn.Linear(odim * (((idim - 1) // 2 - 2)), odim),
        #     pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        # )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim + 1) // 2)), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Pass x through 2 Conv2d layers without subsampling.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim).
                where time' = time - 4.
            torch.Tensor: Subsampled mask (#batch, 1, time').
                where time' = time - 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, ::2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dEmbedding1(torch.nn.Module):
    """Conv2dEmbedding1 module.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dEmbedding1 object."""
        super(Conv2dEmbedding1, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                2, odim, 3, 1, padding=1
            ),  # because real and imag two feature maps
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1, padding=1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * idim, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Pass x through 2 Conv2d layers without subsampling.

        Args:
            x (torch.Tensor): Input tensor (#batch, c, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim).
                where time' = time.
            torch.Tensor: Subsampled mask (#batch, 1, time').
                where time' = time.

        """
        # x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]
