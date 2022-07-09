#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.tasks.asr import ASRTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


class FeatExtractor:
    """FeatExtractor class.
    Extract the encoder output features from an ASR model.

    Examples:
        >>> import soundfile
        >>> extractor = FeatExtractor("asr_config.yaml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> extractor(audio)

    """

    def __init__(
        self,
        asr_train_config: str = None,
        asr_model_file: str = None,
        device: str = "cpu",
        dtype: str = "float32",
    ):
        assert check_argument_types()

        # 1. Build ASR model
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, speech: Union[torch.Tensor, np.ndarray]):
        """Extract encoder outputs

        Args:
            data: Input speech data
        Returns:

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, _ = self.asr_model.encode(**batch)

        if isinstance(enc, tuple):
            enc = enc[0]
        assert len(enc) == 1, len(enc)

        return enc

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None, **kwargs: Optional[Any],
    ):
        """Build FeatExtractor instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            FeatExtractor: FeatExtractor instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return FeatExtractor(**kwargs)


def extract_feat(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: Optional[str],
    asr_model_file: Optional[str],
    model_tag: Optional[str],
):
    assert check_argument_types()
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. FeatExtraxtor instance
    extractor_kwargs = dict(
        asr_train_config=asr_train_config, asr_model_file=asr_model_file, device=device,
    )
    extractor = FeatExtractor.from_pretrained(model_tag=model_tag, **extractor_kwargs,)

    # 3. Build data-iterator
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTask.build_preprocess_fn(extractor.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(extractor.asr_train_args, False),
        inference=True,
    )

    # 4. Start for-loop
    count = 0
    for keys, batch in loader:
        assert len(keys) == 1
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
        results = extractor(**batch)
        results = results.squeeze(0).cpu().numpy()
        np.save(os.path.join(output_dir, keys[0] + ".npy"), results)
        count += 1
        logging.info(f"Processing the {count}-th audio.")


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Extract the encoder output features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu", type=int, default=0, help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument(
        "--batch_size", type=int, default=1, help="The batch size for inference",
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--asr_train_config", type=str, help="ASR training configuration",
    )
    group.add_argument(
        "--asr_model_file", type=str, help="ASR model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    extract_feat(**kwargs)


if __name__ == "__main__":
    main()
