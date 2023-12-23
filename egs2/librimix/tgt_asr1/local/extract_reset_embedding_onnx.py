import argparse
import json

from functools import partial
from pathlib import Path
from tqdm.contrib.concurrent import thread_map

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from espnet2.utils.types import str2bool


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path of dataset.",
    )
    parser.add_argument(
        "--is_train",
        type=str2bool,
        default=True,
        help="Whether is the training set or not",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to the pretrained model in ONNX format",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Path of the output embedding directory",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help="Maximum number of workers to process audio files in parallel",
    )
    parser.add_argument(
        "--max_chunksize",
        type=int,
        default=1000,
        help="Maximum size of chunks sent to worker processes",
    )
    return parser


def compute_fbank(
    wav_path, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0
):
    """Extract fbank.

    Simlilar to the one in wespeaker.dataset.processor,
    While integrating the wave reading and CMN.
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=dither,
        sample_frequency=sample_rate,
        window_type="hamming",
        use_energy=False,
    )
    # CMN, without CVN
    mat = mat - torch.mean(mat, dim=0)
    return mat


def worker(utt_path, session, outdir):
    outdir = outdir.absolute()

    utt, path = utt_path
    feats = compute_fbank(path)
    feats = feats.unsqueeze(0).numpy()  # add batch dimension
    embeddings = session.run(output_names=["embs"], input_feed={"feats": feats})
    key, value = Path(path).stem, np.squeeze(embeddings[0])
    p = outdir / f"{key}.npy"
    np.save(p, value)
    return f"{utt} {p}\n"


def main():
    args = get_parser().parse_args()

    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    session = ort.InferenceSession(args.model_dir, sess_options=so)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tup = []
    if args.is_train:
        # extract embeddings for all utterance
        # during the training, we randomly select a utterance for enrollment
        with open(args.data_dir / "spk2enroll.json", "r", encoding="utf-8") as f:
            jsonstring = json.load(f)
        for spk, utt2paths in jsonstring.items():
            for utt, path in utt2paths:
                tup.append((utt, path))
    else:
        with open(args.data_dir / "enroll.scp", "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                utt, path = line.strip().split(maxsplit=1)
                tup.append((utt, path))

    # List[str]
    ret = thread_map(
        partial(worker, session=session, outdir=args.out_dir),
        tup,
        max_workers=args.max_workers,
        chunksize=args.max_chunksize,
    )

    with open(f"{args.out_dir.parent}/resnet.scp", "w") as f:
        for line in ret:
            f.write(line)


if __name__ == "__main__":
    main()
