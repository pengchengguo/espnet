#!/usr/bin/python

import argparse
import glob
import numpy as np
import os
import re

from espnet.utils.cli_writers import file_writer_helper


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--speaker_embed_dir",
        type=str,
        default=None,
        required=True,
        help="speaker embedding dir",
    )
    parser.add_argument(
        "--session2speaker",
        type=str,
        default=None,
        required=True,
        help="session to speaker file",
    )
    parser.add_argument(
        "--utt2speakers",
        type=str,
        default=None,
        required=True,
        help="utt2speakers file.",
    )
    parser.add_argument(
        "--out_filetype",
        type=str,
        default="mat",
        choices=["mat", "hdf5", "sound.hdf5", "sound"],
        help="Specify the file format for the wspecifier. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "wspecifier",
        type=str,
        default=None,
        help="write specifier for speaker embedding dump",
    )

    return parser


def random_choose(embed: np.ndarray, num_frames=10000):
    if len(embed.shape) == 1:
        return embed
    elif len(embed.shape) == 2:
        start = max(0, embed.shape[0] - num_frames)
        return np.mean(
            embed[start : min(embed.shape[0], start + num_frames), :], axis=0
        )


def main(args):
    assert os.path.isdir(args.speaker_embed_dir)
    assert os.path.exists(args.session2speaker)

    speaker_embed = dict()
    for f in glob.glob(f"{args.speaker_embed_dir}/*"):
        try:
            emb = np.load(f)
            assert len(emb.shape) <= 2
            f_name = os.path.basename(f)
            speaker_embed[os.path.splitext(f_name)[0]] = emb

        except Exception as e:
            raise e

    session2speaker = dict()
    with open(args.session2speaker, "r") as f:
        for line in f.readlines():
            session_id, speaker_ids = line.strip().split(" ")
            assert (
                session_id not in session2speaker
            ), f"{session_id} duplicated. Please check {args.session2speaker} file."
            session2speaker[session_id] = speaker_ids.split(",")

    re_pat = re.compile(r'S[0-9]+')
    utt_speaker_embeddings = dict()
    with open(args.utt2speakers, "r") as f:
        for line in f.readlines():
            line = line.strip()
            lst = line.split(" ")
            session = re_pat.search(lst[0]).group(0)
            speakers = lst[1].split(",")

            try:
                assert set(speakers) <= set(session2speaker[session])
            except Exception as e:
                import pdb; pdb.set_trace()
                raise e

            tmp_speaker_emb = []
            for spkr in session2speaker[session]:
                tmp_speaker_emb.append(random_choose(speaker_embed[spkr]))

            utt_speaker_embeddings[lst[0]] = tmp_speaker_emb

    with file_writer_helper(
        args.wspecifier,
        filetype=args.out_filetype,
        write_num_frames=None,
        compress=False,
    ) as writer:
        for utt, speaker_emb in utt_speaker_embeddings.items():
            writer[utt] = np.stack(speaker_emb, axis=0)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)
