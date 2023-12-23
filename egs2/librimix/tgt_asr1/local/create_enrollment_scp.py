# Modified from egs2/librimix/tse1/local/prepare_librimix_enroll.py

import argparse
import json
import random
import logging
from pathlib import Path

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.utils.types import str2bool


def get_parser():
    parser = argparse.ArgumentParser(description="Create enroll scp files")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path of dataset.")
    parser.add_argument(
        "--is_train",
        type=str2bool,
        default=True,
        help="Whether is the training set or not",
    )
    parser.add_argument(
        "--mix2enroll",
        type=Path,
        default=None,
        help="Path to the downloaded map_mixture2enrollment file.",
    )
    parser.add_argument(
        "--enroll_prefix",
        type=str,
        default="enroll",
        help="Prefix of the generated enrollment scp files",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    return parser


def prepare_enroll_scp_train(data_dir, prefix="enroll"):
    mixtures = []
    with open(data_dir / "wav.scp", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            mix_id = line.strip().split(maxsplit=1)[0]
            mixtures.append(mix_id)

    with DatadirWriter(data_dir) as writer:
        for mix_id in mixtures:
            # 100-121669-0004_3180-138043-0053_spk1
            utts = mix_id.split("_")[:-1]
            index = int(mix_id.split("_")[-1][-1]) - 1
            utt_id = utts[index]
            spk_id = utt_id.split("-")[0]
            # For the training set, we choose the enrollment speech on the fly.
            # Thus, here we use the pattern f"*{utt_id} {spk_id}" to indicate it.
            writer[f"{prefix}.scp"][mix_id] = f"*{utt_id} {spk_id}"


def prepare_enroll_scp(data_dir, map_mix2enroll, prefix="enroll"):
    mixtures = []
    with open(data_dir / "wav.scp", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            mix_id = line.strip().split(maxsplit=1)[0]
            mixtures.append(mix_id)

    with open(data_dir / "spk2enroll.json", "r", encoding="utf-8") as f:
        # {spk_id:
        #   [[utt_id1, path1],
        #   [utt_id2, path2]],
        # }
        jsonstring = json.load(f)
    enroll2path = {}
    for spk_id, utt2paths in jsonstring.items():
        for item in utt2paths:
            enroll2path[item[0]] = item[1]

    mix2enroll = {}
    with open(map_mix2enroll) as f:
        for line in f:
            mix_id, utt_id, enroll = line.strip().split()
            spk_index = mix_id.split("_").index(utt_id) + 1
            index = int(enroll.split("/")[0][-1]) - 1
            enroll_id = enroll.split("/")[1].split("_")[index]
            mix2enroll[f"{mix_id}_spk{spk_index}"] = enroll_id

    with DatadirWriter(data_dir) as writer:
        for mix_id in mixtures:
            # 100-121669-0004_3180-138043-0053_spk1
            enroll_id = mix2enroll[mix_id]
            writer[f"{prefix}.scp"][mix_id] = enroll2path[enroll_id]


def main():
    args = get_parser().parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)

    if args.is_train:
        prepare_enroll_scp_train(args.data_dir, args.enroll_prefix)
    else:
        prepare_enroll_scp(args.data_dir, args.mix2enroll, args.enroll_prefix)


if __name__ == "__main__":
    main()
