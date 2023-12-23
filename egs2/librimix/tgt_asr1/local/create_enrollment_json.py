# Modified from egs2/librimix/tse1/local/prepare_spk2enroll_librispeech.py

import argparse
import json
from collections import defaultdict
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser("Create enrollment json files.")
    parser.add_argument(
        "--audio_path",
        type=Path,
        required=True,
        action="append",
        help="Path of audio files.",
    )
    parser.add_argument(
        "--enroll_json",
        type=Path,
        required=True,
        help="Path of the output spk2enroll json file",
    )
    parser.add_argument("--audio_format", type=str, default="flac")

    return parser


def get_spk2utt(paths, audio_format="flac"):
    spk2utt = defaultdict(list)
    for path in paths:
        for audio in path.rglob(f"*.{audio_format}"):
            spkid = audio.parent.parent.stem
            uid = audio.stem
            assert uid.split("-")[0] == spkid, audio
            spk2utt[spkid].append((uid, str(audio)))

    return spk2utt


def main():
    args = get_parser().parse_args()

    spk2utt = get_spk2utt(args.audio_path, audio_format=args.audio_format)
    args.enroll_json.parent.mkdir(parents=True, exist_ok=True)
    with args.enroll_json.open("w", encoding="utf-8") as f:
        json.dump(spk2utt, f, indent=4)


if __name__ == "__main__":
    main()
