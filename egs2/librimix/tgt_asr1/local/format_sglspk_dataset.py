#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import logging
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(description="format target speaker dataset.")
    parser.add_argument(
        "--in_dir", type=Path, required=True, help="Directory of the input dataset"
    )
    parser.add_argument(
        "--out_dir", type=Path, required=True, help="Directory of the output dataset"
    )

    return parser


def main():
    args = get_parser().parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)

    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)

    # read wav.scp, text_spk1, text_spk2
    fin_wavscp = codecs.open(args.in_dir / "wav.scp", "r", "utf-8")
    fin_textspk1 = codecs.open(args.in_dir / "text_spk1", "r", "utf-8")
    fin_textspk2 = codecs.open(args.in_dir / "text_spk2", "r", "utf-8")

    # write wav.scp, text, utt2spk, spk2utt
    fout_wavscp = codecs.open(args.out_dir / "wav.scp", "w", "utf-8")
    fout_text = codecs.open(args.out_dir / "text", "w", "utf-8")
    fout_utt2spk = codecs.open(args.out_dir / "utt2spk", "w", "utf-8")
    fout_spk2utt = codecs.open(args.out_dir / "spk2utt", "w", "utf-8")

    for wavline, textline1, textline2 in zip(fin_wavscp, fin_textspk1, fin_textspk2):
        mixid, wavpath = wavline.strip().split(" ")
        text1 = textline1.strip().split(" ", 1)[1]
        text2 = textline2.strip().split(" ", 1)[1]

        # write wav.scp
        fout_wavscp.write(f"{mixid}_spk1 {wavpath}\n")
        fout_wavscp.write(f"{mixid}_spk2 {wavpath}\n")

        # write text
        fout_text.write(f"{mixid}_spk1 {text1}\n")
        fout_text.write(f"{mixid}_spk2 {text2}\n")

        # write utt2spk
        fout_utt2spk.write(f"{mixid}_spk1 {mixid}_spk1\n")
        fout_utt2spk.write(f"{mixid}_spk2 {mixid}_spk2\n")

        # write spk2utt
        fout_spk2utt.write(f"{mixid}_spk1 {mixid}_spk1\n")
        fout_spk2utt.write(f"{mixid}_spk2 {mixid}_spk2\n")

    fout_wavscp.close()
    fout_text.close()
    fout_utt2spk.close()
    fout_spk2utt.close()

    fin_wavscp.close()
    fin_textspk1.close()
    fin_textspk2.close()


if __name__ == "__main__":
    main()
