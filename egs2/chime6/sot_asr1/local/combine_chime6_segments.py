#!/usr/bin/python

import argparse
import datetime
import json
import logging
import os
import pdb
import re
from typing import Dict, List, Tuple

import numpy as np

from espnet2.utils.types import str2bool

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger("combine_chime6_segments")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uncombined_segments",
        type=str,
        default=None,
        required=True,
        help="utterance-based segments for CHiME6, but sorted in the order of speaker / session_mic / start time using '<segments awk '{split($1, lst, \"_\"); speaker_id=lst[1]; session_id=lst[2]; print($1, speaker_id, session_id, $2, $3, $4)}' | sort -k 3,4 -k 5,5n '.",
    )
    parser.add_argument(
        "--uncombined_text",
        type=str,
        default=None,
        required=True,
        help="utterance-based text for CHiME6",
    )
    parser.add_argument(
        "--odir", type=str, default=None, required=True, help="output directory."
    )
    parser.add_argument(
        "--speaker_change_symbol",
        type=str,
        default="<sc>",
        help="speaker change symbol.",
    )
    parser.add_argument(
        "--ref_array",
        type=str2bool,
        default=False,
        help="whether to process the reference array data."
    )
    parser.add_argument(
        "--crop_length",
        type=float,
        default=None,
        help="crop the segment if the crop_length (sec.) is satisfied. If None is specified, no crop is applied.",
    )
    parser.add_argument(
        "--max_num_utts",
        type=int,
        default=None,
        help="crop the segment if the number of utterances in the segments larger than max_num_utts.",
    )

    return parser


def generate_segments_text_utt2spk_speakers_files(
    odir: str,
    combined_data: List[Dict],
    speaker_change_symbol: str = "<sc>",
    ref_array: bool=False,
):
    if ref_array:
        replace_pattern = re.compile(r"U0[1-6]")

    with open(os.path.join(odir, "segments"), "w") as segment_f, open(
        os.path.join(odir, "text"), "w"
    ) as text_f, open(os.path.join(odir, "utt2spk"), "w") as utt2spk_f, open(
        os.path.join(odir, "speakers"), "w"
    ) as speakers_f:
        for segment in combined_data:
            if ref_array:
                re.sub(replace_pattern, segment["ref"], segment["session_mic_ch_id"])

            segment_id = segment["session_mic_ch_id"].replace(".", "_NOLOCATION.")
            segment_id += f"-{int(segment['start_time']*100):07}-{int(segment['end_time']*100):07}"

            all_speakers = ",".join([utt["speaker"] for utt in segment["utts"]])
            speakers_f.write(f"{segment_id} {all_speakers}\n")

            segment_f.write(
                f"{segment_id} {segment['session_mic_ch_id']} {segment['start_time']:08.2f} {segment['end_time']:08.2f}\n"
            )
            utt2spk_f.write(f"{segment_id} {segment_id}\n")

            all_text = ""
            for utt in segment["utts"]:
                all_text += f"{speaker_change_symbol} {utt['words']} "
            text_f.write(f"{segment_id} {all_text}\n")


def format_time_str(time: datetime.timedelta):
    return f"{time.seconds // 3600:02}:{time.seconds % 3600 // 60:02}:{time.seconds % 60:02}.{int(time.microseconds/10000):02}"


def str_to_second(time_str: str):
    lst = time_str.split(":")
    return 3600 * float(lst[0]) + 60 * float(lst[1]) + float(lst[2])


def update_segment_info(
    existing_segment_info: List[Tuple],
    utt: Dict,
):
    def find_start_index(start_second: float, l: int, r: int):
        if (r - l) == 0:
            return None
        elif (r - l) == 1:
            if round(start_second * 100) >= round(
                existing_segment_info[l][0] * 100
            ) and round(start_second * 100) <= round(existing_segment_info[l][1] * 100):
                return l
            else:
                return None
        middle_pt = (l + r) // 2
        l_ret = find_start_index(start_second, l, middle_pt)
        if l_ret is not None:
            return l_ret
        r_ret = find_start_index(start_second, middle_pt, r)
        if r_ret is not None:
            return r_ret
        return None

    def find_end_index(end_second: float, l: int, r: int):
        if (r - l) == 0:
            return None
        elif (r - l) == 1:
            if round(end_second * 100) >= round(
                existing_segment_info[l][0] * 100
            ) and round(end_second * 100) <= round(existing_segment_info[l][1] * 100):
                return l
            else:
                return None
        middle_pt = (l + r) // 2
        l_ret = find_end_index(end_second, l, middle_pt)
        if l_ret is not None:
            return l_ret
        r_ret = find_end_index(end_second, middle_pt, r)
        if r_ret is not None:
            return r_ret
        return None

    start_idx = find_start_index(
        str_to_second(utt["start_time"]), 0, len(existing_segment_info)
    )
    if start_idx is None:
        pdb.set_trace()
        start_idx = find_start_index(
            str_to_second(utt["start_time"]), 0, len(existing_segment_info)
        )
        raise ValueError("start_idx is None.")
    end_idx = find_end_index(
        str_to_second(utt["end_time"]), start_idx, len(existing_segment_info)
    )
    if end_idx is None:
        pdb.set_trace()
        end_idx = find_end_index(
            str_to_second(utt["end_time"]), start_idx, len(existing_segment_info)
        )
        raise ValueError("end_idx is None.")
    tmp_segment = []
    for idx in range(start_idx, end_idx + 1):
        seg_to_be_split = existing_segment_info[idx]

        s_t = seg_to_be_split[0]
        utt_s_t = str_to_second(utt["start_time"])
        utt_e_t = str_to_second(utt["end_time"])

        if round(utt_s_t * 100) > round(s_t * 100):
            tmp_segment.append((s_t, utt_s_t, seg_to_be_split[2].copy()))
            s_t = utt_s_t
        if utt_e_t <= seg_to_be_split[1]:
            spkr_set = seg_to_be_split[2].copy()
            spkr_set.add(utt["speaker"])
            tmp_segment.append((s_t, utt_e_t, spkr_set))
            s_t = utt_e_t
            if round(s_t * 100) < round(seg_to_be_split[1] * 100):
                tmp_segment.append(
                    (utt_e_t, seg_to_be_split[1], seg_to_be_split[2].copy())
                )
        else:
            spkr_set = seg_to_be_split[2].copy()
            spkr_set.add(utt["speaker"])
            tmp_segment.append((s_t, seg_to_be_split[1], spkr_set))

        if len(tmp_segment[-1][2]) > 4:
            pdb.set_trace()
            raise ValueError("speaker cnt is greater than 4.")

    new_segments = []
    if start_idx > 0:
        new_segments += existing_segment_info[0:start_idx]
    new_segments += tmp_segment
    if end_idx < len(existing_segment_info):
        new_segments += existing_segment_info[end_idx + 1 :]

    assert round(new_segments[0][0] * 100) == round(existing_segment_info[0][0] * 100)
    assert round(new_segments[-1][1] * 100) == round(
        existing_segment_info[-1][1] * 100
    ), f"{new_segments[-1][1]} vs. {existing_segment_info[-1][1]}"

    # Merge consecutive segments if their speaker cnt is the same
    ret = [new_segments[0]]
    for i in range(1, len(new_segments)):
        if round(ret[-1][1] * 100) != round(new_segments[i][0] * 100):
            pdb.set_trace()
        assert round(ret[-1][1] * 100) == round(
            new_segments[i][0] * 100
        ), f"{ret[-1][1]} vs. {new_segments[i][0]}"
        if (ret[-1][2] == new_segments[i][2]) or (
            ret[-1][1] + 0.2 >= new_segments[i][1]
        ):  # gap less than 0.2s are not considered
            ret[-1] = (ret[-1][0], new_segments[i][1], ret[-1][2])
        else:
            ret.append(new_segments[i])

    assert round(ret[0][0] * 100) == round(existing_segment_info[0][0] * 100)
    assert round(ret[-1][1] * 100) == round(existing_segment_info[-1][1] * 100)

    return ret


def compute_overlap_length(data: List[Dict]):
    segment_s_e_speaker_cnt = [
        (data["start_time"], data["end_time"], set([]))
    ]  # list of segment information: (start, end, speaker_cnt), speaker is initialized as 0
    for utt in data["utts"]:
        # print("start:", segment_s_e_speaker_cnt, utt)
        segment_s_e_speaker_cnt = update_segment_info(segment_s_e_speaker_cnt, utt)
        # print("end:", segment_s_e_speaker_cnt)

    overlap_accum = np.array([0, 0, 0, 0])
    for segment in segment_s_e_speaker_cnt:
        if len(segment[2]) > 4:
            pdb.set_trace()
        overlap_accum[len(segment[2]) - 1] += segment[1] - segment[0]
    return overlap_accum


def main(args):
    assert os.path.exists(args.uncombined_segments) and os.path.exists(
        args.uncombined_text
    ), f"{args.uncombined_segments} or {args.uncombined_text} do not exist."

    # read text
    logger.info(f"Reading {args.uncombined_text}")
    utt2text = dict()
    num_text = 0
    with open(args.uncombined_text, "r") as f:
        for line in f.readlines():
            num_text += 1
            line = line.strip()
            lst = line.split(" ", maxsplit=1)
            if len(lst) == 1:
                lst.append("")
            try:
                utt2text[lst[0]] = lst[1]
            except Exception as e:
                print(f"{num_text} {lst}")
                raise e

    def initialize_data(line):
        lst = line.split(" ")
        assert (
            len(lst) == 6
        ), f"number of fields of line: {line} does not equal to 5. Please check data or the script."

        start_time, end_time, session_mic_ch_id, speaker, utt = (
            float(lst[4]),
            float(lst[5]),
            lst[3],
            lst[1],
            lst[0],
        )
        return dict(
            start_time=start_time,
            end_time=end_time,
            session_mic_ch_id=session_mic_ch_id,
            speakers=[speaker],
            utts=[
                dict(
                    start_time=format_time_str(
                        datetime.timedelta(seconds=start_time)
                    ),  # chime6 transcription json uses the format of "00:01:00.39"
                    end_time=format_time_str(datetime.timedelta(seconds=end_time)),
                    words=utt2text[utt],
                    speaker=speaker,
                    session_id=lst[2],
                    utt_id=utt,
                )
            ],
        )

    # read segments and combine them
    logger.info(f"Reading {args.uncombined_segments} and combining.")
    combined_data = []
    num_utts = 0
    nutt_longer_than_30s = 0
    max_duration = 0
    max_nutt = 0
    total_duration = 0
    overlap_duration = np.array([0, 0, 0, 0])
    with open(args.uncombined_segments, "r") as f:
        line = f.readline().strip()
        tmp_data = initialize_data(line)
        num_uncombined_segments = 1

        if args.ref_array:
            ref_array_vote = np.array([0, 0, 0, 0, 0, 0])  # the votes for array U01 - U06
            ref_array_pattern = re.compile(r"U0[1-6]")
            ref_array = int(re.search(ref_array_pattern, line.split()[3]).group(0)[-1])
            ref_array_vote[ref_array - 1] += 1

        for line in f.readlines():
            num_uncombined_segments += 1
            line = line.strip()
            # (utt_name, speaker_id, session_id, sessin_mic_ch, start_time, end_time)
            # e.g. P12_S03_U01_NOLOCATION.CH1-0005755-0006039 P12 S03 S03_U01.CH1 00057.55 00060.39
            lst = line.split(" ")
            assert (
                len(lst) == 6
            ), f"number of fields of line: {line} does not equal to 5. Please check data or the script."
            start_time = float(lst[4])
            end_time = float(lst[5])
            if (
                round(start_time * 100) >= round(tmp_data["end_time"] * 100)  # non-overlap start
                or lst[3] != tmp_data["session_mic_ch_id"]  # session changed -> no overlap
                or (
                    args.crop_length is not None
                    and (tmp_data["end_time"] - tmp_data["start_time"]) > args.crop_length
                )  # crop_length satisfied.
                or (
                    args.max_num_utts is not None
                    and len(tmp_data["utts"]) > args.max_num_utts
                )  # crop_length satisfied.
            ):  # Start a new utterance
                if args.ref_array:
                    tmp_data["ref"] = f"U0{np.argmax(ref_array_vote) + 1}"
                    ref_array_vote.fill(0)

                combined_data.append(tmp_data)

                # collect utterance related statistics
                num_utts += len(combined_data[-1]["utts"])
                max_duration = max(
                    max_duration,
                    combined_data[-1]["end_time"] - combined_data[-1]["start_time"],
                )
                max_nutt = max(max_nutt, len(combined_data[-1]["utts"]))
                total_duration += (
                    combined_data[-1]["end_time"] - combined_data[-1]["start_time"]
                )
                overlap_duration += compute_overlap_length(combined_data[-1])
                nutt_longer_than_30s += (
                    1
                    if combined_data[-1]["end_time"] - combined_data[-1]["start_time"]
                    > 30
                    else 0
                )

                # Start a new utterance
                tmp_data = initialize_data(line)
            else:  # extend an existing utterance
                if end_time > tmp_data["end_time"]:
                    tmp_data["end_time"] = end_time
                if lst[1] not in tmp_data["speakers"]:
                    tmp_data["speakers"].append(lst[1])
                tmp_data["utts"].append(
                    dict(
                        start_time=format_time_str(
                            datetime.timedelta(seconds=start_time)
                        ),  # chime6 transcription json uses the format of "00:01:00.39"
                        end_time=format_time_str(datetime.timedelta(seconds=end_time)),
                        words=utt2text[lst[0]],
                        speaker=lst[1],
                        session_id=lst[2],
                        utt_id=lst[0],
                    )
                )
            if args.ref_array:
                # collect the reference array voting information
                ref_array = int(re.search(ref_array_pattern, lst[3]).group(0)[-1])
                ref_array_vote[ref_array - 1] += 1

        if len(tmp_data["utts"]) > 0:
            combined_data.append(tmp_data)
            num_utts += len(tmp_data["utts"])
            max_duration = max(
                max_duration,
                combined_data[-1]["end_time"] - combined_data[-1]["start_time"],
            )
            max_nutt = max(max_nutt, len(combined_data[-1]["utts"]))
            total_duration += (
                combined_data[-1]["end_time"] - combined_data[-1]["start_time"]
            )
            overlap_duration += compute_overlap_length(combined_data[-1])
            nutt_longer_than_30s += (
                1
                if combined_data[-1]["end_time"] - combined_data[-1]["start_time"] > 30
                else 0
            )
            if args.ref_array:
                tmp_data["ref"] = f"U0{np.argmax(ref_array_vote) + 1}"
                ref_array_vote.fill(0)

    assert (
        num_text == num_uncombined_segments
    ), f"{num_text} != {num_uncombined_segments}"
    assert (
        num_utts == num_uncombined_segments
    ), f"{num_utts} != {num_uncombined_segments}"

    logger.info(
        f"Time accumulation for 1-4 speakers overlap, and total time: \n\t{overlap_duration}, {np.sum(overlap_duration)}"
    )
    logger.info(
        f"Ratio for 1-4 speakers overlap, and total time: \n\t{overlap_duration / np.sum(overlap_duration)}"
    )
    # output data
    logger.info(f"Max number of utterances in a segment:\n\t{max_nutt}")
    logger.info(
        f"Max utterance lengths:\n\t{format_time_str(datetime.timedelta(seconds=max_duration))}"
    )
    logger.info(
        f"Number of segments longer than 30 sec. vs total number of segments.\n\t{nutt_longer_than_30s}\t{len(combined_data)}"
    )
    output_file = os.path.join(args.odir, "combined_segments.json")
    logger.info(f"Dumping combined json to to {output_file}")
    jsonstring = json.dumps(
        combined_data,
        indent=4,
        sort_keys=False,
        ensure_ascii=False,
        separators=(",", ": "),
    )
    with open(output_file, "w") as f:
        f.write(jsonstring)

    generate_segments_text_utt2spk_speakers_files(
        args.odir, combined_data, speaker_change_symbol=args.speaker_change_symbol, ref_array=args.ref_array,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)
