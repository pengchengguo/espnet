#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=9
train_set=train_worn_simu_uall
dev_set=dev_worn_uall

num_data_reps=4
foreground_snrs="20:10:15:5:0"
background_snrs="20:10:15:5:0"

speaker_embed_root="/projects/tir5/users/xuankaic/experiments/espnet/egs2/chime6/sot_asr1/emb_chime6_ref_array/output_embeddings_better"

python=python3

log "$0 $*"
. utils/parse_options.sh

# chime6 data directories, which are generated from ${chime5_corpus},
# to synchronize audio files across arrays and modify the annotation (JSON) file accordingly
chime6_corpus=${PWD}/CHiME6
json_dir=${chime6_corpus}/transcriptions
audio_dir=${chime6_corpus}/audio
# enhanced_dir=enhanced


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


if [ ! -e "${CHIME5}" ]; then
    log "Fill the value of 'CHIME5' of db.sh"
    exit 1
fi

prepare_sot_data() {
    dataroot=$1
    dset=$2
    mictype=$3
    crop_length=$4
    datadir="${dataroot}/${dset}_${mictype}"

    echo "${dset}_${mictype}"

    utils/copy_data_dir.sh "${datadir}" "${datadir}"_uttsplit
    cp "${datadir}"/session2speaker "${datadir}"_uttsplit

    # uncombined_segments
    <"${datadir}"_uttsplit/segments awk '{
            split($1, lst, "_"); 
            speaker_id=lst[1];
            session_id=lst[2];
            print($1, speaker_id, session_id, $2, $3, $4);
        }' | sort -k 3,4 -k 5,5n > "${datadir}"/uncombined_segments

    _opts=
    _opts+="--crop_length ${crop_length} "
    if [ "${mictype}" = "ref" ]; then
        _opts+="--ref_array true --max_num_utts 8 "
    fi
    # Generate combined segments, text, utt2spk
    ${python} -u local/combine_chime6_segments.py \
        --uncombined_segments "${datadir}"/uncombined_segments \
        --uncombined_text "${datadir}"_uttsplit/text \
        --odir "${datadir}" ${_opts} 2>&1 | tee "${datadir}"/data_info

    if [ "${dset}_${mictype}" = "train_worn_rvb" ]; then
        #TODO(simpleoier): is it necessary to prepare special speaker embeddings for train_worn_rvb.
        log "Modifying the speaker id for ${dset}_${mictype} to be the same as the original train_worn."
        sed -i 's/rev[0-4]-//2' "${datadir}"/speakers
    fi

    # spk2utt
    utils/utt2spk_to_spk2utt.pl "${datadir}"/utt2spk > "${datadir}"/spk2utt

    # utt2speaker_idx
    awk '(FILENAME==ARGV[1]) {
            a[$1]=$2;
        } (FILENAME==ARGV[2]) {
            out=$1;
            match($1, /S[0-9]+/);
            session_id=substr($1, RSTART, RLENGTH);
            n=split(a[session_id], lst, ",");
            for (i=1; i<=n; i++) {tmp[lst[i]] = i-1};
            out=out" "a[$1];
            n=split($2, lst, ",");
            for (i=1; i<=n; i++) {out=out" "tmp[lst[i]]};
            print(out);
        }' "${datadir}"/session2speaker "${datadir}"/speakers \
            > "${datadir}"/utt2speaker_idx

    # multi-sc text
    awk '(FILENAME==ARGV[1]) {
            a[$1]=$0
        } (FILENAME==ARGV[2]) {
            split(a[$1], lst1, " ");
            idx=2;
            sc_idx=lst1[idx]+1;
            for (i=2; i<=NF; i++) {if ($i == "<sc>") {
                $i="<sc"sc_idx">"; idx+=1; sc_idx=lst1[idx]+1;}
            }
            print($0)
        }' "${datadir}"/utt2speaker_idx "${datadir}"/text \
            > "${datadir}"/text.sc_multi

    # speaker_embeddings ark & scp
    mkdir -p "${dataroot}/speaker_embedding/${dset}_${mictype}"
    ${python} local/dump_speaker_embeddings.py \
        --speaker_embed_dir "${speaker_embed_root}/${dset}" \
        --session2speaker "${datadir}"/session2speaker \
        --utt2speakers "${datadir}"/speakers \
        --out_filetype "mat" \
        "ark,scp:${dataroot}/speaker_embedding/${dset}_${mictype}/speaker_embeddings.ark,${datadir}/speakers.scp"

    utils/fix_data_dir.sh --utt_extra_files "utt2speaker_idx speakers.scp" ${datadir}

}

###########################################################################
# We first generate the synchronized audio files across arrays and
# corresponding JSON files. Note that this requires sox v14.4.2,
# which is installed via miniconda in ./local/check_tools.sh
###########################################################################


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data processing"

    log "Perform synchronized CHiME6"
    local/generate_chime6_data.sh \
        --cmd "$train_cmd" \
        ${CHIME5} \
        ${chime6_corpus}
fi


if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 1: Data processing"
    echo "$0:  prepare data..."

    for dataset in train; do
        # skip u03 and u04 as they are missing
        for mictype in worn u01 u02 u05 u06; do
            local/prepare_data.sh --mictype ${mictype} \
                ${audio_dir}/${dataset} ${json_dir}/${dataset} data/${dataset}_${mictype}
        done
    done

    for dataset in dev eval; do
        for mictype in worn u01 u02 ref; do
            local/prepare_data.sh --mictype ${mictype} \
                ${audio_dir}/${dataset} ${json_dir}/${dataset} \
                data/${dataset}_${mictype}
            #TODO(simpleoier): some errors when processing for ref array
            # Manual solutions: to rename the wav.scp, remove 3 out of CH[1-4] channels,
            # and rename the left one to be "SESSION_ID.ENH"
        done
    done
fi


#########################################################################################
# In stages 3 to 6, we augment and fix train data for our training purpose. point source
# noises are extracted from chime corpus. Here we use 400k utterances from array microphones,
# its augmentation and all the worn set utterances in train.
#########################################################################################

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: remove bad sessions"
    # remove possibly bad sessions (P11_S03, P52_S19, P53_S24, P54_S24)
    # see http://spandh.dcs.shef.ac.uk/chime_challenge/data.html for more details
    utils/copy_data_dir.sh data/train_worn data/train_worn_org # back up
    grep -v -e "^P11_S03" -e "^P52_S19" -e "^P53_S24" -e "^P54_S24" data/train_worn_org/text > data/train_worn/text
    utils/fix_data_dir.sh data/train_worn
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: generate reverberations for train worn microphone"

    local/extract_noises.py $chime6_corpus/audio/train $chime6_corpus/transcriptions/train \
        local/distant_audio_list distant_noises
    local/make_noise_list.py distant_noises > distant_noise_list

    noise_list=distant_noise_list

    if [ ! -d RIRS_NOISES/ ]; then
        # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
        wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
        unzip rirs_noises.zip
    fi

    # This is the config for the system using simulated RIRs and point-source noises
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
    rvb_opts+=(--noise-set-parameters $noise_list)

    steps/data/reverberate_data_dir.py \
        "${rvb_opts[@]}" \
        --prefix "rev" \
        --foreground-snrs $foreground_snrs \
        --background-snrs $background_snrs \
        --speech-rvb-probability 1 \
        --pointsource-noise-addition-probability 1 \
        --isotropic-noise-addition-probability 1 \
        --num-replications $num_data_reps \
        --max-noises-per-minute 1 \
        --source-sampling-rate 16000 \
        data/train_worn data/train_worn_rvb
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: dump session to speaker information"

    dset="train"
    for mictype in "worn" "worn_rvb" "u01" "u02" "u05" "u06"; do
        cat <<EOF > data/"${dset}_${mictype}"/session2speaker
S03 P09,P10,P11,P12
S04 P09,P10,P11,P12
S05 P13,P14,P15,P16
S06 P13,P14,P15,P16
S07 P17,P18,P19,P20
S17 P17,P18,P19,P20
S08 P21,P22,P23,P24
S16 P21,P22,P23,P24
S12 P33,P34,P35,P36
S13 P33,P34,P35,P36
S19 P49,P50,P51,P52
S20 P49,P50,P51,P52
S18 P41,P42,P43,P44
S22 P41,P42,P43,P44
S23 P53,P54,P55,P56
S24 P53,P54,P55,P56
EOF

    done

    for dset in "dev" "eval"; do
        for mictype in "worn" "u01" "u02" "ref"; do

            if [ "${dset}" = "dev" ]; then
                cat <<EOF > data/"${dset}_${mictype}"/session2speaker
S02 P05,P06,P07,P08
S09 P25,P26,P27,P28
EOF
            else
                cat <<EOF > data/"${dset}_${mictype}"/session2speaker
S01 P01,P02,P03,P04
S21 P45,P46,P47,P48
EOF
            fi

        done
    done

fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: reorder input and combine the overlapped input."

    # Train
    dset="train"
    # for mictype in "worn" "worn_rvb" "u01" "u02" "u05" "u06"; do
    for mictype in "u01" "u02" "u05" "u06"; do
        prepare_sot_data "data" ${dset} ${mictype} 20
    done

    # Dev
    for dset in dev eval; do
        for mictype in "worn" "u01" "u02" "ref"; do
            prepare_sot_data "data" ${dset} ${mictype} 10
        done

    done

fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: combine mix array and worn mics."

    utils/combine_data.sh --extra_files "utt2speaker_idx speakers.scp" data/train_uall data/train_u01 data/train_u02 data/train_u05 data/train_u06
    utils/combine_data.sh --extra_files "utt2speaker_idx speakers.scp" data/${train_set} data/train_worn data/train_worn_rvb data/train_uall

    # only use left channel for worn mic recognition
    # you can use both left and right channels for training
    for dset in dev; do
        utils/copy_data_dir.sh data/${dset}_worn data/${dset}_worn_stereo
        grep "\.L-" data/${dset}_worn_stereo/text > data/${dset}_worn/text
        utils/fix_data_dir.sh --utt_extra_files "utt2speaker_idx speakers.scp" data/${dset}_worn
    done
    utils/combine_data.sh --extra_files "utt2speaker_idx speakers.scp" data/${dev_set} data/dev_worn data/dev_u01 data/dev_u02
fi

# if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
#     log "Stage 8: Split speakers up into 3-minute chunks."
#     # This doesn't hurt adaptation, and
#     # lets us use more jobs for decoding etc.
#     for dset in ${train_set}; do
#         utils/copy_data_dir.sh data/${dset} data/${dset}_nosplit
#         utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}_nosplit data/${dset}
#     done
# fi


nlsyms=data/nlsyms.txt

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    log "stage 9: Create non linguistic symbols: ${nlsyms}"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "\[" > ${nlsyms}
    cat ${nlsyms}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
