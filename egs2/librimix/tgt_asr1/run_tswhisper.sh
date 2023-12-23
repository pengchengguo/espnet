#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_100_sglspk"
valid_set="dev_sglspk"
test_sets="dev_sglspk_org test_sglspk"

expdir=exp/tswhisper

ngpu=2
device="0,1"

asr_config=conf/tswhisper/train_tsasr_whisper_medium_full_con20_q16_l2_crop10_lr5e-5.yaml

inference_config=conf/tswhisper/decode_asr_whisper_beam1.yaml
inference_args="--tgtspk_infer True"

CUDA_VISIBLE_DEVICES=${device} ./asr_my.sh \
    --ngpu ${ngpu} \
    --expdir ${expdir} \
    --nj 32 \
    --gpu_inference true \
    --inference_nj 1 \
    --lang en \
    --tgtspk_asr true \
    --enroll_prefix "enroll" \
    --enroll_type "text" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --feats_normalize "" \
    --token_type whisper_multilingual \
    --max_wav_duration 30 \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_args "${inference_args}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
