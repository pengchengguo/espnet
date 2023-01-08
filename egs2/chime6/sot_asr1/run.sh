#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail



train_set=train_worn_simu_uall
valid_set=dev_ref
test_sets="dev_ref"


opts=""
asr_config="conf/tuning/train_asr_transformer_wavlm_lr1e-4_specaug_accum1_preenc128_warmup20k_speaker_emb_concat_feats_enc_rnn_dec.yaml"; opts=${opts:-}" --feats_normalize utt_mvn --sot_asr true --speaker_inventory true --speaker_inventory_size 4 --text_suf sc_multi"
inference_config="conf/decode_asr_transformer.yaml"
lm_config="conf/train_lm.yaml"

bpe_nlsyms="[inaudible],[laughs],[noise]"

use_lm=false
use_word_lm=false
word_vocab_size=65000

./sot_asr.sh \
    --lang en \
    --asr_args "--only_collect_shape_info true --allow_variable_data_keys true" \
    --inference_args "--allow_variable_data_keys true " \
    --token_type char \
    --nbpe 1000 \
    --bpe_nlsyms "${bpe_nlsyms}" \
    --nlsyms_txt "data/nlsyms.txt" \
    --feats_type raw \
    --audio_format "flac" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --use_lm ${use_lm} \
    --lm_config "${lm_config}" \
    --use_word_lm ${use_word_lm} \
    --word_vocab_size ${word_vocab_size} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" ${opts} "$@"
