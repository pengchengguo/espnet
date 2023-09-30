#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
skip_upload_hf=true  # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Kmeans realted
kmeans_opts=                    # The options given to scripts/feats/perform_kmeans.sh
kmeans_feature="wavlm_large/21" # format: ssl_model_type/layer_idx (e.g. mfcc, hubert_large/21, wavlm_large/21)
portion=0.1
nclusters=2000                  # The number of clusters for discrete tokenss
storage_save_mode=true          # Save storage on SSL feature extraction
                                # If true, feature extraction and kmeans clustering on the fly

# Tokenization related
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
token_joint=false       # whether to use a single bpe system for both source and target languages
src_ctc_case=lc.rm
src_ctc_token_type=bpe      # Tokenization type (char or bpe) for source languages.
src_ctc_nbpe=30             # The number of BPE vocabulary for source language.
src_ctc_bpemode=unigram     # Mode of BPE for source language (unigram or bpe).
src_ctc_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for source language.
src_ctc_bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE of source language
src_ctc_bpe_char_cover=1.0  # character coverage when modeling BPE for source language
src_case=ts
src_token_type=bpe      # Tokenization type (char or bpe) for source languages.
src_nbpe=30             # The number of BPE vocabulary for source language.
src_bpemode=unigram     # Mode of BPE for source language (unigram or bpe).
src_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for source language.
src_bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE of source language
src_bpe_char_cover=1.0  # character coverage when modeling BPE for source language
tgt_case=tc
tgt_token_type=bpe      # Tokenization type (char or bpe) for target language.
tgt_nbpe=30             # The number of BPE vocabulary for target language.
tgt_bpemode=unigram     # Mode of BPE (unigram or bpe) for target language.
tgt_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for target language.
tgt_bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE for target language.
tgt_bpe_char_cover=1.0  # character coverage when modeling BPE for target language.
hugging_face_model_name_or_path="" # Hugging Face model or path for hugging_face tokenizer

# Ngram model related
use_ngram=false
ngram_exp=
ngram_num=3
use_src_ngram=false

# Language model related
use_lm=false      # Use language model for ST decoding.
use_src_lm=false  # Use language model for ASR multi-decoder decoding.
lm_tag=           # Suffix to the result dir for language model training.
lm_exp=           # Specify the directory path for LM experiment.
                  # If this option is specified, lm_tag is ignored.
src_lm_exp=       # Specify the directory path for LM experiment.
lm_stats_dir=     # Specify the directory path for LM statistics.
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".
                  # Note that it will overwrite args in lm config.
use_word_lm=false # Whether to use word language model.
use_src_word_lm=false # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus.
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# ST model related
st_tag=        # Suffix to the result dir for st model training.
st_exp=        # Specify the directory path for ST experiment.
               # If this option is specified, st_tag is ignored.
st_stats_dir=  # Specify the directory path for ST statistics.
st_config=     # Config for st model training.
st_args=       # Arguments for st model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in st config.
pretrained_asr=                 # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch
feats_normalize=global_mvn      # Normalizaton layer type.
num_splits_st=1                 # Number of splitting for lm corpus.
src_ctc_lang=es                 # ctc source language abbrev. id (e.g., wavlm)
src_lang=wavlm_large_21_km2000  # discreate source language abbrev. id (e.g., es)
tgt_lang=en                     # target language abbrev. id (e.g., en)
use_src_ctc_lang=true           # Incorporate ASR loss (use src texts) or not

# Upload model related
hf_repo=

# Decoding related
use_k2=false      # Whether to use k2 based decoder
use_streaming=false # Whether to use streaming decoding
batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_lm=valid.loss.ave.pth       # Language model path for decoding.
inference_asr_lm=valid.loss.ave.pth       # Language model path for decoding.
inference_ngram=${ngram_num}gram.bin
inference_st_model=valid.acc.ave.pth # ST model path for decoding.
                                      # e.g.
                                      # inference_st_model=train.loss.best.pth
                                      # inference_st_model=3epoch.pth
                                      # inference_st_model=valid.acc.best.pth
                                      # inference_st_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
src_ctc_bpe_train_text= # Text file path of bpe training set for ctc source language 
src_bpe_train_text=  # Text file path of bpe training set for discrete source language.
tgt_bpe_train_text=  # Text file path of bpe training set for target language.
lm_train_text=   # Text file path of language model training set.
lm_dev_text=     # Text file path of language model development set.
lm_test_text=    # Text file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
st_speech_fold_length=800 # fold_length for speech data during ST training.
st_text_fold_length=150   # fold_length for text data during ST training.
lm_fold_length=150         # fold_length for LM training.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --token_joint=false       # Whether to use a single bpe system for both source and target languages.
                              # if set as true, will use tgt_* for processing (default="${token_joint}").
    --src_token_type=bpe      # Tokenization type (char or bpe) for source languages. (default="${src_token_type}").
    --src_nbpe=30             # The number of BPE vocabulary for source language. (default="${src_nbpe}").
    --src_bpemode=unigram     # Mode of BPE for source language (unigram or bpe). (default="${src_bpemode}").
    --src_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for source language. (default="${src_bpe_input_sentence_size}").
    --src_bpe_nlsyms=         # Non-linguistic symbols list, separated by a comma, for BPE of source language. (default="${src_bpe_nlsyms}").
    --src_bpe_char_cover=1.0  # Character coverage when modeling BPE for source language. (default="${src_bpe_char_cover}").
    --tgt_token_type=bpe      # Tokenization type (char or bpe) for target language. (default="${tgt_token_type}").
    --tgt_nbpe=30             # The number of BPE vocabulary for target language. (default="${tgt_nbpe}").
    --tgt_bpemode=unigram     # Mode of BPE (unigram or bpe) for target language. (default="${tgt_bpemode}").
    --tgt_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for target language. (default="${tgt_bpe_input_sentence_size}").
    --tgt_bpe_nlsyms=         # Non-linguistic symbols list, separated by a comma, for BPE for target language. (default="${tgt_bpe_nlsyms}").
    --tgt_bpe_char_cover=1.0  # Character coverage when modeling BPE for target language. (default="${tgt_bpe_char_cover}").

    # Language model related
    --lm_tag          # Suffix to the result dir for language model training (default="${lm_tag}").
    --lm_exp          # Specify the directory path for LM experiment.
                      # If this option is specified, lm_tag is ignored (default="${lm_exp}").
    --lm_stats_dir    # Specify the directory path for LM statistics (default="${lm_stats_dir}").
    --lm_config       # Config for language model training (default="${lm_config}").
    --lm_args         # Arguments for language model training (default="${lm_args}").
                      # e.g., --lm_args "--max_epoch 10"
                      # Note that it will overwrite args in lm config.
    --use_word_lm     # Whether to use word language model (default="${use_word_lm}").
    --word_vocab_size # Size of word vocabulary (default="${word_vocab_size}").
    --num_splits_lm   # Number of splitting for lm corpus (default="${num_splits_lm}").

    # ST model related
    --st_tag           # Suffix to the result dir for st model training (default="${st_tag}").
    --st_exp           # Specify the directory path for ST experiment.
                       # If this option is specified, st_tag is ignored (default="${st_exp}").
    --st_stats_dir     # Specify the directory path for ST statistics (default="${st_stats_dir}").
    --st_config        # Config for st model training (default="${st_config}").
    --st_args          # Arguments for st model training (default="${st_args}").
                       # e.g., --st_args "--max_epoch 10"
                       # Note that it will overwrite args in st config.
    --pretrained_asr=          # Pretrained model to load (default="${pretrained_asr}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --feats_normalize  # Normalizaton layer type. (default="${feats_normalize}").
    --num_splits_st    # Number of splitting for lm corpus.  (default="${num_splits_st}").
    --src_lang=        # source language abbrev. id (e.g., es). (default="${src_lang}")
    --tgt_lang=        # target language abbrev. id (e.g., en). (default="${tgt_lang}")
    --use_src_ctc_lang=    # Incorporate ASR loss (use src texts) or not

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_lm        # Language model path for decoding (default="${inference_lm}").
    --inference_st_model # ST model path for decoding (default="${inference_st_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --src_bpe_train_text # Text file path of bpe training set for source language.
    --tgt_bpe_train_text # Text file path of bpe training set for target language
    --lm_train_text  # Text file path of language model training set.
    --lm_dev_text   # Text file path of language model development set (default="${lm_dev_text}").
    --lm_test_text  # Text file path of language model evaluation set (default="${lm_test_text}").
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
    --st_speech_fold_length # fold_length for speech data during ST training (default="${st_speech_fold_length}").
    --st_text_fold_length   # fold_length for text data during ST training (default="${st_text_fold_length}").
    --lm_fold_length         # fold_length for LM training (default="${lm_fold_length}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(scripts/utils/print_args.sh $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_audio=${dumpdir}/audio_raw
    data_extract=${dumpdir}/extracted
    data_feats=${dumpdir}/raw
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Extra files for translation process
if [ $use_src_ctc_lang = true ]; then
    utt_extra_files="text.${src_ctc_case}.${src_ctc_lang} text.${tgt_case}.${tgt_lang}"
else
    utt_extra_files="text.${tgt_case}.${tgt_lang}"
fi

# Use the same text as ST for bpe training if not specified.
[ -z "${src_ctc_bpe_train_text}" ] && [ $use_src_ctc_lang = true ] && src_ctc_bpe_train_text="${data_feats}/${train_set}/text.${src_ctc_case}.${src_ctc_lang}"
[ -z "${src_bpe_train_text}" ] && src_bpe_train_text="${data_feats}/${train_set}/text.${src_case}.${src_lang}"
[ -z "${tgt_bpe_train_text}" ] && tgt_bpe_train_text="${data_feats}/${train_set}/text.${tgt_case}.${tgt_lang}"
# Use the same text as ST for lm training if not specified.
[ -z "${lm_train_text}" ] && lm_train_text="${data_feats}/${train_set}/text.${tgt_case}.${tgt_lang}"
# Use the same text as ST for lm training if not specified.
[ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${valid_set}/text.${tgt_case}.${tgt_lang}"
# Use the text of the 1st evaldir if lm_test is not specified
[ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${test_sets%% *}/text.${tgt_case}.${tgt_lang}"

# Check tokenization type
token_listdir=data/${src_lang}_${tgt_lang}_token_list
# The src_ctc bpedir is set
src_ctc_bpedir="${token_listdir}/src_ctc_bpe_${src_ctc_bpemode}${src_ctc_nbpe}"
src_ctc_bpeprefix="${src_ctc_bpedir}"/bpe
src_ctc_bpemodel="${src_ctc_bpeprefix}".model
src_ctc_bpetoken_list="${src_ctc_bpedir}"/tokens.txt
src_ctc_chartoken_list="${token_listdir}"/char/src_ctc_tokens.txt
# The tgt bpedir is set for all cases when using bpe
tgt_bpedir="${token_listdir}/tgt_bpe_${tgt_bpemode}${tgt_nbpe}"
tgt_bpeprefix="${tgt_bpedir}"/bpe
tgt_bpemodel="${tgt_bpeprefix}".model
tgt_bpetoken_list="${tgt_bpedir}"/tokens.txt
tgt_chartoken_list="${token_listdir}"/char/tgt_tokens.txt
hugging_face_token_list="${token_listdir}/hugging_face_"${hugging_face_model_name_or_path/\//-}/tokens.txt
if "${token_joint}"; then
    # if token_joint, the bpe training will use both src_lang and tgt_lang to train a single bpe model
    src_bpedir="${tgt_bpedir}"
    src_bpeprefix="${tgt_bpeprefix}"
    src_bpemodel="${tgt_bpemodel}"
    src_bpetoken_list="${tgt_bpetoken_list}"
    src_chartoken_list="${tgt_chartoken_list}"
else
    src_bpedir="${token_listdir}/src_bpe_${src_bpemode}${src_nbpe}"
    src_bpeprefix="${src_bpedir}"/bpe
    src_bpemodel="${src_bpeprefix}".model
    src_bpetoken_list="${src_bpedir}"/tokens.txt
    src_chartoken_list="${token_listdir}"/char/src_tokens.txt
fi

# NOTE: keep for future development.
# shellcheck disable=SC2034
src_ctc_wordtoken_list="${token_listdir}"/word/src_ctc_tokens.txt
tgt_wordtoken_list="${token_listdir}"/word/tgt_tokens.txt
if "${token_joint}"; then
    src_wordtoken_list="${tgt_wordtoken_list}"
else
    src_wordtoken_list="${token_listdir}"/word/src_tokens.txt
fi

# Set token types for src and tgt langs
if [ $use_src_ctc_lang = false ]; then
    src_ctc_token_type=none
    src_ctc_token_list=none
elif [ "${src_ctc_token_type}" = bpe ]; then
    src_ctc_token_list="${src_ctc_bpetoken_list}"
elif [ "${src_ctc_token_type}" = char ]; then
    src_ctc_token_list="${src_ctc_chartoken_list}"
    src_ctc_bpemodel=none
elif [ "${src_ctc_token_type}" = word ]; then
    src_ctc_token_list="${src_ctc_wordtoken_list}"
    src_ctc_bpemodel=none
else
    log "Error: not supported --src_ctc_token_type '${src_ctc_token_type}'"
    exit 2
fi

if [ "${src_token_type}" = bpe ]; then
    src_token_list="${src_bpetoken_list}"
elif [ "${src_token_type}" = char ]; then
    src_token_list="${src_chartoken_list}"
    src_bpemodel=none
elif [ "${src_token_type}" = word ]; then
    src_token_list="${src_wordtoken_list}"
    src_bpemodel=none
else
    log "Error: not supported --src_token_type '${src_token_type}'"
    exit 2
fi
if [ "${tgt_token_type}" = bpe ]; then
    tgt_token_list="${tgt_bpetoken_list}"
elif [ "${tgt_token_type}" = char ]; then
    tgt_token_list="${tgt_chartoken_list}"
    tgt_bpemodel=none
elif [ "${tgt_token_type}" = word ]; then
    tgt_token_list="${tgt_wordtoken_list}"
    tgt_bpemodel=none
elif [ "${tgt_token_type}" = hugging_face ]; then
    tgt_token_list="${hugging_face_token_list}"
    tgt_bpemodel=${hugging_face_model_name_or_path}
else
    log "Error: not supported --tgt_token_type '${tgt_token_type}'"
    exit 2
fi

if ${use_word_lm}; then
    log "Error: Word LM is not supported yet"
    exit 2

    lm_token_list="${tgt_wordtoken_list}"
    lm_token_type=word
else
    lm_token_list="${tgt_token_list}"
    lm_token_type="${tgt_token_type}"
fi

if [ ${kmeans_feature} = "mfcc" ]; then  # MFCC has no layer
    kmeans_feature_type=$(echo "${kmeans_feature}" | cut -d/ -f1)
    layer=
    kmeans_feature_conf="{type=mfcc}"
else
    kmeans_feature_type=$(echo "${kmeans_feature}" | cut -d/ -f1)
    layer=$(echo "${kmeans_feature}" | cut -d/ -f2)
    # TODO(simpleoier): to support features beyond s3prl
    s3prl_conf="{upstream=${kmeans_feature_type}}"
    kmeans_feature_conf="{type=s3prl,conf={s3prl_conf=${s3prl_conf},download_dir=ckpt,multilayer_feature=False,layer=${layer}}}"
fi
km_dir="${expdir}"/kmeans/$(echo "${kmeans_feature}" | tr "/" "_")_${nclusters}clusters

# Set tag for naming of model directory
if [ -z "${st_tag}" ]; then
    if [ -n "${st_config}" ]; then
        st_tag="$(basename "${st_config}" .yaml)_${feats_type}"
    else
        st_tag="train_${feats_type}"
    fi
    st_tag+="_${src_lang}_${tgt_lang}_${tgt_token_type}_${tgt_case}"
    if [ "${tgt_token_type}" = bpe ]; then
        st_tag+="${tgt_nbpe}"
    fi
    if [ "${tgt_token_type}" = hugging_face ]; then
        st_tag+="_"${hugging_face_model_name_or_path/\//-}
    fi
    # Add overwritten arg's info
    if [ -n "${st_args}" ]; then
        st_tag+="$(echo "${st_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        st_tag+="_sp"
    fi
fi
if [ -z "${lm_tag}" ]; then
    if [ -n "${lm_config}" ]; then
        lm_tag="$(basename "${lm_config}" .yaml)"
    else
        lm_tag="train"
    fi
    lm_tag+="_${src_lang}_${tgt_lang}_${lm_token_type}"
    if [ "${lm_token_type}" = bpe ]; then
        lm_tag+="${tgt_nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${st_stats_dir}" ]; then
    st_stats_dir="${expdir}/st_stats_${feats_type}_${src_lang}_${tgt_lang}_${tgt_token_type}"
    if [ "${tgt_token_type}" = bpe ]; then
        st_stats_dir+="${tgt_nbpe}"
    fi
    if [ "${tgt_token_type}" = hugging_face ]; then
        st_stats_dir+="_"${hugging_face_model_name_or_path/\//-}
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        st_stats_dir+="_sp"
    fi
fi
if [ -z "${lm_stats_dir}" ]; then
    lm_stats_dir="${expdir}/lm_stats_${src_lang}_${tgt_lang}_${lm_token_type}"
    if [ "${lm_token_type}" = bpe ]; then
        lm_stats_dir+="${tgt_nbpe}"
    fi
fi
# The directory used for training commands
if [ -z "${st_exp}" ]; then
    st_exp="${expdir}/st_${st_tag}"
fi
if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}"
fi
if [ -z "${ngram_exp}" ]; then
    ngram_exp="${expdir}/ngram"
fi


if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if "${use_lm}"; then
        inference_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    if "${use_ngram}"; then
        inference_tag+="_ngram_$(basename "${ngram_exp}")_$(echo "${inference_ngram}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    inference_tag+="_st_model_$(echo "${inference_st_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

    if "${use_k2}"; then
      inference_tag+="_use_k2"
    fi
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if [ -n "${speed_perturb_factors}" ]; then
            log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"
            for factor in ${speed_perturb_factors}; do
                if python3 -c "assert ${factor} != 1.0" 2>/dev/null; then
                    scripts/utils/perturb_data_dir_speed.sh --utt_extra_files "${utt_extra_files}" \
                         "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}"
                    _dirs+="data/${train_set}_sp${factor} "
                else
                    # If speed factor is 1, same as the original
                    _dirs+="data/${train_set} "
                fi
            done
            utils/combine_data.sh --extra_files "${utt_extra_files}" "data/${train_set}_sp" ${_dirs}
            for extra_file in ${utt_extra_files}; do
                python pyscripts/utils/remove_duplicate_keys.py data/"${train_set}_sp"/${extra_file} > data/"${train_set}_sp"/${extra_file}.tmp
                mv data/"${train_set}_sp"/${extra_file}.tmp data/"${train_set}_sp"/${extra_file}
            done
        else
           log "Skip stage 2: Speed perturbation"
        fi
    fi

    train_sp_sets=
    if [ -n "${speed_perturb_factors}" ]; then
        for factor in ${speed_perturb_factors}; do
            if python3 -c "assert ${factor} != 1.0" 2>/dev/null; then
                train_sp_sets+="${train_set}_sp${factor} "
            fi
        done
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        _dsets="${train_set} ${train_sp_sets} ${valid_set} ${test_sets}"
        if [ "${feats_type}" = raw ]; then
            log "Stage 3: Format wav.scp: data/ -> ${data_audio}"

            # ====== Recreating "wav.scp" ======
            # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
            # shouldn't be used in training process.
            # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
            # and it can also change the audio-format and sampling rate.
            # If nothing is need, then format_wav_scp.sh does nothing:
            # i.e. the input file format and rate is same as the output.

            for dset in ${_dsets}; do
                utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_audio}/${dset}"

                # expand the utt_extra_files for multi-references
                expand_utt_extra_files=""
                for extra_file in ${utt_extra_files}; do
                    # with regex to support multi-references
                    for single_file in $(ls data/"${dset}"/${extra_file}*); do
                        cp ${single_file} "${data_audio}/${dset}"
                        expand_utt_extra_files="${expand_utt_extra_files} $(basename ${single_file})"
                    done
                done
                echo "${expand_utt_extra_files}"
                utils/fix_data_dir.sh --utt_extra_files "${expand_utt_extra_files}" "${data_audio}/${dset}"
                for extra_file in ${expand_utt_extra_files}; do
                    LC_ALL=C sort -u -k1,1 "${data_audio}/${dset}/${extra_file}" -o "${data_audio}/${dset}/${extra_file}"
                done

                rm -f "${data_audio}/${dset}"/{segments,wav.scp,reco2file_and_channel,reco2dur}
                _opts=
                if [ -e data/"${dset}"/segments ]; then
                    # "segments" is used for splitting wav files which are written in "wav".scp
                    # into utterances. The file format of segments:
                    #   <segment_id> <record_id> <start_time> <end_time>
                    #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                    # Where the time is written in seconds.
                    _opts+="--segments data/${dset}/segments "
                fi
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "data/${dset}/wav.scp" "${data_audio}/${dset}"

                echo "${feats_type}" > "${data_audio}/${dset}/feats_type"
            done
        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi
    fi

    utt_extra_files+=" text.${src_case}.${src_lang}"
    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        # log "Stage 4a: Perform Kmeans using ${kmeans_feature_type} features"

        # scripts/feats/perform_kmeans.sh \
        #     --stage 1 --stop-stage 4 \
        #     --train_set "${train_set}" \
        #     --dev_set "${valid_set}" \
        #     --other_sets "${test_sets} ${train_sp_sets}" \
        #     --datadir "${data_audio}" \
        #     --featdir "${data_extract}" \
        #     --audio_format "${audio_format}" \
        #     --feature_type "${kmeans_feature_type}" \
        #     --layer "${layer}" \
        #     --feature_conf "${kmeans_feature_conf}" \
        #     --km_dir "${km_dir}" \
        #     --portion "${portion}" \
        #     --nclusters "${nclusters}" \
        #     --storage_save_mode ${storage_save_mode} \
        #     --use_gpu true \
        #     --nj ${nj} \
        #     --cpu_cmd "${train_cmd}" \
        #     --cuda_cmd "${cuda_cmd}" \
        #     ${kmeans_opts}
        
        log "Stage 4b: Prepare token_list and convert number indices to CJK tokens"

        # Get uniq chars
        if [ ! -f "${km_dir}/../"distinct_cjk_token_lists ]; then
            if [ ${nclusters} -ge 20900 ]; then
                echo "Warning: too many clusters, be careful with the distinct token list."
            fi
            python3 -c "for i in range(${nclusters}): print(i, chr(int('4e00', 16) + i))" \
                > "${km_dir}/../"distinct_cjk_token_lists
        fi

        _suf=
        if [ -n "${layer}" ]; then
            _suf="layer${layer}/"
        fi

        if [ "${src_case}" = ts ]; then
            echo "keep the original discrete token sequence"
            for dset in "${train_set}" ${train_sp_sets} "${valid_set}" ${test_sets}; do
                awk '
                    (FILENAME==ARGV[1]) {a[$1]=$2}
                    (FILENAME==ARGV[2]) {
                        out="";
                        for (i=2; i<=NF; i++) {
                            out=out""a[$i];
                        }
                        print($1,out);
                    }' "${km_dir}/../"distinct_cjk_token_lists \
                    "${data_extract}/${kmeans_feature_type}/${_suf}${dset}/pseudo_labels_km${nclusters}.txt" \
                    > "data/${dset}"/text.${src_case}.${src_lang}
            done
        elif [ "${src_case}" = rm ]; then
            echo "remove repetitions in the discrete token sequence"
            for dset in "${train_set}" ${train_sp_sets} "${valid_set}" ${test_sets}; do
                awk '
                    (FILENAME==ARGV[1]) {a[$1]=$2}
                    (FILENAME==ARGV[2]) {
                        out="";
                        for (i=2; i<=NF; i++) {
                            if ($i != $(i-1)) {out=out""a[$i]}
                        }
                        print($1,out);
                    }' "${km_dir}/../"distinct_cjk_token_lists \
                    "${data_extract}/${kmeans_feature_type}/${_suf}${dset}/pseudo_labels_km${nclusters}.txt" \
                    > "data/${dset}/text.${src_case}.${src_lang}"
            done
        else
            echo "Unrecognized src_case ${src_case}" && exit 1;
        fi

        expand_utt_extra_files="${utt_extra_files} feats.scp utt2num_frames"
        if [ -n "${speed_perturb_factors}" ]; then
            _dirs="data/${train_set} "
            for factor in ${speed_perturb_factors}; do
                if python3 -c "assert ${factor} != 1.0" 2>/dev/null; then
                    _dirs+="data/${train_set}_sp${factor} "
                fi
            done
            utils/combine_data.sh \
                --extra_files "${expand_utt_extra_files}" \
                "data/${train_set}_sp" ${_dirs}
        fi
    fi

    # TODP; stage 5 and stage 6 have bugs: 1) no fix datadir, 2) missing multi-ref
    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
    fi
    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _dsets="${train_set} ${valid_set} ${test_sets}"
        if [ "${feats_type}" = raw ]; then
            log "Stage 5: Format discrete data/ -> ${data_feats}"

            for dset in ${_dsets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                mkdir -p "${data_feats}${_suf}/${dset}"

                for extra_file in ${utt_extra_files}; do
                    # with regex to suuport multi-references
                    for single_file in data/"${dset}"/*; do
                        base=$(basename "${single_file}")
                        [ "${base}" = "${extra_file}" ] && cp ${single_file} "${data_feats}${_suf}/${dset}"
                    done
                done
                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done
        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi
    fi


    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: Data filtering: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do
            # Copy data dir
            mkdir -p "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            for utt_extra_file in ${utt_extra_files}; do
                cp "${data_feats}/org/${dset}/${utt_extra_file}" "${data_feats}/${dset}"
            done
        done

        # shellcheck disable=SC2002
        cat ${lm_train_text} | awk ' { if( NF != 1 ) print $0; } ' \
            > "${data_feats}/lm_train.${src_lang}.${tgt_case}.${tgt_lang}.txt"
    fi

    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        # Combine source and target texts when using joint tokenization
        if "${token_joint}"; then
            log "Merge src and target data if joint BPE"

            cat $tgt_bpe_train_text > ${data_feats}/${train_set}/text.${src_lang}_${tgt_lang}
            [ ! -z "${src_bpe_train_text}" ] && cat ${src_bpe_train_text} >> ${data_feats}/${train_set}/text.${src_lang}_${tgt_lang}
            # Set the new text as the target text
            tgt_bpe_train_text="${data_feats}/${train_set}/text.${src_lang}_${tgt_lang}"
        fi

        # First generate tgt lang
        # if [ "${tgt_token_type}" = bpe ]; then
        #     log "Stage 7a: Generate token_list from ${tgt_bpe_train_text} using BPE for tgt_lang"

        #     mkdir -p "${tgt_bpedir}"
        #     # shellcheck disable=SC2002
        #     cat ${tgt_bpe_train_text} | cut -f 2- -d" "  > "${tgt_bpedir}"/train.txt

        #     if [ -n "${tgt_bpe_nlsyms}" ]; then
        #         _opts_spm="--user_defined_symbols=${tgt_bpe_nlsyms}"
        #     else
        #         _opts_spm=""
        #     fi

        #     spm_train \
        #         --input="${tgt_bpedir}"/train.txt \
        #         --vocab_size="${tgt_nbpe}" \
        #         --model_type="${tgt_bpemode}" \
        #         --model_prefix="${tgt_bpeprefix}" \
        #         --character_coverage=${tgt_bpe_char_cover} \
        #         --input_sentence_size="${tgt_bpe_input_sentence_size}" \
        #         ${_opts_spm}

        #     {
        #     echo "${blank}"
        #     echo "${oov}"
        #     # Remove <unk>, <s>, </s> from the vocabulary
        #     <"${tgt_bpeprefix}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
        #     echo "${sos_eos}"
        #     } > "${tgt_token_list}"

        # elif [ "${tgt_token_type}" = char ] || [ "${tgt_token_type}" = word ]; then
        #     log "Stage 7a: Generate character level token_list from ${tgt_bpe_train_text}  for tgt_lang"

        #     _opts="--non_linguistic_symbols ${nlsyms_txt}"

        #     # shellcheck disable=SC2002
        #     cat ${tgt_bpe_train_text} | cut -f 2- -d" "  > "${data_feats}"/token_train.txt

        #     # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        #     # 0 is reserved for CTC-blank for ST and also used as ignore-index in the other task
        #     ${python} -m espnet2.bin.tokenize_text  \
        #         --token_type "${tgt_token_type}" \
        #         --input "${data_feats}/token_train.txt" --output "${tgt_token_list}" ${_opts} \
        #         --field 2- \
        #         --cleaner "${cleaner}" \
        #         --g2p "${g2p}" \
        #         --write_vocabulary true \
        #         --add_symbol "${blank}:0" \
        #         --add_symbol "${oov}:1" \
        #         --add_symbol "${sos_eos}:-1"

        # else
        #     log "Error: not supported --token_type '${tgt_token_type}'"
        #     exit 2
        # fi

        # # Create word-list for word-LM training
        # if ${use_word_lm} && [ "${tgt_token_type}" != word ]; then
        #     log "Generate word level token_list from ${data_feats}/lm_train.${src_lang}.${tgt_case}.${tgt_lang}.txt"
        #     ${python} -m espnet2.bin.tokenize_text \
        #         --token_type word \
        #         --input "${data_feats}/lm_train.${src_lang}.${tgt_case}.${tgt_lang}.txt" --output "${lm_token_list}" \
        #         --field 2- \
        #         --cleaner "${cleaner}" \
        #         --g2p "${g2p}" \
        #         --write_vocabulary true \
        #         --vocabulary_size "${word_vocab_size}" \
        #         --add_symbol "${blank}:0" \
        #         --add_symbol "${oov}:1" \
        #         --add_symbol "${sos_eos}:-1"
        # fi

        # Then generate src_ctc lang
        # if [ $use_src_ctc_lang = true ]; then
        #     if [ "${src_ctc_token_type}" = bpe ]; then
        #         log "Stage 7b: Generate token_list from ${src_ctc_bpe_train_text} using BPE for src_ctc_lang"

        #         mkdir -p "${src_ctc_bpedir}"
        #         # shellcheck disable=SC2002
        #         cat ${src_ctc_bpe_train_text} | cut -f 2- -d" "  > "${src_ctc_bpedir}"/train.txt

        #         if [ -n "${src_ctc_bpe_nlsyms}" ]; then
        #             _opts_spm="--user_defined_symbols=${src_ctc_bpe_nlsyms}"
        #         else
        #             _opts_spm=""
        #         fi

        #         spm_train \
        #             --input="${src_ctc_bpedir}"/train.txt \
        #             --vocab_size="${src_ctc_nbpe}" \
        #             --model_type="${src_ctc_bpemode}" \
        #             --model_prefix="${src_ctc_bpeprefix}" \
        #             --character_coverage=${src_ctc_bpe_char_cover} \
        #             --input_sentence_size="${src_ctc_bpe_input_sentence_size}" \
        #             ${_opts_spm}

        #         {
        #         echo "${blank}"
        #         echo "${oov}"
        #         # Remove <unk>, <s>, </s> from the vocabulary
        #         <"${src_ctc_bpeprefix}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
        #         echo "${sos_eos}"
        #         } > "${src_ctc_token_list}"

        #     elif [ "${src_ctc_token_type}" = char ] || [ "${src_ctc_token_type}" = word ]; then
        #         log "Stage 7b: Generate character level token_list from ${src_ctc_bpe_train_text}  for src_ctc_lang"

        #         _opts="--non_linguistic_symbols ${nlsyms_txt}"

        #         # shellcheck disable=SC2002
        #         cat ${src_ctc_bpe_train_text} | cut -f 2- -d" "  > "${data_feats}"/token_train.txt

        #         # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        #         # 0 is reserved for CTC-blank for ST and also used as ignore-index in the other task
        #         ${python} -m espnet2.bin.tokenize_text  \
        #             --token_type "${src_ctc_token_type}" \
        #             --input "${data_feats}/token_train.txt" --output "${src_ctc_token_list}" ${_opts} \
        #             --field 2- \
        #             --cleaner "${cleaner}" \
        #             --g2p "${g2p}" \
        #             --write_vocabulary true \
        #             --add_symbol "${blank}:0" \
        #             --add_symbol "${oov}:1" \
        #             --add_symbol "${sos_eos}:-1"

        #     else
        #         log "Error: not supported --token_type '${src_ctc_token_type}'"
        #         exit 2
        #     fi
        # fi

        # Final generate src lang
        if "${token_joint}"; then
            log "Stage 7c: Skip separate token construction for src_lang when setting ${token_joint} as true"
        else:
            if [ "${src_token_type}" = bpe ]; then
                log "Stage 7c: Generate token_list from ${src_bpe_train_text} using BPE for src_lang"

                mkdir -p "${src_bpedir}"
                # shellcheck disable=SC2002
                cat ${src_bpe_train_text} | cut -f 2- -d" "  > "${src_bpedir}"/train.txt

                if [ -n "${src_bpe_nlsyms}" ]; then
                    _opts_spm="--user_defined_symbols=${src_bpe_nlsyms}"
                else
                    _opts_spm=""
                fi

                spm_train \
                    --input="${src_bpedir}"/train.txt \
                    --vocab_size="${src_nbpe}" \
                    --model_type="${src_bpemode}" \
                    --model_prefix="${src_bpeprefix}" \
                    --character_coverage=${src_bpe_char_cover} \
                    --input_sentence_size="${src_bpe_input_sentence_size}" \
                    ${_opts_spm}

                {
                echo "${blank}"
                echo "${oov}"
                # Remove <unk>, <s>, </s> from the vocabulary
                <"${src_bpeprefix}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
                echo "${sos_eos}"
                } > "${src_token_list}"

            elif [ "${src_token_type}" = char ] || [ "${src_token_type}" = word ]; then
                log "Stage 5b: Generate character level token_list from ${src_bpe_train_text}  for src_lang"

                _opts="--non_linguistic_symbols ${nlsyms_txt}"

                # shellcheck disable=SC2002
                cat ${src_bpe_train_text} | cut -f 2- -d" "  > "${data_feats}"/token_train.txt

                # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
                # 0 is reserved for CTC-blank for ST and also used as ignore-index in the other task
                ${python} -m espnet2.bin.tokenize_text  \
                    --token_type "${src_token_type}" \
                    --input "${data_feats}/token_train.txt" --output "${src_token_list}" ${_opts} \
                    --field 2- \
                    --cleaner "${cleaner}" \
                    --g2p "${g2p}" \
                    --write_vocabulary true \
                    --add_symbol "${blank}:0" \
                    --add_symbol "${oov}:1" \
                    --add_symbol "${sos_eos}:-1"

            else
                log "Error: not supported --token_type '${src_token_type}'"
                exit 2
            fi
        fi
    fi

else
    log "Skip the stages for data preparation"
fi


# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then
    if "${use_lm}"; then
        if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
            log "Stage 8: LM collect stats: train_set=${data_feats}/lm_train.${src_lang}.${tgt_case}.${tgt_lang}.txt, dev_set=${lm_dev_text}"

            _opts=
            if [ -n "${lm_config}" ]; then
                # To generate the config file: e.g.
                #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
                _opts+="--config ${lm_config} "
            fi

            # 1. Split the key file
            _logdir="${lm_stats_dir}/logdir"
            mkdir -p "${_logdir}"
            # Get the minimum number among ${nj} and the number lines of input files
            _nj=$(min "${nj}" "$(<${data_feats}/lm_train.${src_lang}.${tgt_case}.${tgt_lang}.txt wc -l)" "$(<${lm_dev_text} wc -l)")

            key_file="${data_feats}/lm_train.${src_lang}.${tgt_case}.${tgt_lang}.txt"
            split_scps=""
            for n in $(seq ${_nj}); do
                split_scps+=" ${_logdir}/train.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            key_file="${lm_dev_text}"
            split_scps=""
            for n in $(seq ${_nj}); do
                split_scps+=" ${_logdir}/dev.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Generate run.sh
            log "Generate '${lm_stats_dir}/run.sh'. You can resume the process from stage 8 using this script"
            mkdir -p "${lm_stats_dir}"; echo "${run_args} --stage 8 \"\$@\"; exit \$?" > "${lm_stats_dir}/run.sh"; chmod +x "${lm_stats_dir}/run.sh"

            # 3. Submit jobs
            log "LM collect-stats started... log: '${_logdir}/stats.*.log'"
            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.
            # shellcheck disable=SC2046,SC2086
            ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
                ${python} -m espnet2.bin.lm_train \
                    --collect_stats true \
                    --use_preprocessor true \
                    --bpemodel "${tgt_bpemodel}" \
                    --token_type "${lm_token_type}"\
                    --token_list "${lm_token_list}" \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --cleaner "${cleaner}" \
                    --g2p "${g2p}" \
                    --train_data_path_and_name_and_type "${data_feats}/lm_train.${src_lang}.${tgt_case}.${tgt_lang}.txt,text,text" \
                    --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
                    --train_shape_file "${_logdir}/train.JOB.scp" \
                    --valid_shape_file "${_logdir}/dev.JOB.scp" \
                    --output_dir "${_logdir}/stats.JOB" \
                    ${_opts} ${lm_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

            # 4. Aggregate shape files
            _opts=
            for i in $(seq "${_nj}"); do
                _opts+="--input_dir ${_logdir}/stats.${i} "
            done
            # shellcheck disable=SC2086
            ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${lm_stats_dir}"

            # Append the num-tokens at the last dimensions. This is used for batch-bins count
            <"${lm_stats_dir}/train/text_shape" \
                awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
                >"${lm_stats_dir}/train/text_shape.${lm_token_type}"

            <"${lm_stats_dir}/valid/text_shape" \
                awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
                >"${lm_stats_dir}/valid/text_shape.${lm_token_type}"
        fi


        if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
            log "Stage 9: LM Training: train_set=${data_feats}/lm_train.${src_lang}.${tgt_case}.${tgt_lang}.txt, dev_set=${lm_dev_text}"

            _opts=
            if [ -n "${lm_config}" ]; then
                # To generate the config file: e.g.
                #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
                _opts+="--config ${lm_config} "
            fi

            if [ "${num_splits_lm}" -gt 1 ]; then
                # If you met a memory error when parsing text files, this option may help you.
                # The corpus is split into subsets and each subset is used for training one by one in order,
                # so the memory footprint can be limited to the memory required for each dataset.

                _split_dir="${lm_stats_dir}/splits${num_splits_lm}"
                if [ ! -f "${_split_dir}/.done" ]; then
                    rm -f "${_split_dir}/.done"
                    ${python} -m espnet2.bin.split_scps \
                      --scps "${data_feats}/lm_train.${src_lang}.${tgt_case}.${tgt_lang}.txt" "${lm_stats_dir}/train/text_shape.${lm_token_type}" \
                      --num_splits "${num_splits_lm}" \
                      --output_dir "${_split_dir}"
                    touch "${_split_dir}/.done"
                else
                    log "${_split_dir}/.done exists. Spliting is skipped"
                fi

                _opts+="--train_data_path_and_name_and_type ${_split_dir}/lm_train.${src_lang}.${tgt_case}.${tgt_lang}.txt,text,text "
                _opts+="--train_shape_file ${_split_dir}/text_shape.${lm_token_type} "
                _opts+="--multiple_iterator true "

            else
                _opts+="--train_data_path_and_name_and_type ${data_feats}/lm_train.${src_lang}.${tgt_case}.${tgt_lang}.txt,text,text "
                _opts+="--train_shape_file ${lm_stats_dir}/train/text_shape.${lm_token_type} "
            fi

            # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

            log "Generate '${lm_exp}/run.sh'. You can resume the process from stage 9 using this script"
            mkdir -p "${lm_exp}"; echo "${run_args} --stage 9 \"\$@\"; exit \$?" > "${lm_exp}/run.sh"; chmod +x "${lm_exp}/run.sh"

            log "LM training started... log: '${lm_exp}/train.log'"
            if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
                # SGE can't include "/" in a job name
                jobname="$(basename ${lm_exp})"
            else
                jobname="${lm_exp}/train.log"
            fi

            # TODO(jiatong): fix bpe
            # shellcheck disable=SC2086
            ${python} -m espnet2.bin.launch \
                --cmd "${cuda_cmd} --name ${jobname}" \
                --log "${lm_exp}"/train.log \
                --ngpu "${ngpu}" \
                --num_nodes "${num_nodes}" \
                --init_file_prefix "${lm_exp}"/.dist_init_ \
                --multiprocessing_distributed true -- \
                ${python} -m espnet2.bin.lm_train \
                    --ngpu "${ngpu}" \
                    --use_preprocessor true \
                    --bpemodel "${tgt_bpemodel}" \
                    --token_type "${lm_token_type}"\
                    --token_list "${lm_token_list}" \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --cleaner "${cleaner}" \
                    --g2p "${g2p}" \
                    --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
                    --valid_shape_file "${lm_stats_dir}/valid/text_shape.${lm_token_type}" \
                    --fold_length "${lm_fold_length}" \
                    --resume true \
                    --output_dir "${lm_exp}" \
                    ${_opts} ${lm_args}

        fi


        if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
            log "Stage 10: Calc perplexity: ${lm_test_text}"
            _opts=
            # TODO(kamo): Parallelize?
            log "Perplexity calculation started... log: '${lm_exp}/perplexity_test/lm_calc_perplexity.log'"
            # shellcheck disable=SC2086
            ${cuda_cmd} --gpu "${ngpu}" "${lm_exp}"/perplexity_test/lm_calc_perplexity.log \
                ${python} -m espnet2.bin.lm_calc_perplexity \
                    --ngpu "${ngpu}" \
                    --data_path_and_name_and_type "${lm_test_text},text,text" \
                    --train_config "${lm_exp}"/config.yaml \
                    --model_file "${lm_exp}/${inference_lm}" \
                    --output_dir "${lm_exp}/perplexity_test" \
                    ${_opts}
            log "PPL: ${lm_test_text}: $(cat ${lm_exp}/perplexity_test/ppl)"

        fi

    else
        log "Stage 8-10: Skip lm-related stages: use_lm=${use_lm}"
    fi


    if "${use_ngram}"; then
        mkdir -p ${ngram_exp}
    fi
    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        if "${use_ngram}"; then
            log "Stage 11: Ngram Training: train_set=${data_feats}/lm_train.${src_lang}.${tgt_case}.${tgt_lang}.txt"
            cut -f 2 -d " " ${data_feats}/lm_train.${src_lang}.${tgt_case}.${tgt_lang}.txt | lmplz -S "20%" --discount_fallback -o ${ngram_num} - >${ngram_exp}/${ngram_num}gram.arpa
            build_binary -s ${ngram_exp}/${ngram_num}gram.arpa ${ngram_exp}/${ngram_num}gram.bin
        else
            log "Stage 11: Skip ngram stages: use_ngram=${use_ngram}"
        fi
    fi


    if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
        _st_train_dir="${data_feats}/${train_set}"
        _st_valid_dir="${data_feats}/${valid_set}"
        log "Stage 12: ST collect stats: train_set=${_st_train_dir}, valid_set=${_st_valid_dir}"

        _opts=
        if [ -n "${st_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.st_train --print_config --optim adam
            _opts+="--config ${st_config} "
        fi

        # 1. Split the key file
        _logdir="${st_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        _scp=text.${src_case}.${src_lang}

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_st_train_dir}/${_scp} wc -l)" "$(<${_st_valid_dir}/${_scp} wc -l)")

        key_file="${_st_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_st_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${st_stats_dir}/run.sh'. You can resume the process from stage 12 using this script"
        mkdir -p "${st_stats_dir}"; echo "${run_args} --stage 12 \"\$@\"; exit \$?" > "${st_stats_dir}/run.sh"; chmod +x "${st_stats_dir}/run.sh"

        # 3. Submit jobs
        log "ST collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        if [ $use_src_ctc_lang = true ]; then
            _opts+="--ctc_bpemodel ${src_ctc_bpemodel} "
            _opst+="--ctc_token_type ${src_ctc_token_type} "
            _opts+="--ctc_token_list ${src_ctc_token_list} "
            _opts+="--train_data_path_and_name_and_type ${_st_train_dir}/text.${src_ctc_case}.${src_ctc_lang},text_ctc,text "
            _opts+="--valid_data_path_and_name_and_type ${_st_valid_dir}/text.${src_ctc_case}.${src_ctc_lang},text_ctc,text "
        fi
        # TODO(jiatong): fix different bpe model
        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.mt_train \
                --collect_stats true \
                --use_preprocessor true \
                --bpemodel "${tgt_bpemodel}" \
                --src_bpemodel "${src_bpemodel}" \
                --token_type "${tgt_token_type}" \
                --src_token_type "${src_token_type}" \
                --token_list "${tgt_token_list}" \
                --src_token_list "${src_token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --train_data_path_and_name_and_type "${_st_train_dir}/text.${src_case}.${src_lang},src_text,text" \
                --train_data_path_and_name_and_type "${_st_train_dir}/text.${tgt_case}.${tgt_lang},text,text" \
                --valid_data_path_and_name_and_type "${_st_valid_dir}/text.${src_case}.${src_lang},src_text,text" \
                --valid_data_path_and_name_and_type "${_st_valid_dir}/text.${tgt_case}.${tgt_lang},text,text" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${st_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${st_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        <"${st_stats_dir}/train/text_shape" \
            awk -v N="$(<${tgt_token_list} wc -l)" '{ print $0 "," N }' \
            >"${st_stats_dir}/train/text_shape.${tgt_token_type}"

        <"${st_stats_dir}/train/src_text_shape" \
            awk -v N="$(<${src_token_list} wc -l)" '{ print $0 "," N }' \
            >"${st_stats_dir}/train/src_text_shape.${src_token_type}"

        <"${st_stats_dir}/valid/text_shape" \
            awk -v N="$(<${tgt_token_list} wc -l)" '{ print $0 "," N }' \
            >"${st_stats_dir}/valid/text_shape.${tgt_token_type}"
        
        <"${st_stats_dir}/valid/src_text_shape" \
            awk -v N="$(<${src_token_list} wc -l)" '{ print $0 "," N }' \
            >"${st_stats_dir}/valid/src_text_shape.${src_token_type}"

        if [ $use_src_ctc_lang = true ]; then
            <"${st_stats_dir}/train/text_ctc_shape" \
                awk -v N="$(<${src_ctc_token_list} wc -l)" '{ print $0 "," N }' \
                >"${st_stats_dir}/train/text_ctc_shape.${src_ctc_token_type}"

            <"${st_stats_dir}/valid/text_ctc_shape" \
                awk -v N="$(<${src_ctc_token_list} wc -l)" '{ print $0 "," N }' \
                >"${st_stats_dir}/valid/text_ctc_shape.${src_ctc_token_type}"
        fi
    fi


    if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
        _st_train_dir="${data_feats}/${train_set}"
        _st_valid_dir="${data_feats}/${valid_set}"
        log "Stage 13: ST Training: train_set=${_st_train_dir}, valid_set=${_st_valid_dir}"

        _opts=
        if [ -n "${st_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.st_train --print_config --optim adam
            _opts+="--config ${st_config} "
        fi

        _num_splits_opts=
        if [ $use_src_ctc_lang = true ]; then
            _num_splits_opts+="${_st_train_dir}/text.${src_ctc_case}.${src_ctc_lang} "
            _num_splits_opts+="${st_stats_dir}/train/text_ctc_shape.${src_ctc_token_type} "
        fi

        if [ "${num_splits_st}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${st_stats_dir}/splits${num_splits_st}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps \
                      "${_st_train_dir}/${_scp}" \
                      "${_st_train_dir}/text.${tgt_case}.${tgt_lang}" \
                      "${_st_train_dir}/text.${src_case}.${src_lang}" \
                      "${st_stats_dir}/train/text_shape.${tgt_token_type}" \
                      "${st_stats_dir}/train/src_text_shape.${src_token_type}" \
                      $_num_splits_opts \
                  --num_splits "${num_splits_st}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text.${tgt_case}.${tgt_lang},text,text "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text.${src_case}.${src_lang},src_text,text "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${tgt_token_type} "
            _opts+="--train_shape_file ${_split_dir}/src_text_shape.${src_token_type} "
            _opts+="--multiple_iterator true "
            if [ $use_src_ctc_lang = true ]; then
                _opts+="--train_data_path_and_name_and_type ${_split_dir}/text.${src_ctc_case}.${src_ctc_lang},text_ctc,text "
                _opts+="--train_shape_file ${_split_dir}/text_ctc_shape.${src_ctc_token_type} "
            fi
        else
            _opts+="--train_data_path_and_name_and_type ${_st_train_dir}/text.${tgt_case}.${tgt_lang},text,text "
            _opts+="--train_data_path_and_name_and_type ${_st_train_dir}/text.${src_case}.${src_lang},src_text,text "
            _opts+="--train_shape_file ${st_stats_dir}/train/text_shape.${tgt_token_type} "
            _opts+="--train_shape_file ${st_stats_dir}/train/src_text_shape.${src_token_type} "
            if [ $use_src_ctc_lang = true ]; then
                _opts+="--train_data_path_and_name_and_type ${_st_train_dir}/text.${src_ctc_case}.${src_ctc_lang},text_ctc,text "
                _opts+="--train_shape_file ${st_stats_dir}/train/text_ctc_shape.${src_ctc_token_type} "
            fi
        fi

        log "Generate '${st_exp}/run.sh'. You can resume the process from stage 13 using this script"
        mkdir -p "${st_exp}"; echo "${run_args} --stage 11 \"\$@\"; exit \$?" > "${st_exp}/run.sh"; chmod +x "${st_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "ST training started... log: '${st_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${st_exp})"
        else
            jobname="${st_exp}/train.log"
        fi

        if [ $use_src_ctc_lang = true ]; then
            _opts+="--ctc_bpemodel ${src_ctc_bpemodel} "
            _opts+="--ctc_token_type ${src_ctc_token_type} "
            _opts+="--ctc_token_list ${src_ctc_token_list} "
            _opts+="--valid_data_path_and_name_and_type ${_st_valid_dir}/text.${src_ctc_case}.${src_ctc_lang},text_ctc,text "
            _opts+="--valid_shape_file ${st_stats_dir}/valid/text_ctc_shape.${src_ctc_token_type} "
            _opts+="--fold_length ${st_text_fold_length} "
        fi

        # TODO(jiatong): fix bpe
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${st_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${st_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.mt_train \
                --ngpu ${ngpu} \
                --use_preprocessor true \
                --bpemodel "${tgt_bpemodel}" \
                --token_type "${tgt_token_type}" \
                --token_list "${tgt_token_list}" \
                --src_bpemodel "${src_bpemodel}" \
                --src_token_type "${src_token_type}" \
                --src_token_list "${src_token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --valid_data_path_and_name_and_type "${_st_valid_dir}/text.${tgt_case}.${tgt_lang},text,text" \
                --valid_data_path_and_name_and_type "${_st_valid_dir}/text.${src_case}.${src_lang},src_text,text" \
                --valid_shape_file "${st_stats_dir}/valid/text_shape.${tgt_token_type}" \
                --valid_shape_file "${st_stats_dir}/valid/src_text_shape.${src_token_type}" \
                --resume true \
                --fold_length "${st_text_fold_length}" \
                --fold_length "${st_text_fold_length}" \
                --output_dir "${st_exp}" \
                ${_opts} ${st_args}

    fi
else
    log "Skip the training stages"
fi


if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    st_exp="${expdir}/${download_model}"
    mkdir -p "${st_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${st_exp}/config.txt"

    # Get the path of each file
    _st_model_file=$(<"${st_exp}/config.txt" sed -e "s/.*'st_model_file': '\([^']*\)'.*$/\1/")
    _st_train_config=$(<"${st_exp}/config.txt" sed -e "s/.*'st_train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_st_model_file}" "${st_exp}"
    ln -sf "${_st_train_config}" "${st_exp}"
    inference_st_model=$(basename "${_st_model_file}")

    if [ "$(<${st_exp}/config.txt grep -c lm_file)" -gt 0 ]; then
        _lm_file=$(<"${st_exp}/config.txt" sed -e "s/.*'lm_file': '\([^']*\)'.*$/\1/")
        _lm_train_config=$(<"${st_exp}/config.txt" sed -e "s/.*'lm_train_config': '\([^']*\)'.*$/\1/")

        lm_exp="${expdir}/${download_model}/lm"
        mkdir -p "${lm_exp}"

        ln -sf "${_lm_file}" "${lm_exp}"
        ln -sf "${_lm_train_config}" "${lm_exp}"
        inference_lm=$(basename "${_lm_file}")
    fi

fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
        log "Stage 12: Decoding: training_dir=${st_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi
        if "${use_lm}"; then
            if "${use_word_lm}"; then
                _opts+="--word_lm_train_config ${lm_exp}/config.yaml "
                _opts+="--word_lm_file ${lm_exp}/${inference_lm} "
            else
                _opts+="--lm_train_config ${lm_exp}/config.yaml "
                _opts+="--lm_file ${lm_exp}/${inference_lm} "
            fi
        fi
        if "${use_ngram}"; then
             _opts+="--ngram_file ${ngram_exp}/${inference_ngram}"
        fi

        if "${use_src_lm}"; then
            if "${use_src_word_lm}"; then
                _opts+="--src_word_lm_train_config ${src_lm_exp}/config.yaml "
                _opts+="--src_word_lm_file ${src_lm_exp}/${src_inference_lm} "
            else
                _opts+="--src_lm_train_config ${src_lm_exp}/config.yaml "
                _opts+="--src_lm_file ${src_lm_exp}/${src_inference_lm} "
            fi
        fi
        if "${use_src_ngram}"; then
             _opts+="--src_ngram_file ${src_ngram_exp}/${src_inference_ngram}"
        fi

        # 2. Generate run.sh
        log "Generate '${st_exp}/${inference_tag}/run.sh'. You can resume the process from stage 12 using this script"
        mkdir -p "${st_exp}/${inference_tag}"; echo "${run_args} --stage 12 \"\$@\"; exit \$?" > "${st_exp}/${inference_tag}/run.sh"; chmod +x "${st_exp}/${inference_tag}/run.sh"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${st_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                if [[ "${audio_format}" == *ark* ]]; then
                    _type=kaldi_ark
                else
                    _type=sound
                fi
            else
                _scp=feats.scp
                _type=kaldi_ark
            fi

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            if "${use_streaming}"; then
                st_inference_tool="espnet2.bin.st_inference_streaming"
            else
                st_inference_tool="espnet2.bin.st_inference"
            fi

            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/st_inference.*.log'"
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/st_inference.JOB.log \
                ${python} -m ${st_inference_tool} \
                    --batch_size ${batch_size} \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --st_train_config "${st_exp}"/config.yaml \
                    --st_model_file "${st_exp}"/"${inference_st_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/st_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            for f in token token_int score text; do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | LC_ALL=C sort -k1 >"${_dir}/${f}"
            done

            if [ -d "${_logdir}/output.1/1asr_best_recog" ]; then
                for f in asr_token asr_token_int asr_score asr_text; do
                    for i in $(seq "${_nj}"); do
                        cat "${_logdir}/output.${i}/1asr_best_recog/${f}"
                    done | LC_ALL=C sort -k1 >"${_dir}/${f}"
                done
            fi
        done
    fi

    if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
        log "Stage 13: Scoring"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${st_exp}/${inference_tag}/${dset}"

            # TODO(jiatong): add asr scoring and inference

            _scoredir="${_dir}/score_bleu"
            mkdir -p "${_scoredir}"

            paste \
                <(<"${_data}/text.${tgt_case}.${tgt_lang}" \
                    ${python} -m espnet2.bin.tokenize_text  \
                        -f 2- --input - --output - \
                        --token_type word \
                        --non_linguistic_symbols "${nlsyms_txt}" \
                        --remove_non_linguistic_symbols true \
                        --cleaner "${cleaner}" \
                        ) \
                <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                    >"${_scoredir}/ref.trn.org"

            # NOTE(kamo): Don't use cleaner for hyp
            paste \
                <(<"${_dir}/text"  \
                        ${python} -m espnet2.bin.tokenize_text  \
                            -f 2- --input - --output - \
                            --token_type word \
                            --non_linguistic_symbols "${nlsyms_txt}" \
                            --remove_non_linguistic_symbols true \
                            ) \
                <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                    >"${_scoredir}/hyp.trn.org"

            # remove utterance id
            perl -pe 's/\([^\)]+\)$//g;' "${_scoredir}/ref.trn.org" > "${_scoredir}/ref.trn"
            perl -pe 's/\([^\)]+\)$//g;' "${_scoredir}/hyp.trn.org" > "${_scoredir}/hyp.trn"

            # detokenizer
            detokenizer.perl -l ${tgt_lang} -q < "${_scoredir}/ref.trn" > "${_scoredir}/ref.trn.detok"
            detokenizer.perl -l ${tgt_lang} -q < "${_scoredir}/hyp.trn" > "${_scoredir}/hyp.trn.detok"

            # rotate result files
            if [ ${tgt_case} = "tc" ]; then
                pyscripts/utils/rotate_logfile.py ${_scoredir}/result.tc.txt
            fi
            pyscripts/utils/rotate_logfile.py ${_scoredir}/result.lc.txt

            if [ ${tgt_case} = "tc" ]; then
                echo "Case sensitive BLEU result (single-reference)" > ${_scoredir}/result.tc.txt
                sacrebleu "${_scoredir}/ref.trn.detok" \
                          -i "${_scoredir}/hyp.trn.detok" \
                          -m bleu chrf ter \
                          >> ${_scoredir}/result.tc.txt

                log "Write a case-sensitive BLEU (single-reference) result in ${_scoredir}/result.tc.txt"
            fi

            # detokenize & remove punctuation except apostrophe
            scripts/utils/remove_punctuation.pl < "${_scoredir}/ref.trn.detok" > "${_scoredir}/ref.trn.detok.lc.rm"
            scripts/utils/remove_punctuation.pl < "${_scoredir}/hyp.trn.detok" > "${_scoredir}/hyp.trn.detok.lc.rm"
            echo "Case insensitive BLEU result (single-reference)" > ${_scoredir}/result.lc.txt
            sacrebleu -lc "${_scoredir}/ref.trn.detok.lc.rm" \
                      -i "${_scoredir}/hyp.trn.detok.lc.rm" \
                      -m bleu chrf ter \
                      >> ${_scoredir}/result.lc.txt
            log "Write a case-insensitve BLEU (single-reference) result in ${_scoredir}/result.lc.txt"

            # process multi-references cases
            multi_references=$(ls "${_data}/text.${tgt_case}.${tgt_lang}".* || echo "")
            if [ "${multi_references}" != "" ]; then
                case_sensitive_refs=""
                case_insensitive_refs=""
                for multi_reference in ${multi_references}; do
                    ref_idx="${multi_reference##*.}"
                    paste \
                        <(<${multi_reference} \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type word \
                                --non_linguistic_symbols "${nlsyms_txt}" \
                                --remove_non_linguistic_symbols true \
                                --cleaner "${cleaner}" \
                                ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn.org.${ref_idx}"

                    # remove utterance id
                    perl -pe 's/\([^\)]+\)$//g;' "${_scoredir}/ref.trn.org.${ref_idx}" > "${_scoredir}/ref.trn.${ref_idx}"
                    detokenizer.perl -l ${tgt_lang} -q < "${_scoredir}/ref.trn.${ref_idx}" > "${_scoredir}/ref.trn.detok.${ref_idx}"
                    scripts/utils/remove_punctuation.pl < "${_scoredir}/ref.trn.detok.${ref_idx}" > "${_scoredir}/ref.trn.detok.lc.rm.${ref_idx}"
                    case_sensitive_refs="${case_sensitive_refs} ${_scoredir}/ref.trn.detok.${ref_idx}"
                    case_insensitive_refs="${case_insensitive_refs} ${_scoredir}/ref.trn.detok.lc.rm.${ref_idx}"
                done

                if [ ${tgt_case} = "tc" ]; then
                    echo "Case sensitive BLEU result (multi-references)" >> ${_scoredir}/result.tc.txt
                    sacrebleu ${case_sensitive_refs} \
                        -i ${_scoredir}/hyp.trn.detok.lc.rm -m bleu chrf ter \
                        >> ${_scoredir}/result.tc.txt
                    log "Write a case-sensitve BLEU (multi-reference) result in ${_scoredir}/result.tc.txt"
                fi

                echo "Case insensitive BLEU result (multi-references)" >> ${_scoredir}/result.lc.txt
                sacrebleu -lc ${case_insensitive_refs} \
                    -i ${_scoredir}/hyp.trn.detok.lc.rm -m bleu chrf ter \
                    >> ${_scoredir}/result.lc.txt
                log "Write a case-insensitve BLEU (multi-reference) result in ${_scoredir}/result.lc.txt"
            fi

            # WER scoring for multi-decoder
            if [ -f "${_dir}/asr_text" ]; then
                _scoredir="${_dir}/score_wer"
                mkdir -p "${_scoredir}"

                # Tokenize text to word level
                paste \
                    <(<"${_data}/text" \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type word \
                                --non_linguistic_symbols "${nlsyms_txt}" \
                                --remove_non_linguistic_symbols true \
                                --cleaner "${cleaner}" \
                                ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/ref.trn"

                # NOTE(kamo): Don't use cleaner for hyp
                paste \
                    <(<"${_dir}/asr_text"  \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type word \
                                --non_linguistic_symbols "${nlsyms_txt}" \
                                --remove_non_linguistic_symbols true \
                                ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"

                sclite \
                    ${score_opts} \
                    -r "${_scoredir}/ref.trn" trn \
                    -h "${_scoredir}/hyp.trn" trn \
                    -i rm -o all stdout > "${_scoredir}/result.txt"

                log "Write WER result in ${_scoredir}/result.txt"
                grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
            fi
        done

        # Show results in Markdown syntax
        scripts/utils/show_translation_result.sh --case $tgt_case "${st_exp}" > "${st_exp}"/RESULTS.md
        cat "${st_exp}"/RESULTS.md
    fi
else
    log "Skip the evaluation stages"
fi


packed_model="${st_exp}/${st_exp##*/}_${inference_st_model%.*}.zip"
if ! "${skip_upload}"; then
    if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
        log "Stage 14: Pack model: ${packed_model}"

        _opts=
        if "${use_lm}"; then
            _opts+="--lm_train_config ${lm_exp}/config.yaml "
            _opts+="--lm_file ${lm_exp}/${inference_lm} "
            _opts+="--option ${lm_exp}/perplexity_test/ppl "
            _opts+="--option ${lm_exp}/images "
        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            _opts+="--option ${st_stats_dir}/train/feats_stats.npz "
        fi
        if [ "${tgt_token_type}" = bpe ]; then
            _opts+="--option ${tgt_bpemodel} "
            _opts+="--option ${src_bpemodel} "
        fi
        if [ "${nlsyms_txt}" != none ]; then
            _opts+="--option ${nlsyms_txt} "
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.pack st \
            --st_train_config "${st_exp}"/config.yaml \
            --st_model_file "${st_exp}"/"${inference_st_model}" \
            ${_opts} \
            --option "${st_exp}"/RESULTS.md \
            --option "${st_exp}"/RESULTS.md \
            --option "${st_exp}"/images \
            --outpath "${packed_model}"
    fi


    if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ]; then
        log "Stage 15: Upload model to Zenodo: ${packed_model}"

        # To upload your model, you need to do:
        #   1. Sign up to Zenodo: https://zenodo.org/
        #   2. Create access token: https://zenodo.org/account/settings/applications/tokens/new/
        #   3. Set your environment: % export ACCESS_TOKEN="<your token>"

        if command -v git &> /dev/null; then
            _creator_name="$(git config user.name)"
            _checkout="
git checkout $(git show -s --format=%H)"

        else
            _creator_name="$(whoami)"
            _checkout=""
        fi
        # /some/where/espnet/egs2/foo/st1/ -> foo/st1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/st1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # Generate description file
        cat << EOF > "${st_exp}"/description
This model was trained by ${_creator_name} using ${_task} recipe in <a href="https://github.com/espnet/espnet/">espnet</a>.
<p>&nbsp;</p>
<ul>
<li><strong>Python API</strong><pre><code class="language-python">See https://github.com/espnet/espnet_model_zoo</code></pre></li>
<li><strong>Evaluate in the recipe</strong><pre>
<code class="language-bash">git clone https://github.com/espnet/espnet
cd espnet${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
./run.sh --skip_data_prep false --skip_train true --download_model ${_model_name}</code>
</pre></li>
<li><strong>Results</strong><pre><code>$(cat "${st_exp}"/RESULTS.md)</code></pre></li>
<li><strong>ST config</strong><pre><code>$(cat "${st_exp}"/config.yaml)</code></pre></li>
<li><strong>LM config</strong><pre><code>$(if ${use_lm}; then cat "${lm_exp}"/config.yaml; else echo NONE; fi)</code></pre></li>
</ul>
EOF

        # NOTE(kamo): The model file is uploaded here, but not published yet.
        #   Please confirm your record at Zenodo and publish it by yourself.

        # shellcheck disable=SC2086
        espnet_model_zoo_upload \
            --file "${packed_model}" \
            --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${src_lang}_${tgt_lang}" \
            --description_file "${st_exp}"/description \
            --creator_name "${_creator_name}" \
            --license "CC-BY-4.0" \
            --use_sandbox false \
            --publish false
    fi
else
    log "Skip the uploading stages"
fi

if ! "${skip_upload_hf}"; then
    if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ]; then
        [ -z "${hf_repo}" ] && \
            log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace" && \
            exit 1
        log "Stage 16: Upload model to HuggingFace: ${hf_repo}"

        gitlfs=$(git lfs --version 2> /dev/null || true)
        [ -z "${gitlfs}" ] && \
            log "ERROR: You need to install git-lfs first" && \
            exit 1

        dir_repo=${expdir}/hf_${hf_repo//"/"/"_"}
        [ ! -d "${dir_repo}" ] && git clone https://huggingface.co/${hf_repo} ${dir_repo}

        if command -v git &> /dev/null; then
            _creator_name="$(git config user.name)"
            _checkout="git checkout $(git show -s --format=%H)"
        else
            _creator_name="$(whoami)"
            _checkout=""
        fi
        # /some/where/espnet/egs2/foo/asr1/ -> foo/asr1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/asr1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # copy files in ${dir_repo}
        unzip -o ${packed_model} -d ${dir_repo}
        # Generate description file
        # shellcheck disable=SC2034
        hf_task=speech-translation
        # shellcheck disable=SC2034
        espnet_task=ST
        # shellcheck disable=SC2034
        task_exp=${st_exp}
        eval "echo \"$(cat scripts/utils/TEMPLATE_HF_Readme.md)\"" > "${dir_repo}"/README.md

        this_folder=${PWD}
        cd ${dir_repo}
        if [ -n "$(git status --porcelain)" ]; then
            git add .
            git commit -m "Update model"
        fi
        git push
        cd ${this_folder}
    fi
else
    log "Skip the uploading to HuggingFace stage"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
