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

help_message=$(cat << EOF
Usage: $0

EOF
)
SECONDS=0

log "$0 $*"


. ./utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -gt 1 ]; then
    log "${help_message}"
    exit 2
fi

if [ -z "${MUSAN}" ]; then
    log "Error: \$MUSAN is not set in db.sh."
    exit 2
fi

# To absolute path
MUSAN=$(cd ${MUSAN}; pwd)

log "Data Preparation"
# Prepare the MUSAN corpus, which consists of music, speech, and noise
# suitable for augmentation.
steps/data/make_musan.sh ${MUSAN} data

# Regard free-sound noises as matched type
grep "free-sound" data/musan_noise/wav.scp > data/musan_noise/free-sound.scp
utils/subset_data_dir.sh --utt-list data/musan_noise/free-sound.scp data/musan_noise \
    data/musan_noise_match

# Regard sound-bible noises as unmatched type
grep "sound-bible" data/musan_noise/wav.scp > data/musan_noise/sound-bible.scp
utils/subset_data_dir.sh --utt-list data/musan_noise/sound-bible.scp data/musan_noise \
    data/musan_noise_unmatch

log "Successfully finished. [elapsed=${SECONDS}s]"
