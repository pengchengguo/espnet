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

Options:
    --no_overlap (bool): Whether to ignore the overlapping utterance in the training set.
EOF
)

SECONDS=0
no_overlap=true

log "$0 $*"


. ./utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -gt 1 ]; then
  log "${help_message}"
  exit 2
fi

if [ -z "${AISHELL4}" ]; then
  log "Error: \$AISHELL4 is not set in db.sh."
  exit 2
fi

if [ ! -d "${AISHELL4}" ]; then
  log "Error: ${AISHELL4} is empty."
  exit 2
fi

# To absolute path
AISHELL4=$(cd ${AISHELL4}; pwd)
aishell4_audio_dir=${AISHELL4}/wav
aishell4_text_dir=${AISHELL4}/textgrid

log "Aishell4 Data Preparation"
train_dir=data/local/aishell4_train
test_dir=data/local/aishell4_test
tmp_dir=data/local/tmp

mkdir -p $train_dir
mkdir -p $test_dir
mkdir -p $tmp_dir

# find wav audio files
find -L $aishell4_audio_dir -iname "*.wav" > $tmp_dir/wav.flist
n=$(wc -l < $tmp_dir/wav.flist)
[ $n -ne 211 ] && log Warning: expected 211 data data files, found $n

grep -i "wav/train" $tmp_dir/wav.flist > $train_dir/wav.flist || exit 1;
grep -i "wav/test" $tmp_dir/wav.flist > $test_dir/wav.flist || exit 1;

# find textgrid files
find -L $aishell4_text_dir -iname "*.textgrid" > $tmp_dir/textgrid.flist
n=$(wc -l < $tmp_dir/textgrid.flist)
[ $n -ne 211 ] && log Warning: expected 211 data data files, found $n

grep -i "textgrid/train" $tmp_dir/textgrid.flist > $train_dir/textgrid.flist || exit 1;
grep -i "textgrid/test" $tmp_dir/textgrid.flist > $test_dir/textgrid.flist || exit 1;

rm -r $tmp_dir

# transcriptions preparation
# training set
sed -e 's/\.wav//' $train_dir/wav.flist | awk -F '/' '{print $NF}' > $train_dir/utt.list
paste -d' ' $train_dir/utt.list $train_dir/wav.flist > $train_dir/wav_all.scp
cat $train_dir/wav_all.scp | awk '{printf("%s sox -R -t wav %s -c 1 -t wav - |\n", $1, $2)}' | \
  sort -u > $train_dir/wav.scp
python local/aishell4_process_textgrid.py --path $train_dir --no-overlap $no_overlap
cat $train_dir/text_all | local/text_normalize.pl | local/text_format.pl | sort -u > $train_dir/text
utils/filter_scp.pl -f 1 $train_dir/text $train_dir/utt2spk_all | sort -u > $train_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $train_dir/utt2spk > $train_dir/spk2utt
utils/filter_scp.pl -f 1 $train_dir/text $train_dir/segments_all | sort -u > $train_dir/segments

# test set
sed -e 's/\.wav//' $test_dir/wav.flist | awk -F '/' '{print $NF}' > $test_dir/utt.list
paste -d' ' $test_dir/utt.list $test_dir/wav.flist > $test_dir/wav_all.scp
cat $test_dir/wav_all.scp | awk '{printf("%s sox -R -t wav %s -c 1 -t wav - |\n", $1, $2)}' | \
  sort -u > $test_dir/wav.scp
python local/aishell4_process_textgrid.py --path $test_dir --no-overlap false
cat $test_dir/text_all | local/text_normalize.pl | local/text_format.pl | sort -u > $test_dir/text
utils/filter_scp.pl -f 1 $test_dir/text $test_dir/utt2spk_all | sort -u > $test_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $test_dir/utt2spk > $test_dir/spk2utt
utils/filter_scp.pl -f 1 $test_dir/text $test_dir/segments_all | sort -u > $test_dir/segments

utils/copy_data_dir.sh --utt-prefix aishell4- --spk-prefix aishell4- \
  $train_dir data/aishell4_train
utils/copy_data_dir.sh --utt-prefix aishell4- --spk-prefix aishell4- \
  $test_dir data/aishell4_test

# remove space in text
for x in aishell4_train aishell4_test; do
  cp data/${x}/text data/${x}/text.org
  paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
      > data/${x}/text
  rm data/${x}/text.org
done

log "Successfully finished. [elapsed=${SECONDS}s]"
