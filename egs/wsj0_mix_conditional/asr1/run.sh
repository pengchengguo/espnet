#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=true

# config files
preprocess_config=conf/no_preprocess.yaml  # use conf/specaug.yaml for data augmentation
train_config=conf/train_multispkr.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# multi-speaker asr related
num_spkrs=2         # number of speakers
use_spa=false       # speaker parallel attention

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
n_average=10 # use 1 for RNN models
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# data
wsj0=/home/backup_nfs2/pcguo/corpora/wsj/wsj0
wsj1=/home/backup_nfs2/pcguo/corpora/wsj/wsj1
wsj_full_wav=$PWD/data/wsj0/wsj0_wav
wsj_2mix_wav=$PWD/data/wsj0_mix/2speakers
wsj_3mix_wav=$PWD/data/wsj0_mix/3speakers
wsj_2mix_scripts=$PWD/data/wsj0_mix/scripts

# exp tag
tag="" # tag for managing experiments.

# for training
device=0

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="tr"
train_dev="cv"
recog_set="tt"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    echo "stage 0: Data preparation"
    ### This part is for WSJ0 mix
    ### Download mixture scripts and create mixtures for 2 speakers
    # local/wsj0_create_mixture.sh ${wsj_2mix_scripts} ${wsj0} ${wsj_full_wav} \
    #     ${wsj_2mix_wav} || exit 1;
    # local/wsj0_2mix_data_prep.sh ${wsj_2mix_wav}/wav16k/max ${wsj_2mix_scripts} \
    #     ${wsj_full_wav} || exit 1;

    local/wsj0_create_3mixture.sh ${wsj_2mix_scripts} ${wsj0} ${wsj_full_wav} \
        ${wsj_3mix_wav} || exit 1;
    local/wsj0_3mix_data_prep.sh ${wsj_3mix_wav}/wav16k/max ${wsj_2mix_scripts} \
        ${wsj_full_wav} || exit 1;

    ### Also need wsj corpus to prepare language information
    ### This is from Kaldi WSJ recipe
    local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/wsj_format_data.sh
    mkdir -p data/wsj
    mv data/{dev_dt_*,local,test_dev*,test_eval*,train_si284} data/wsj

    # prepare for wsj0 tr single spkr data
    mkdir -p data_wsj0/tr
    awk -v dir=${wsj_full_wav} '{split($1, lst1, "/"); utt1 = substr(lst1[4], 1, 8); split($3, lst2, "/"); utt2 = substr(lst2[4], 1, 8); 
        printf("%s %s/%s\n", utt1, dir, $1); printf("%s %s/%s\n", utt2, dir, $3);}' \
        ${wsj_2mix_scripts}/mix_2_spk_tr.txt | sort -u > data_wsj0/tr/wav.scp
    awk '{id=substr($1, 1, 3); print $1, id}' data_wsj0/tr/wav.scp > data_wsj0/tr/utt2spk
    utils/utt2spk_to_spk2utt.pl data_wsj0/tr/utt2spk > data_wsj0/tr/spk2utt
    awk '{split($1, lst, "_"); utt = lst[3]; for(i=2; i<=NF; i++) {utt = utt" "$i} print utt;}' \
        data_wsj0_2mix/tr/text_spk1 | sort -u > data_wsj0/tr/text_1
    awk '{split($1, lst, "_"); utt = lst[5]; for(i=2; i<=NF; i++) {utt = utt" "$i} print utt;}' \
        data_wsj0_2mix/tr/text_spk2 | sort -u > data_wsj0/tr/text_2
    cat data_wsj0/tr/text_1 data_wsj0/tr/text_2 | sort -u > data_wsj0/tr/text

    # prepare for wsj0 cv single spkr data
    mkdir -p data_wsj0/cv
    awk -v dir=${wsj_full_wav} '{split($1, lst1, "/"); utt1 = substr(lst1[4], 1, 8); split($3, lst2, "/"); utt2 = substr(lst2[4], 1, 8); 
        printf("%s %s/%s\n", utt1, dir, $1); printf("%s %s/%s\n", utt2, dir, $3);}' \
        ${wsj_2mix_scripts}/mix_2_spk_cv.txt | sort -u > data_wsj0/cv/wav.scp
    awk '{id=substr($1, 1, 3); print $1, id}' data_wsj0/cv/wav.scp > data_wsj0/cv/utt2spk
    utils/utt2spk_to_spk2utt.pl data_wsj0/cv/utt2spk > data_wsj0/cv/spk2utt
    awk '{split($1, lst, "_"); utt = lst[3]; for(i=2; i<=NF; i++) {utt = utt" "$i} print utt;}' \
        data_wsj0_2mix/cv/text_spk1 | sort -u > data_wsj0/cv/text_1
    awk '{split($1, lst, "_"); utt = lst[5]; for(i=2; i<=NF; i++) {utt = utt" "$i} print utt;}' \
        data_wsj0_2mix/cv/text_spk2 | sort -u > data_wsj0/cv/text_2
    cat data_wsj0/cv/text_1 data_wsj0/cv/text_2 | sort -u > data_wsj0/cv/text

    ### Or this part is for WSJ mix, which is a larger two-speaker mixture corpus created from WSJ corpus. Used in
    ### Seki H, Hori T, Watanabe S, et al. End-to-End Multi-Lingual Multi-Speaker Speech Recognition[J]. 2018. and
    ### Chang X, Qian Y, Yu K, et al. End-to-End Monaural Multi-speaker ASR System without Pretraining[J]. 2019
    ### Before next step, suppose wsj_2mix_corpus has been generated (please refer to wsj0_mixture for more details).
    # local/wsj_2mix_data_prep.sh ${wsj_2mix_wav}/wav16k/max ${wsj_2mix_script} || exit 1;
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation started @ `date`"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    # wsj0 data is manually generated from wsj0_2mix
    for data in wsj0 wsj0_2mix wsj0_3mix; do
        for x in tr cv tt; do
            if [ -d "./data_${data}/${x}" ]; then
                steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 60 --write_utt2num_frames true \
                    data_${data}/${x} exp/make_fbank_${data}/${x} ${fbankdir}_${data}
                utils/fix_data_dir.sh data_${data}/${x}
            fi
        done
    done

    # compute global CMVN
    for data in wsj0 wsj0_2mix wsj0_3mix; do
        compute-cmvn-stats scp:data_${data}/${train_set}/feats.scp data_${data}/${train_set}/cmvn.ark
    done

    # here the features of wj0 are only used to generate ground truth ctc alignments
    # use different \'delta\' because pre-trained wsj transformer use delta equals to false
    # you can change these setups to be same of the pre-trained wsj transformer
    # feat_tr_dir=${dumpdir}_wsj0/${train_set}/deltafalse; mkdir -p ${feat_tr_dir}
    # dump.sh --cmd "$train_cmd" --nj 60 --do_delta false \
    #     data_wsj0/${train_set}/feats.scp data_wsj0/${train_set}/cmvn.ark exp/dump_feats_wsj0/train ${feat_tr_dir}
    # feat_dt_dir=${dumpdir}_wsj0/${train_dev}/deltafalse; mkdir -p ${feat_dt_dir}
    # dump.sh --cmd "$train_cmd" --nj 60 --do_delta false \
    #     data_wsj0/${train_dev}/feats.scp data_wsj0/${train_set}/cmvn.ark exp/dump_feats_wsj0/dev ${feat_dt_dir}

    # different from our conditional chain recipes, we use the ctc alignments genereated before
    # and use \'delta\' to dump the wsj0 features, which means the wsj0 data will also be included
    # in the training
    feat_tr_dir=${dumpdir}_wsj0/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 60 --do_delta ${do_delta} \
        data_wsj0/${train_set}/feats.scp data_wsj0/${train_set}/cmvn.ark exp/dump_feats_wsj0/train ${feat_tr_dir}
    feat_dt_dir=${dumpdir}_wsj0/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj 60 --do_delta ${do_delta} \
        data_wsj0/${train_dev}/feats.scp data_wsj0/${train_set}/cmvn.ark exp/dump_feats_wsj0/dev ${feat_dt_dir} 

    # dump features for training
    for data in wsj0_2mix wsj0_3mix; do
        feat_tr_dir=${dumpdir}_${data}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
        dump.sh --cmd "$train_cmd" --nj 60 --do_delta ${do_delta} \
            data_${data}/${train_set}/feats.scp data_${data}/${train_set}/cmvn.ark exp/dump_feats_${data}/train ${feat_tr_dir}
        feat_dt_dir=${dumpdir}_${data}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
        dump.sh --cmd "$train_cmd" --nj 60 --do_delta ${do_delta} \
            data_${data}/${train_dev}/feats.scp data_${data}/${train_set}/cmvn.ark exp/dump_feats_${data}/dev ${feat_dt_dir}
        for rtask in ${recog_set}; do
            feat_recog_dir=${dumpdir}_${data}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
            dump.sh --cmd "$train_cmd" --nj 60 --do_delta ${do_delta} \
                data_${data}/${rtask}/feats.scp data_${data}/${train_set}/cmvn.ark exp/dump_feats_${data}/recog/${rtask} \
                ${feat_recog_dir}
        done
    done
    echo "stage 1: Done @ `date`"
fi

dict=data/lang_1char/${train_set}_units.txt
dict_wblank=data/lang_1char/${train_set}_units_wblank.txt
nlsyms=data/lang_1char/non_lang_syms.txt
wsj_train_set=wsj/train_si284
wsj_train_dev=wsj/test_dev93
wsj_train_test=wsj/test_eval92

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation started @ `date`"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${wsj_train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${wsj_train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}
    # add blank for dict, only use to convert CTC alignment into training units index format
    sed '1 i <blank> 0' ${dict} > ${dict_wblank}

    # different from our conditional chain recipes, we use the ctc alignments genereated before
    # echo "make json files for generate ctc alignment"
    # feat_tr_dir=${dumpdir}_wsj0/${train_set}/deltafalse
    # local/data2json.sh --cmd "${train_cmd}" --nj 60 \
    #     --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} --num-spkrs 1 \
    #     data_wsj0/${train_set} ${dict} > ${feat_tr_dir}/data.json
    # feat_dt_dir=${dumpdir}_wsj0/${train_dev}/deltafalse
    # local/data2json.sh --cmd "${train_cmd}" --nj 60 \
    #     --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} --num-spkrs 1 \
    #     data_wsj0/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    
    echo "stage 2: Done @ `date`"
fi

# It takes about one day. If you just want to do end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation started @ `date`"

    if [ ${use_wordlm} = true ]; then
        lmdatadir=data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}
        cut -f 2- -d" " data/${wsj_train_set}/text > ${lmdatadir}/train_trans.txt
        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
                | grep -v "<" | tr "[:lower:]" "[:upper:]" > ${lmdatadir}/train_others.txt
        cut -f 2- -d" " data/${wsj_train_dev}/text > ${lmdatadir}/valid.txt
        cut -f 2- -d" " data/${wsj_train_test}/text > ${lmdatadir}/test.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
        lmdatadir=data/local/lm_train
        lmdict=${dict}
        mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${wsj_train_set}/text \
            | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt
        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
            | grep -v "<" | tr "[:lower:]" "[:upper:]" \
            | text2token.py -n 1 | cut -f 2- -d" " > ${lmdatadir}/train_others.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${wsj_train_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${wsj_train_test}/text \
                | cut -f 2- -d" " > ${lmdatadir}/test.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
    fi

    ${tts_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir ${lmexpdir}/tensorboard \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --test-label ${lmdatadir}/test.txt \
        --resume ${lm_resume} \
        --dict ${lmdict}
    echo "stage 3: Done @ `date`"
fi

# here use a pre-trained joint ctc/attention model to generating CTC alignment
# the detail of the pre-trained joint ctc/attention model can refer to:
# https://github.com/espnet/espnet/blob/master/egs/wsj/asr1/run.sh
pre_trained=exp/pre_trained
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Generating CTC alignment started @ `date`"
    nj=60
    recog_model=model.last10.avg.best

    pids=() # initialize pids
    for rtask in ${train_set} ${train_dev}; do
    (
        decode_dir=align_wsj0_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        feat_recog_dir=${dumpdir}_wsj0/${rtask}/deltafalse

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${pre_trained}/${decode_dir}/log/decode.JOB.log \
            asr_ctc_align.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${pre_trained}/${decode_dir}/data.JOB.json \
            --model ${pre_trained}/results/${recog_model}  \
            ${recog_opts}

        concatjson.py ${pre_trained}/${decode_dir}/data.*.json > ${pre_trained}/${decode_dir}/data.json

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "stage 4: Done @ `date`"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Generating CTC alignment and data.json for wsj0_2mix started `date`"
    # for rtask in ${train_set} ${train_dev}; do
    #     decode_dir=align_wsj0_${rtask}_$(basename ${decode_config%.*})_${lmtag}
    #     local/wsj_mix_alignments_prep.py \
    #         --input-json ${pre_trained}/${decode_dir}/data.json \
    #         --utt2spk data_wsj0_2mix/${rtask}/utt2spk \
    #         -O data_wsj0_2mix/${rtask}/ctc_alignment_spk1 data_wsj0_2mix/${rtask}/ctc_alignment_spk2
    # done

    ## copy the alignments from NeurIPS's Experiments
    ## CLSP: /export/c05/xkc09/asr/wsj0_2mix_conditional
    feat_tr_dir=${dumpdir}_wsj0_2mix/${train_set}/delta${do_delta}
    local/data2json.sh --cmd "${train_cmd}" --nj 60 \
        --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} --num-spkrs 2 \
        data_wsj0_2mix/${train_set} ${dict_wblank} > ${feat_tr_dir}/data.json
    feat_dt_dir=${dumpdir}_wsj0_2mix/${train_dev}/delta${do_delta}
    local/data2json.sh --cmd "${train_cmd}" --nj 60 \
        --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} --num-spkrs 2 \
        data_wsj0_2mix/${train_dev} ${dict_wblank} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}_wsj0_2mix/${rtask}/delta${do_delta}
        local/data2json.sh --cmd "${train_cmd}" --nj 60 \
        --feat ${feat_recog_dir}/feats.scp --nlsyms ${nlsyms} --num-spkrs 2 \
        data_wsj0_2mix/${rtask} ${dict_wblank} > ${feat_recog_dir}/data.json
    done
    echo "stage 5: Done @ `date`"
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
${use_spa} && spa=true
expdir=exp_wsj0_mix/${expname}
mkdir -p ${expdir}

feat_tr_dir=${dumpdir}_wsj0_2mix/${train_set}/delta${do_delta}
feat_dt_dir=${dumpdir}_wsj0_2mix/${train_dev}/delta${do_delta}
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Network Training started @ `date`"

    # ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    # CUDA_VISIBLE_DEVICES=${device} ${local_cmd} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=${device}\
        asr_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir ${expdir}/tensorboard \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --num-spkrs ${num_spkrs} \
        ${spa:+--spa}
    
    echo "stage 6: Done @ `date`"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Decoding started @ `date`"
    nj=32
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
        recog_model=model.last${n_average}.avg.best
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${recog_model} \
                               --num ${n_average}
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        feat_recog_dir=${dumpdir}_wsj0_2mix/${rtask}/delta${do_delta}

        # split data
        # splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --num-spkrs ${num_spkrs} \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            ${recog_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} --num_spkrs ${num_spkrs} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "stage 7: Done @ `date`"
fi
