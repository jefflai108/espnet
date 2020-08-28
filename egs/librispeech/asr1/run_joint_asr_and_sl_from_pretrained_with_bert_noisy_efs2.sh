#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=-1
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=30
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train2.yaml # current default recipe requires 4 gpus.
                             # if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
lm_config=conf/lm.yaml
decode_config=conf/decode_simple.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# bpemode (unigram or bpe)
nbpe=1000
bpemode=bpe

# exp tag
tag="" # tag for managing experiments.

# storage
efs=""

# training and recognition seed
seed=1

. utils/parse_options.sh || exit 1;

echo $seed
echo $efs
echo $decode_config
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=atis.train_noise
train_dev=atis.valid_noise
recog_set="atis.test3_noise"

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in atis.train atis.valid atis.test; do
        utils/fix_data_dir.sh data/${x}
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    utils/combine_data.sh --extra_files utt2num_frames data/${train_set}_org data/atis.train
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev}_org data/atis.valid

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_org data/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_org data/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_char/atis.train_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/atis.train_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt
    spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi

# You can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train_${bpemode}${nbpe}
    if [ ! -e ${lmdatadir} ]; then
        mkdir -p ${lmdatadir}
        cut -f 2- -d" " data/${train_set}/text | gzip -c > data/local/lm_train/${train_set}_text.gz
        # combine external text and transcriptions and shuffle them with seed 777
        #zcat data/local/lm_train/librispeech-lm-norm.txt.gz data/local/lm_train/${train_set}_text.gz |\
        zcat data/local/lm_train/${train_set}_text.gz |\
            spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
        cut -f 2- -d" " data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
                                                            > ${lmdatadir}/valid.txt
    fi
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict} \
        --dump-hdf5-path ${lmdatadir}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=${efs}/exp/${expname}
mkdir -p ${expdir}
echo $expdir

###### modified for Bert ######
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    task_slot_filling=NN_crf # NN, NN_crf
    task_intent_detection=CLS_decoder # CLS, max, CLS_max
    balance_weight=0.5
    pretrained_model_name=bert-base-uncased #bert-large-uncased-whole-word-masking #bert-base-uncased
    batch_size=32 # 16, 32

    #optimizer=bertadam #bertadam, adamw, adam, sgd
    #learning_rate=5e-5 # 1e-5, 5e-5, 1e-4, 1e-3
    #max_norm_of_gradient_clip=1 # working for adamw, adam, sgd
    dropout_rate=0.1 # 0.1, 0.5
    #max_epoch=30
    #experiment_output_path=exp

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train_bert.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}_slots2.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}_slots2.json \
        --enc-init /efs/exp/clai24//exp/atis.train_noise_pytorch_train2-noise-exp2_specaug/results/model.val5.avg.best \
        --dec-init /efs/exp/clai24//exp/atis.train_noise_pytorch_train2-noise-exp2_specaug/results/model.val5.avg.best \
        --enc-init-mods encoder \
        --dec-init-mods decoder \
        --report-interval-iters 50 \
        --report-cer --report-wer \
        --seed $seed \
        --slots_task_st $task_slot_filling --slots_task_sc $task_intent_detection \
        --slots_dropout $dropout_rate --slots_batchSize $batch_size \
        --slots_st_weight ${balance_weight} --slots_pretrained_model_name ${pretrained_model_name} \
        --slots_fuse_asr_decoder_with_bert

        # initialize from /efs/exp/clai24//exp/atis.train_noise_pytorch_train2-noise-exp2_specaug/
        #--enc-init /efs/exp/clai24//exp/atis.train_noise_pytorch_train2-noise-exp2_specaug/results/model.val5.avg.best \
        #--dec-init /efs/exp/clai24//exp/atis.train_noise_pytorch_train2-noise-exp2_specaug/results/model.val5.avg.best \

        # initialize from /efs/exp/clai24/exp/atis.train_pytorch_fine-tune-train2
        #--enc-init /efs/exp/clai24/exp/atis.train_pytorch_fine-tune-train2/results/model.val5.avg.best \
        #--dec-init /efs/exp/clai24/exp/atis.train_pytorch_fine-tune-train2/results/model.val5.avg.best \

        ## asr pretrained only
        #--enc-init model.v1.transformerlm.v1/librispeech.transformer.v1/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best \
        #--dec-init model.v1.transformerlm.v1/librispeech.transformer.v1/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best \
        #--dec-init-mods decoder.decoders.yy \
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding ASR"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if [ ${lm_n_average} -eq 0 ]; then
            lang_model=rnnlm.model.best
        else
            if ${use_lm_valbest_average}; then
                lang_model=rnnlm.val${lm_n_average}.avg.best
                opt="--log ${lmexpdir}/log"
            else
                lang_model=rnnlm.last${lm_n_average}.avg.best
                opt="--log"
            fi
            average_checkpoints.py \
                ${opt} \
                --backend ${backend} \
                --snapshots ${lmexpdir}/snapshot.ep.* \
                --out ${lmexpdir}/${lang_model} \
                --num ${lm_n_average}
        fi
    fi

    # slots related
    task_slot_filling=NN_crf # NN, NN_crf
    task_intent_detection=CLS_decoder # CLS, max, CLS_max
    balance_weight=0.5
    pretrained_model_name=bert-base-uncased #bert-large-uncased-whole-word-masking #bert-base-uncased
    batch_size=32 # 16, 32
    dropout_rate=0.1 # 0.1, 0.5

    for rtask in ${recog_set}; do
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}_slots2.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog_bert.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}_slots2.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/${lang_model} \
            --seed $seed \
            --slots_testing \
            --slots_task_st $task_slot_filling --slots_task_sc $task_intent_detection \
            --slots_dropout $dropout_rate --slots_batchSize $batch_size \
            --slots_st_weight ${balance_weight} --slots_pretrained_model_name ${pretrained_model_name} \
            --slots_fuse_asr_decoder_with_bert

            #--api v2
    done
fi

if [ ${stage} -le 6  ] && [ ${stop_stage} -ge 6  ]; then

    for rtask in ${recog_set}; do

        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi

        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        #score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

        concatjson.py ${expdir}/${decode_dir}/data.*.json > ${expdir}/${decode_dir}/data.json

        recog_text=$(grep rec_text ${expdir}/${decode_dir}/data.json | sed -e 's/.*: "\(.*\)".*/\1/' | sed -e 's/<eos>//')
        echo "Recognized text: ${recog_text}"
        echo ""

        json2trn_wo_dict.py ${expdir}/${decode_dir}/data.json --num-spkrs 1 --refs ${expdir}/${decode_dir}/ref_org.wrd.trn --hyps ${expdir}/${decode_dir}/hyp_org.wrd.trn
        #json2trn.py ${expdir}/${decode_dir}/data.json ${dict} --num-spkrs 1 --refs ${expdir}/${decode_dir}/ref_org.wrd.trn --hyps ${expdir}/${decode_dir}/hyp_org.wrd.trn

        cat ${expdir}/${decode_dir}/hyp_org.wrd.trn | sed -e 's/▁//' | sed -e 's/▁/ /g' > ${expdir}/${decode_dir}/hyp.wrd.trn
        cat ${expdir}/${decode_dir}/ref_org.wrd.trn | sed -e 's/\.//g' -e 's/\,//g' > ${expdir}/${decode_dir}/ref.wrd.trn

        cat ${expdir}/${decode_dir}/hyp.wrd.trn | awk -v FS='' '{a=0;for(i=1;i<=NF;i++){if($i=="("){a=1};if(a==0){printf("%s ",$i)}else{printf("%s",$i)}}printf("\n")}' > ${expdir}/${decode_dir}/hyp.trn
        cat ${expdir}/${decode_dir}/ref.wrd.trn | awk -v FS='' '{a=0;for(i=1;i<=NF;i++){if($i=="("){a=1};if(a==0){printf("%s ",$i)}else{printf("%s",$i)}}printf("\n")}' > ${expdir}/${decode_dir}/ref.trn

        sclite -r ${expdir}/${decode_dir}/ref.trn trn -h ${expdir}/${decode_dir}/hyp.trn -i rm -o all stdout > ${expdir}/${decode_dir}/results.md
        echo "write a CER result in ${expdir}/${decode_dir}/results.md"
        grep -e Avg -e SPKR -m 2 ${expdir}/${decode_dir}/results.md

        sclite -r ${expdir}/${decode_dir}/ref.wrd.trn trn -h ${expdir}/${decode_dir}/hyp.wrd.trn -i rm -o all stdout > ${expdir}/${decode_dir}/results.wrd.md
        echo "write a WER result in ${expdir}/${decode_dir}/results.wrd.md"
        grep -e Avg -e SPKR -m 2 ${expdir}/${decode_dir}/results.wrd.md

        sclite -r ${expdir}/${decode_dir}/ref_org.wrd.trn trn -h ${expdir}/${decode_dir}/hyp.wrd.trn trn -i rm -o all stdout > ${expdir}/${decode_dir}/results_w_punc.wrd.md
        echo "write a WER result in ${expdir}/${decode_dir}/results_w_punc.wrd.md"
        grep -e Avg -e SPKR -m 2 ${expdir}/${decode_dir}/results_w_punc.wrd.md
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Decoding BERT"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if [ ${lm_n_average} -eq 0 ]; then
            lang_model=rnnlm.model.best
        else
            if ${use_lm_valbest_average}; then
                lang_model=rnnlm.val${lm_n_average}.avg.best
                opt="--log ${lmexpdir}/log"
            else
                lang_model=rnnlm.last${lm_n_average}.avg.best
                opt="--log"
            fi
            average_checkpoints.py \
                ${opt} \
                --backend ${backend} \
                --snapshots ${lmexpdir}/snapshot.ep.* \
                --out ${lmexpdir}/${lang_model} \
                --num ${lm_n_average}
        fi
    fi

    # slots related
    task_slot_filling=NN_crf # NN, NN_crf
    task_intent_detection=CLS_decoder # CLS, max, CLS_max
    balance_weight=0.5
    pretrained_model_name=bert-base-uncased #bert-large-uncased-whole-word-masking #bert-base-uncased
    batch_size=32 # 16, 32
    dropout_rate=0.1 # 0.1, 0.5

    for rtask in ${recog_set}; do
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        echo ${expdir}/${decode_dir}/data.json
        cp ${expdir}/${decode_dir}/data.json ${feat_recog_dir}/data_for_bert.json
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_for_bert.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode_bert.JOB.log \
            asr_recog_bert2.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_for_bert.JOB.json \
            --result-label ${expdir}/${decode_dir}/data_for_bert.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/${lang_model} \
            --seed $seed \
            --slots_testing \
            --slots_task_st $task_slot_filling --slots_task_sc $task_intent_detection \
            --slots_dropout $dropout_rate --slots_batchSize $batch_size \
            --slots_st_weight ${balance_weight} --slots_pretrained_model_name ${pretrained_model_name} \
            --slots_fuse_asr_decoder_with_bert \
            --output_path ${expdir}/${decode_dir}/hypothesis_slots_prediction.JOB.txt

            #--api v2
    done
    cat ${expdir}/${decode_dir}/hypothesis_slots_prediction.*.txt > ${expdir}/${decode_dir}/hypothesis_slots_prediction.txt
fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Scoring"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
    fi

    for rtask in ${recog_set}; do
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        slots_home=/home/ubuntu/tools/slot_filling_and_intent_detection_of_SLU/data/atis-2
        mkdir -p ${slots_home}/${expdir}/${decode_dir}
        cp ${expdir}/${decode_dir}/hypothesis_slots_prediction.txt ${slots_home}/${expdir}/${decode_dir}/hypothesis_slots_prediction.txt
        python ${slots_home}/generate_scoring_format.py \
                ${slots_home}/${expdir}/${decode_dir}/hypothesis_slots_prediction.txt \
                ${slots_home}/${expdir}/${decode_dir}/hypothesis_slots_prediction-scoring.txt
        python src/compute_extended_f1.py ${slots_home}/${expdir}/${decode_dir}/hypothesis_slots_prediction-scoring.txt
    done
fi
