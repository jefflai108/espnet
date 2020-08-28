#!/bin/bash
. path.sh

stage=1

if [ $stage -eq 0 ]; then
# decode single wav file with pretrained models
    CUDA_VISIBLE_DEVICES=1 recog_wav.sh \
        --cmvn data/train_960/cmvn.ark \
        --lang_model model.v1.transformerlm.v1/exp/train_rnnlm_pytorch_lm_transformer_cosine_batchsize32_lr1e-4_layer16_unigram5000_ngpu4/rnnlm.model.best \
        --recog_model model.v1.transformerlm.v1/librispeech.transformer.v1/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best \
        --decode_config model.v1.transformerlm.v1/conf/decode_simple.yaml \
        --ngpu 1 \
        --stage 0 --stop_stage 100 \
        /home/ubuntu/data/atis3/17_2.1/atis3/sp_trn/mit/8ky/4/8ky014ss.wav-2

## results:
# GT: LIST FLIGHTS FROM CINCINNATI TO SAN JOSE FRIDAY EVENING
# 4m58.616s -- model.v1.transformerlm.v1/conf/decode.yaml; decode: ▁LISZT▁FLIGHTS▁FROM▁CINCINNATI▁TO▁SAN▁JOSE▁FRIDAY▁EVENING
# 0m46.437s --  model.v1.transformerlm.v1/conf/decode_simple.yaml; decode: ▁LISZT▁FLIGHTS▁FROM▁CINCINNATI▁TO▁SAN▁JOSE▁FRIDAY▁EVENING
fi

if [ $stage -eq 1 ]; then
    #for data_dir in atis.test atis.train atis.valid; do
    #for data_dir in atis.test2 atis.train2 atis.valid2; do
    for data_dir in atis.test3; do
        #CUDA_VISIBLE_DEVICES=1
        recog_wav_2.sh \
            --cmvn data/train_960/cmvn.ark \
            --lang_model model.v1.transformerlm.v1/exp/train_rnnlm_pytorch_lm_transformer_cosine_batchsize32_lr1e-4_layer16_unigram5000_ngpu4/rnnlm.model.best \
            --recog_model model.v1.transformerlm.v1/librispeech.transformer.v1/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best \
            --decode_config model.v1.transformerlm.v1/conf/decode_simple.yaml \
            --ngpu 0 \
            --nj 60 \
            --stage 4 --stop_stage 4 \
            /home/ubuntu/tools/atis-mapping/data/${data_dir}/
    done
fi
