$CUDA_VISIBLE_DEVICES=2 ./run_joint_asr_and_sl_from_pretrained_with_bert_noisy_efs2.sh --stage 4 --stop_stage 8 --ngpu 1 --train_config conf/train-bert-noise-exp2.yaml --lm_config conf/lm4.yaml --decode_config conf/decode-bert.yaml --seed 20 --efs /efs/efs/clai24/



CUDA_VISIBLE_DEVICES=4 ./run_joint_asr_and_sl_from_pretrained_with_bert_efs_1_stage_noisy2.sh --stage 4 --stop_stage 8 --ngpu 1 --train_config conf/train-bert-1-stage-noise-exp2.yaml --lm_config conf/lm4.yaml --decode_config conf/decode-bert-1-stage.yaml --efs /efs/exp/clai24 --seed 12


CUDA_VISIBLE_DEVICES=5 ./run_joint_asr_and_sl_from_pretrained_with_bert_efs_1_stage_noisy2.sh --stage 4 --stop_stage 8 --ngpu 1 --train_config conf/train-bert-1-stage-noise-exp3.yaml --lm_config conf/lm4.yaml --decode_config conf/decode-bert-1-stage.yaml --efs /efs/exp/clai24 --seed 126

CUDA_VISIBLE_DEVICES=7 ./run_fine-tune_atis_from_pretrained_efs_noisy2.sh --stage 4 --stop_stage 8 --ngpu 1 --train_config conf/train2-noise-exp3.yaml --lm_config conf/lm4.yaml --decode_config conf/decode2-exp3.yaml --efs /efs/exp/clai24 --seed 392

CUDA_VISIBLE_DEVICES=6 ./run_fine-tune_atis_from_pretrained_efs_noisy2.sh --stage 4 --stop_stage 8 --ngpu 1 --train_config conf/train2-noise-exp2.yaml --lm_config conf/lm4.yaml --decode_config conf/decode2-exp2.yaml --efs /efs/exp/clai24 --seed 391

CUDA_VISIBLE_DEVICES=0 ./run_fine-tune_atis_from_pretrained_efs_noisy2.sh --stage 4 --stop_stage 8 --ngpu 1 --train_config conf/train2-noise-exp1.yaml --lm_config conf/lm4.yaml --decode_config conf/decode2-exp1.yaml --efs /efs/exp/clai24 --seed 390


CUDA_VISIBLE_DEVICES=0 ./run_joint_asr_and_sl_from_pretrained_with_bert_noisy_efs2.sh --stage 4 --stop_stage 8 --ngpu 1 --train_config conf/train-bert-noise-noise-exp1.yaml --lm_config conf/lm4.yaml --decode_config conf/decode-bert.yaml --efs /efs/exp/clai24 --seed 1390
CUDA_VISIBLE_DEVICES=1 ./run_joint_asr_and_sl_from_pretrained_with_bert_noisy_efs2.sh --stage 4 --stop_stage 8 --ngpu 1 --train_config conf/train-bert-noise-noise-exp2.yaml --lm_config conf/lm4.yaml --decode_config conf/decode-bert.yaml --efs /efs/exp/clai24 --seed 1391
CUDA_VISIBLE_DEVICES=2 ./run_joint_asr_and_sl_from_pretrained_with_bert_noisy_efs2.sh --stage 4 --stop_stage 8 --ngpu 1 --train_config conf/train-bert-noise-noise-exp3.yaml --lm_config conf/lm4.yaml --decode_config conf/decode-bert.yaml --efs /efs/exp/clai24 --seed 1392


CUDA_VISIBLE_DEVICES=0 ./run_train_atis_from_mlm_efs.sh --stage 4 --stop_stage 4 --ngpu 1 --train_config conf/train-mlm.yaml --lm_config conf/lm4.yaml --decode_config conf/decode-bert.yaml --efs /efs/exp/clai24 --seed 1378
