#!/bin/bash

## HOW TO USE SBATCH : DOCS
## https://github.com/dasandata/Open_HPC/blob/master/Document/User%20Guide/5_use_resource/5.2_Allocate_Resource.md

## HOW TO USE SBATCH : QUICK RUN
## gpu 4개를 사용할 수 있는 리소스(node)에, 사용 시간은 48시간으로 지정하여 작업을 제출(submit)
## --time=일-시간:분:초
# sbatch --gres=gpu:4 --time=48:00:00 ./job_test_summarization.sh

## sbatch 돌아가고 있는 상태 확인
# squeuelong -u ysnamgoong42

#echo "### START DATE=\$(date)"
#echo "### HOSTNAME=\$(hostname)"
#echo "### CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"

# conda 환경 활성화.
source ~/.bashrc
conda activate xlcost

# cuda 11.7 환경 구성.
ml purge
ml load cuda/11.7

# GPU 체크
nvidia-smi
nvcc -V

# 활성화된 환경에서 코드 실행.

#python run.py --do_train --do_eval --model_type codet5 --config_name Salesforce/codet5-base --tokenizer_name Salesforce/codet5-base --model_name_or_path Salesforce/codet5-base --train_filename /home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/train-Python-desc-tok.py,/home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/train-Python-desc-tok.txt --dev_filename /home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/val-Python-desc-tok.py,/home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/val-Python-desc-tok.txt --output_dir /home/ysnamgoong42/ws/XLCoST/codet5_pl_nl_program/Python-desc --max_source_length 400 --max_target_length 100 --num_train_epochs 10 --train_steps 5000 --eval_steps 2500 --train_batch_size 16 --eval_batch_size 16 --beam_size 5 --learning_rate 5e-5

#python run.py --do_test --model_type codet5 --config_name Salesforce/codet5-base --tokenizer_name Salesforce/codet5-base --model_name_or_path Salesforce/codet5-base --load_model_path /home/ysnamgoong42/ws/XLCoST/codet5_pl_nl_program/Python-desc/checkpoint-best-ppl/pytorch_model.bin --test_filename /home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/test-Python-desc-tok.py,/home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/test-Python-desc-tok.txt --output_dir /home/ysnamgoong42/ws/XLCoST/codet5_pl_nl_program/Python-desc --max_source_length 400 --max_target_length 100 --beam_size 5 --eval_batch_size 16

bash run_NL_PL.sh 0 python desc program codet5 eval

# slurm-298564.out : python run.py --do_train ... 과 python run.py --do_test ... 를 돌린 결과 / train batch size 16, eval batch size 16 으로 설정함.
# slurm-300280.out : bash run_NL_PL.sh 0 python desc program codet5 eval 를 돌린 결과 / eval batch size 16 으로 설정함. / evaluator.py랑 calc_code_bleu.py 실행시 arg 이상하게들어간듯. 여기서 에러발생