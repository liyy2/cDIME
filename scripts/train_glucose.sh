#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
model_name=TimeLLM
train_epochs=10
learning_rate=0.001
llama_layers=32

master_port=8889
num_process=2
batch_size=2
d_model=16
d_ff=32

comment='TimeLLM-ECL'

eval "$(conda shell.bash hook)"
conda activate Time-LLM

python run_pl.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path /home/yl2428/Time-LLM/dataset/glucose \
  --data_path_pretrain combined_data.csv \
  --data_path combined_data.csv \
  --model_id ETTh1_ETTh2_512_96 \
  --model $model_name \
  --data_pretrain Glucose \
  --data Glucose \
  --features S \
  --seq_len 48 \
  --label_len 12 \
  --pred_len 12 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs 20 \
  --enable_covariates 1 \
  --model_comment $comment