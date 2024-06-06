#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100
#SBATCH --mem=400G
#SBATCH --cpus-per-task=64
model_name=TimeLLM
learning_rate=0.001
llama_layers=32

master_port=8889
batch_size=32
d_model=16
d_ff=32

comment='TimeLLM-ECL'

eval "$(conda shell.bash hook)"
conda activate Time-LLM

python run_pl.py \
  --task_name short_term_forecast \
  --num_nodes 4 \
  --is_training 1 \
  --precision bf16 \
  --root_path /home/yl2428/Time-LLM/dataset/glucose \
  --data_path_pretrain combined_data.csv \
  --data_path combined_data.csv \
  --model_id TimeLLM \
  --model $model_name \
  --data_pretrain Glucose \
  --features MS \
  --seq_len 48 \
  --label_len 12 \
  --pred_len 12 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs 100 \
  --enable_covariates 1 \
  --model_comment $comment \
  --use_deep_speed 0 \