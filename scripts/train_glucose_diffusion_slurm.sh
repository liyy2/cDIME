#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a100

model_name=ns_Transformer
learning_rate=0.0004
llama_layers=32
master_port=8889
batch_size=256
d_model=32
d_ff=128
comment='TimeLLM-ECL'

eval "$(conda shell.bash hook)"
conda activate Time-LLM

python run_pl_diffusion.py \
  --task_name long_term_forecast \
  --num_nodes 4 \
  --is_training 1 \
  --precision 32 \
  --root_path /home/yl2428/Time-LLM/dataset/glucose \
  --data_path_pretrain combined_data_Jun_28.csv\
  --data_path combined_data_Jun_28.csv \
  --model_id ETTh1_ETTh2_512_192 \
  --model $model_name \
  --data_pretrain Glucose \
  --features M \
  --seq_len 128 \
  --label_len 12 \
  --pred_len 32 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs 100 \
  --enable_covariates 1 \
  --num_individuals -1