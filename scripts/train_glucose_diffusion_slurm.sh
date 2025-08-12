#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

model_name=ns_Transformer
learning_rate=0.00005
llama_layers=32
master_port=8889
batch_size=48
d_model=32
d_ff=256
comment='TimeLLM-ECL'

eval "$(conda shell.bash hook)"
conda activate cDIME

CUDA_VISIBLE_DEVICES=0 python run_pl_diffusion.py \
  --task_name long_term_forecast \
  --num_nodes 1 \
  --is_training 1 \
  --precision 32 \
  --root_path /home/yl2428/Time-LLM/dataset/glucose \
  --data_path_pretrain output_Junt_16_3.csv\
  --data_path output_Junt_16_3.csv \
  --model_id ETTh1_ETTh2_512_192 \
  --model $model_name \
  --data_pretrain Glucose \
  --features MS \
  --seq_len 72 \
  --label_len 32 \
  --pred_len 48 \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs 100 \
  --enable_covariates 1 \
  --num_individuals -1 \
  --use_moe 1 \
  --num_experts 8 \
  --top_k_experts 4 \
  --moe_loss_weight 0.01 \
  --log_routing_stats 1 \
  --stride 1 \
  --use_deep_speed 1  \
  --enable_context_aware 1 \
  --k_z 1e-3 \
  --k_cond 1e-3 \
  --patience 40 