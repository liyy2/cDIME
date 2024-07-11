model_name=DLinearChannelMix
learning_rate=0.004
llama_layers=32

master_port=8889
batch_size=64
d_model=16
d_ff=32

comment='TimeLLM-ECL'

python run_pl.py \
  --task_name short_term_forecast \
  --num_nodes 1 \
  --is_training 1 \
  --precision bf16 \
  --root_path /home/yl2428/Time-LLM/dataset/glucose \
  --data_path_pretrain combined_data.csv \
  --data_path combined_data.csv \
  --model_id DLinear \
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
  --num_heads 6 \
  --model_comment $comment \
  --use_deep_speed 1 \