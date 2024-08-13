config = {
    'num_nodes': 1,
    'task_name': 'long_term_forecast',
    'is_training': 1,
    'model_id': 'test',
    'model_comment': 'none',
    'model': 'Autoformer',
    'precision': '32',
    
    # Data loader
    'data_pretrain': 'ETTm1',
    'root_path': '/home/yl2428/Time-LLM/dataset',
    'data_path': 'ETTh1.csv',
    'data_path_pretrain': 'ETTh1.csv',
    'features': 'M',
    'target': 'OT',
    'loader': 'modal',
    'freq': 't',
    'checkpoints': '/gpfs/gibbs/pi/gerstein/yl2428/checkpoints/',
    'log_dir': '/gpfs/gibbs/pi/gerstein/yl2428/logs',
    
    # Forecasting task
    'seq_len': 96,
    'label_len': 48,
    'pred_len': 96,
    'seasonal_patterns': 'Monthly',
    'stride': 8,
    
    # Model define
    'enc_in': 3,
    'dec_in': 3,
    'c_out': 1,
    'd_model': 16,
    'n_heads': 8,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': 32,
    'moving_avg': 25,
    'factor': 1,
    'dropout': 0.1,
    'embed': 'timeF',
    'activation': 'gelu',
    'output_attention': False,
    'patch_len': 16,
    'prompt_domain': 0,
    'llm_model': 'LLAMA',
    'llm_dim': 4096,
    
    # Optimization
    'num_workers': 10,
    'itr': 1,
    'train_epochs': 10,
    'align_epochs': 10,
    'ema_decay': 0.995,
    'batch_size': 32,
    'eval_batch_size': 8,
    'patience': 10,
    'learning_rate': 0.0001,
    'des': 'test',
    'loss': 'MSE',
    'lradj': 'COS',
    'pct_start': 0.2,
    'use_amp': False,
    'llm_layers': 6,
    'percent': 100,
    'num_individuals': -1,
    'enable_covariates': 0,
    'cov_type': 'tensor',
    'gradient_accumulation_steps': 1,
    'use_deep_speed': 1,
    
    # WandB
    'wandb': 1,
    'wandb_group': None,
    'wandb_api_key': '6f1080f993d5d7ad6103e69ef57dd9291f1bf366',
    'num_experts': 8,
    'head_dropout': 0.1,

    # TimeMixer-specific parameters
    'channel_independence': 0,
    'decomp_method': 'moving_avg',
    'use_norm': 1,
    'down_sampling_layers': 2,
    'down_sampling_window': 1,
    'down_sampling_method': 'avg',
    'use_future_temporal_feature': 0,
    
    # Diffusion-specific parameters
    'k_z': 1e-2,
    'k_cond': 1,
    'd_z': 8,
    
    # De-stationary projector parameters
    'p_hidden_dims': [64, 64],
    'p_hidden_layers': 2,
    
    # CART-specific parameters
    'diffusion_config_dir': '/home/yl2428/Time-LLM/models/model9_NS_transformer/configs/toy_8gauss.yml',

    # 'cond_pred_model_dir':'./checkpoints/cond_pred_model_pertrain_NS_Transformer/checkpoint.pth',
    'cond_pred_model_pertrain_dir': None,
    'CART_input_x_embed_dim': 32,
    'mse_timestep': 0,
    'MLP_diffusion_net': False,
    
    # Ax-specific parameters
    'timesteps': 1000,
}