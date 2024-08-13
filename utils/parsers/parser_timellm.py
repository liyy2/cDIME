import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Time-LLM')

    # Basic config
    parser.add_argument('--num_nodes', type=int, help='number of nodes for GPU')
    parser.add_argument('--task_name', type=str, help='task name')
    parser.add_argument('--is_training', type=int, help='status')
    parser.add_argument('--model_id', type=str, help='model id')
    parser.add_argument('--model_comment', type=str, help='prefix when saving test results')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--precision', type=str, help='precision')
    
    # Data loader
    parser.add_argument('--data_pretrain', type=str, help='dataset type')
    parser.add_argument('--root_path', type=str, help='root path of the data file')
    parser.add_argument('--data_path', type=str, help='data file')
    parser.add_argument('--data_path_pretrain', type=str, help='data file')
    parser.add_argument('--features', type=str, help='forecasting task')
    parser.add_argument('--target', type=str, help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, help='dataset type')
    parser.add_argument('--freq', type=str, help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, help='location of model checkpoints')
    parser.add_argument('--log_dir', type=str, help='location of log')
    
    # Forecasting task
    parser.add_argument('--seq_len', type=int, help='input sequence length')
    parser.add_argument('--label_len', type=int, help='start token length')
    parser.add_argument('--pred_len', type=int, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, help='subset for M4')
    parser.add_argument('--stride', type=int, help='stride in dataset construction')
    
    # Model define
    parser.add_argument('--enc_in', type=int, help='encoder input size')
    parser.add_argument('--dec_in', type=int, help='decoder input size')
    parser.add_argument('--c_out', type=int, help='output size')
    parser.add_argument('--d_model', type=int, help='dimension of model')
    parser.add_argument('--n_heads', type=int, help='num of heads')
    parser.add_argument('--e_layers', type=int, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, help='window size of moving average')
    parser.add_argument('--factor', type=int, help='attn factor')
    parser.add_argument('--dropout', type=float, help='dropout')
    parser.add_argument('--embed', type=str, help='time features encoding')
    parser.add_argument('--activation', type=str, help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, help='patch length')
    parser.add_argument('--prompt_domain', type=int, help='')
    parser.add_argument('--llm_model', type=str, help='LLM model')
    parser.add_argument('--llm_dim', type=int, help='LLM model dimension')
    
    # Optimization
    parser.add_argument('--num_workers', type=int, help='data loader num workers')
    parser.add_argument('--itr', type=int, help='experiments times')
    parser.add_argument('--train_epochs', type=int, help='train epochs')
    parser.add_argument('--align_epochs', type=int, help='alignment epochs')
    parser.add_argument('--ema_decay', type=float, help='ema decay')
    parser.add_argument('--batch_size', type=int, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, help='optimizer learning rate')
    parser.add_argument('--des', type=str, help='exp description')
    parser.add_argument('--loss', type=str, help='loss function')
    parser.add_argument('--lradj', type=str, help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, help='number of LLM layers')
    parser.add_argument('--percent', type=int, help='percentage of data to use')
    parser.add_argument('--num_individuals', type=int, help='number of individuals')
    parser.add_argument('--enable_covariates', type=int, help='enable covariates')
    parser.add_argument('--cov_type', type=str, choices=['text', 'tensor'], help='type of covariates')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='number of gradient accumulation steps')
    parser.add_argument('--use_deep_speed', type=int, help='use DeepSpeed')
    
    # WandB
    parser.add_argument('--wandb', type=int, help='whether to use wandb')
    parser.add_argument('--wandb_group', type=str, help='wandb group')
    parser.add_argument('--wandb_api_key', type=str, help='wandb API key')
    
    # TimeMixer-specific parameters
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, help='number of hidden layers in projector')
    parser.add_argument('--channel_independence', type=int, help='channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, help='method of series decomposition')
    parser.add_argument('--use_norm', type=int, help='whether to use normalization')
    parser.add_argument('--down_sampling_layers', type=int, help='number of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, help='down sampling method')
    parser.add_argument('--use_future_temporal_feature', type=int, help='whether to use future temporal feature')

    return parser.parse_args()