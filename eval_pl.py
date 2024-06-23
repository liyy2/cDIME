import argparse
import torch
from data_provider_pretrain.data_factory import data_provider
from models.time_series_model import TimeSeriesModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from utils.callbacks import EMA
from lightning.pytorch.loggers import WandbLogger
import time
import random
import numpy as np
import os
import wandb
from datetime import timedelta
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.cuda.manual_seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# basic config
parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes for gpu')
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=False, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=False, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--precision', type=str, default='32', help='precision')
# data loader
parser.add_argument('--data_pretrain', type=str, required=False, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='/home/yl2428/Time-LLM/dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--data_path_pretrain', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='/gpfs/gibbs/pi/gerstein/yl2428/checkpoints/', help='location of model checkpoints')
parser.add_argument('--log_dir', type=str, default='/gpfs/gibbs/pi/gerstein/yl2428/logs', help='location of log')
# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--stride', type=int, default=8, help='stride in dataset construction')
# model define
parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=3, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default=None,
                    help='down sampling method, only support avg, max, conv')
# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--ema_decay', type=float, default=0.995, help='ema decay')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='COS', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--num_individuals', type=int, default=-1)
parser.add_argument('--enable_covariates', type=int, default=0)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--use_deep_speed', type=int, default=1)
# wandb
parser.add_argument('--wandb', type=int, default=1, help='whether to use wandb')
parser.add_argument('--wandb_group', type=str, default=None, help='wandb group')
parser.add_argument('--wandb_api_key', type=str, default='6f1080f993d5d7ad6103e69ef57dd9291f1bf366')
parser.add_argument('--num_heads', type=str, default=8)
parser.add_argument('--head_dropout', type=float, default=0.1)
parser.add_argument('--ckpt_path', type=str)






args = parser.parse_args()
for ii in range(args.itr):
    train_data, train_loader = data_provider(args, args.data_pretrain, args.data_path_pretrain, True, 'train')
    vali_data, vali_loader = data_provider(args, args.data_pretrain, args.data_path_pretrain, True, 'val')
    test_data, test_loader = data_provider(args, args.data_pretrain, args.data_path_pretrain, False, 'test')
    callbacks = []

    model = TimeSeriesModel(args, train_loader, vali_loader, test_loader)
    # callbacks.append(ModelCheckpoint(
    #     dirpath='/gpfs/gibbs/pi/gerstein/yl2428/logs/DLinearMoE/fragrant-tree-74/checkpoints',
    # ))

    trainer = pl.Trainer(
        accelerator='auto',
        strategy='deepspeed' if args.use_deep_speed else 'ddp',
        precision=args.precision,
        enable_checkpointing=True,)

    # load checkpoint
    trainer.test(model, test_loader, ckpt_path=args.ckpt_path)