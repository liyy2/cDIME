import torch
import pytorch_lightning as pl
from data_provider_pretrain.data_factory import data_provider
from models.time_series_model import TimeSeriesModel
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import time
import os
from datetime import timedelta
from utils.callbacks import EMA
from utils.clean_args import clean_args
from utils.seeding import set_seed
from utils.wandb import setup_wandb

# timellm specific configs
from utils.parsers.parser_timellm import timellm_parse
from configs.config_timellm import config

## Set as env variables?
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# set seed
fix_seed = 2021
set_seed(2021)

# parse args
parser = timellm_parse()
args = parser.parse_args()
args = clean_args(args)

# Update config with command-line arguments, overriding defaults
args_dict = vars(args)  # Convert Namespace to a dictionary
config.update({k: v for k, v in args_dict.items() if v is not None})

for ii in range(args.itr):
    train_data, train_loader, args = data_provider(args, args.data_pretrain, args.data_path_pretrain, True, 'train')
    vali_data, vali_loader, args = data_provider(args, args.data_pretrain, args.data_path_pretrain, True, 'val')
    test_data, test_loader, args = data_provider(args, args.data_pretrain, args.data_path_pretrain, False, 'test')
    model = TimeSeriesModel(args, train_loader, vali_loader, test_loader)
    
    # setup callbacks
    checkpoint_path = os.path.join(args.log_dir, args.model, str(run_name), 'checkpoints')
    callbacks = [ModelCheckpoint(
                    dirpath=checkpoint_path,
                    monitor="val_loss",
                    save_top_k=1,  # -1 to save all
                    filename="{epoch}-{step}-{val_loss:.4f}",
                    save_last=True,),
                 ModelCheckpoint(
                    dirpath=checkpoint_path,
                    train_time_interval=timedelta(hours=2), # 2 hours safeguard
                    filename="time-checkpoint-{step}"),
                 EarlyStopping("val_loss", patience=args.patience),
                 LearningRateMonitor(logging_interval='step')]
    if args.ema_decay!=1:
        callbacks.append(EMA(decay=args.ema_decay, deep_speed=args.use_deep_speed))

    # setup logger    
    wandb_logger = setup_wandb(args) if args.wandb else None
    
    run_name = wandb_logger.experiment.name if wandb_logger else time.strftime('%Y-%m-%d-%H-%M-%S')
    print(run_name)
    
    #ENABLE TENSOR CORES
    if args.precision == '32':
       torch.set_float32_matmul_precision('high') # set from highest to high
    # ddp_plugin = None

    trainer = pl.Trainer(
        max_epochs=args.train_epochs,
        devices=args.num_nodes,
        accelerator='auto',
        strategy='deepspeed' if args.use_deep_speed else 'ddp',
        logger=wandb_logger,
        callbacks=callbacks,
        precision=args.precision,
        enable_checkpointing=True,
        gradient_clip_val=0.5,
        gradient_clip_algorithm='norm',
        accumulate_grad_batches=args.gradient_accumulation_steps, 
        default_root_dir=checkpoint_path)

    trainer.fit(model, train_loader, vali_loader)
    # load checkpoint
    trainer.test(model, test_loader)