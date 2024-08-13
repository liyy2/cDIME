import wandb
from pytorch_lightning.loggers import WandbLogger

# wandb is project/group/name format to save all the log
def setup_wandb(args):
    wandb.login(key=args.wandb_api_key, relogin=True)
    return WandbLogger(
        project='Glucose Forecasting',
        group=args.wandb_group,
        settings=wandb.Settings(start_method='fork', code_dir="."),
        config=args,
        save_dir=args.log_dir,
        dir=args.log_dir,
        log_model=True,
    )