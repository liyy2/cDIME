
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from models import Autoformer, DLinear, TimeLLM, DLinearChannelMix, DLinearMoE, TimeMixer, DLinearMoECov
import pytorch_lightning as pl


class TimeSeriesModel(pl.LightningModule):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        if args.model == 'Autoformer':
            self.model = Autoformer.Model(args).float()
        elif args.model.startswith('DLinear'):
            self.model = eval(args.model).Model(args).float()
        elif args.model.startswith('TimeMixer'):
            self.model = eval(args.model).Model(args).float()
        else:
            self.model = TimeLLM.Model(args).float()
        
        self.criterion = nn.MSELoss()
        self.mae_metric = nn.L1Loss()
        self.remove_key = 'llm_model' # remove pretrained model from checkpoint

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cov=None):
        if self.args.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, covariates=batch_cov)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, covariates=batch_cov)
        return outputs

    def training_step(self, batch, batch_idx):
        if self.args.enable_covariates:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[0]
            batch_cov = batch[1]
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_cov = None

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        outputs = self(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cov)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        loss = self.criterion(outputs, batch_y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.args.enable_covariates:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[0]
            batch_cov = batch[1]
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_cov = None
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        outputs = self(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cov)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        loss = self.criterion(outputs, batch_y)
        mae_loss = self.mae_metric(outputs, batch_y)
        self.log("val_loss", loss)
        self.log("val_mae_loss", mae_loss)

    def test_step(self, batch, batch_idx):
        if self.args.enable_covariates:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[0]
            batch_cov = batch[1]
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_cov = None

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        outputs = self(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cov)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        loss = self.criterion(outputs, batch_y)
        mae_loss = self.mae_metric(outputs, batch_y)

        mae_loss_0 = self.mae_metric(outputs[:, 0:self.args.pred_len//4, :], batch_y[:, 0:self.args.pred_len//4, :])
        mae_loss_1 = self.mae_metric(outputs[:, self.args.pred_len//4:self.args.pred_len//2, :], batch_y[:, self.args.pred_len//4:self.args.pred_len//2, :])
        mae_loss_2 = self.mae_metric(outputs[:, self.args.pred_len//2:3*self.args.pred_len//4, :], batch_y[:, self.args.pred_len//2:3*self.args.pred_len//4, :])
        mae_loss_3 = self.mae_metric(outputs[:, 3*self.args.pred_len//4:, :], batch_y[:, 3*self.args.pred_len//4:, :])


        self.log("test_loss", loss)
        self.log("test_mae_loss", mae_loss)
        self.log("test_mae_loss_0", mae_loss_0)
        self.log("test_mae_loss_1", mae_loss_1)
        self.log("test_mae_loss_2", mae_loss_2)
        self.log("test_mae_loss_3", mae_loss_3)
        # 

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
    
    
    def on_save_checkpoint(self, checkpoint) -> None:
        if self.remove_key is not None:
            self.remove_params(checkpoint, key=self.remove_key)


    def remove_params(self, checkpoint, key) -> None:
        del_keys = []

        for k in checkpoint["state_dict"]:
            if key in k:
                del_keys.append(k)

        for k in del_keys:
            checkpoint["state_dict"].pop(k)

    def configure_optimizers(self):
        trained_parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trained_parameters, lr=self.args.learning_rate, weight_decay=1e-4)

        if self.args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= self.args.train_epochs, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                steps_per_epoch=len(self.train_dataloader()),
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)

        return [optimizer], [scheduler]