import pytorch_lightning as pl
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping
from utils.metrics import metric
from torch.optim import lr_scheduler
from models.model9_NS_transformer.ns_models import ns_Transformer
from models.model9_NS_transformer.diffusion_models import diffuMTS
from models.model9_NS_transformer.diffusion_models.diffusion_utils import *

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time
import CRPS.CRPS as pscore

import warnings
warnings.filterwarnings('ignore')

def ccc(id, pred, true):
    res_box = np.zeros(len(true))
    for i in range(len(true)):
        res = pscore(pred[i], true[i]).compute()
        res_box[i] = res[0]
    return res_box

def log_normal(x, mu, var):
    eps = 1e-8
    if eps > 0.0:
        var = var + eps
    return 0.5 * torch.mean(np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)

class TimeSeriesDiffusionModel(pl.LightningModule):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None):
        super(TimeSeriesDiffusionModel, self).__init__()
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model, self.cond_pred_model, self.cond_pred_model_train = self._build_model()
        self.save_hyperparameters()


    def _build_model(self):
        model = diffuMTS.Model(self.args).float()
        cond_pred_model = ns_Transformer.Model(self.args).float()
        cond_pred_model_train = ns_Transformer.Model(self.args).float()
        return model, cond_pred_model, cond_pred_model_train


    def configure_optimizers(self):
        optimizer = optim.AdamW([{'params': self.model.parameters()}, {'params': self.cond_pred_model.parameters()}], 
                                lr=self.args.learning_rate,  weight_decay=1e-4)
        if self.args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= self.args.train_epochs, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                steps_per_epoch=len(self.train_dataloader()),
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        if self.args.enable_covariates:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[0]
            batch_cov = batch[1]
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_cov = None

        # tensor now contains the past 10 days of temperatures followed by 5 zero-initialized time steps.
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # sample time steps of size batch_size
        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(self.device)
        # Symmetrical Sampling
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        _, y_0_hat_batch, KL_loss, z_sample = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))
        loss_vae_all = loss_vae + self.args.k_z * KL_loss

        y_T_mean = y_0_hat_batch
        e = torch.randn_like(batch_y).to(self.device)
        y_t_batch = q_sample(batch_y, y_T_mean, self.model.alphas_bar_sqrt, self.model.one_minus_alphas_bar_sqrt, t, noise=e)
        output = self.model(batch_x, batch_x_mark, batch_y, y_t_batch, y_0_hat_batch, t)

        loss = (e[:, -self.args.pred_len:, :] - output[:, -self.args.pred_len:, :]).square().mean() + self.args.k_cond * loss_vae_all
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.args.enable_covariates:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[0]
            batch_cov = batch[1]
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_cov = None
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        _, y_0_hat_batch, KL_loss, z_sample = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))
        loss_vae_all = loss_vae + self.args.k_z * KL_loss

        y_T_mean = y_0_hat_batch
        e = torch.randn_like(batch_y).to(self.device)
        y_t_batch = q_sample(batch_y, y_T_mean, self.model.alphas_bar_sqrt, self.model.one_minus_alphas_bar_sqrt, t, noise=e)
        output = self.model(batch_x, batch_x_mark, batch_y, y_t_batch, y_0_hat_batch, t)

        loss = (e - output).square().mean() + self.args.k_cond * loss_vae_all
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        if self.args.enable_covariates:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[0]
            batch_cov = batch[1]
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_cov = None
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

        _, y_0_hat_batch, _, _ = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        repeat_n = int(self.model.diffusion_config.testing.n_z_samples / self.model.diffusion_config.testing.n_z_samples_depart)
        y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)
        y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1)
        y_T_mean_tile = y_0_hat_tile
        x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
        x_tile = x_tile.transpose(0, 1).flatten(0, 1)
        x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)
        x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1)

        gen_y_box = []
        for _ in range(self.model.diffusion_config.testing.n_z_samples_depart):
            y_tile_seq = p_sample_loop(self.model, x_tile, x_mark_tile, y_0_hat_tile, y_T_mean_tile,
                                       self.model.num_timesteps,
                                       self.model.alphas, self.model.one_minus_alphas_bar_sqrt)
            gen_y = y_tile_seq[-1].reshape(batch_x.shape[0],
                                           int(self.model.diffusion_config.testing.n_z_samples / self.model.diffusion_config.testing.n_z_samples_depart),
                                           (self.args.label_len + self.args.pred_len),
                                           self.args.c_out).cpu().numpy()
            gen_y_box.append(gen_y)
        
        outputs = np.concatenate(gen_y_box, axis=1)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, :, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cpu().numpy()

        return {'pred': outputs, 'true': batch_y}
    
    def test_epoch_end(self, outputs):
        all_preds = np.concatenate([x['pred'] for x in outputs])
        all_trues = np.concatenate([x['true'] for x in outputs])

        # Compute metrics
        preds_ns = all_preds.mean(axis=1)
        preds_ns = preds_ns.reshape(-1, preds_ns.shape[-2], preds_ns.shape[-1])
        trues_ns = all_trues.reshape(-1, all_trues.shape[-2], all_trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds_ns, trues_ns)
        self.log('test_mse', mse)
        self.log('test_mae', mae)


    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
    