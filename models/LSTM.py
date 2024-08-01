import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_ff,
            num_layers=3,
            batch_first=True,
            dropout=configs.dropout
        )

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_ff,
            num_layers=3,
            batch_first=True,
            dropout=configs.dropout
        )

        self.out_layer = nn.Linear(configs.d_ff, configs.c_out, bias=False)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # Encoder
        x_enc_emb = self.embedding(x_enc, x_mark_enc)
        _, (hidden, cell) = self.encoder_lstm(x_enc_emb)

        # Decoder
        x_dec_emb = self.embedding(x_dec, x_mark_dec)
        decoder_outputs, _ = self.decoder_lstm(x_dec_emb, (hidden, cell))

        x_out = self.out_layer(decoder_outputs)

        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, covariates=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return x_out[:, -self.pred_len:, :]
