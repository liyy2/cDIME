import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List

from torch import Tensor
from torch.nn import Linear, Module, ModuleList

from .ns_layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from .ns_layers.SelfAttention_Family import DSAttention, AttentionLayer

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.conv import TabTransformerConv
from layers.Embed import DataEmbedding, PatchEmbedding
from torch_frame.nn.encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeWiseFeatureEncoder,
)
from torch_frame.nn import Trompt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Routing statistics will not be logged.")


class ExampleTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        num_heads: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
    ):
        super().__init__()
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder()
            },
        )
        self.tab_transformer_convs = ModuleList([
            TabTransformerConv(
                channels=channels,
                num_heads=num_heads,
            ) for _ in range(num_layers)
        ])
        self.decoder = Linear(channels, out_channels)

    def forward(self, tf: TensorFrame) -> Tensor:
        x, _ = self.encoder(tf)
        for tab_transformer_conv in self.tab_transformer_convs:
            x = tab_transformer_conv(x)
        out = self.decoder(x.mean(dim=1))
        return out

class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


def configure_moe(configs, use_moe=True, num_experts=4, expert_layers=2, 
                  moe_layer_indices=None, use_sparse_gating=False, top_k_experts=2, 
                  moe_loss_weight=0.01, log_routing_stats=True, num_universal_experts=1,
                  universal_expert_weight=0.3):
    """
    Helper function to configure Mixture of Experts parameters
    
    Args:
        configs: Model configuration object
        use_moe: Whether to use MoE (default: True)
        num_experts: Number of expert networks (default: 4)
        expert_layers: Number of layers per expert (default: 2)
        moe_layer_indices: Which encoder layers to replace with MoE (default: [1, 3])
        use_sparse_gating: Whether to use sparse gating (default: False)
        top_k_experts: Number of top experts to use when sparse gating is enabled (default: 2)
        moe_loss_weight: Weight for MoE loss in the combined KL loss (default: 0.01)
        log_routing_stats: Whether to log routing statistics to wandb (default: True)
        num_universal_experts: Number of universal experts that are always used (default: 1)
        universal_expert_weight: Weight given to universal experts (default: 0.3)
    """
    configs.use_moe = use_moe
    configs.num_experts = num_experts
    configs.expert_layers = expert_layers
    configs.moe_layer_indices = moe_layer_indices if moe_layer_indices is not None else [1, 3]
    configs.use_sparse_gating = use_sparse_gating
    configs.top_k_experts = top_k_experts
    configs.moe_loss_weight = moe_loss_weight
    configs.log_routing_stats = log_routing_stats
    configs.num_universal_experts = num_universal_experts
    configs.universal_expert_weight = universal_expert_weight
    
    return configs


def configure_contrastive_learning(configs, use_contrastive_learning=True, 
                                   contrastive_loss_weight=0.1, contrastive_temperature=0.1,
                                   use_momentum_encoder=True, momentum_factor=0.999,
                                   augmentation_strength=0.1):
    """
    Helper function to configure contrastive learning parameters for context representation
    
    Args:
        configs: Model configuration object
        use_contrastive_learning: Whether to use InfoNCE contrastive learning (default: True)
        contrastive_loss_weight: Weight for contrastive loss in combined loss (default: 0.1)
        contrastive_temperature: Temperature parameter for InfoNCE loss (default: 0.1)
        use_momentum_encoder: Whether to use momentum encoder for key generation (default: True)
        momentum_factor: Momentum factor for exponential moving average (default: 0.999)
        augmentation_strength: Strength of context augmentation (default: 0.1)
    """
    configs.use_contrastive_learning = use_contrastive_learning
    configs.contrastive_loss_weight = contrastive_loss_weight
    configs.contrastive_temperature = contrastive_temperature
    configs.use_momentum_encoder = use_momentum_encoder
    configs.momentum_factor = momentum_factor
    configs.augmentation_strength = augmentation_strength
    
    return configs


class TimeSeriesCovariateEncoder(nn.Module):
    """
    1D CNN encoder for time-series covariates to get both time-invariant and dynamic embeddings
    """
    def __init__(self, in_channels, d_model, seq_len, dropout=0.1):
        super(TimeSeriesCovariateEncoder, self).__init__()
        
        # Multi-scale 1D CNN layers
        self.conv_layers = nn.ModuleList([
            # First layer: capture local patterns
            nn.Conv1d(in_channels, d_model // 4, kernel_size=3, padding=1),
            # Second layer: capture medium-term patterns  
            nn.Conv1d(d_model // 4, d_model // 2, kernel_size=5, padding=2),
            # Third layer: capture long-term patterns
            nn.Conv1d(d_model // 2, d_model, kernel_size=7, padding=3),
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(d_model // 4),
            nn.BatchNorm1d(d_model // 2),
            nn.BatchNorm1d(d_model),
        ])
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Global pooling layers for time-invariant representation
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Final projection to get time-invariant embedding
        self.invariant_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 because we concatenate avg and max pool
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Projection for dynamic covariate tokens (temporal dimension preserved)
        self.dynamic_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x):
        """
        Args:
            x: Time-series covariates [batch_size, seq_len, in_channels]
        Returns:
            time_invariant_embedding: [batch_size, d_model]
            dynamic_embedding: [batch_size, seq_len, d_model]
        """
        # Transpose for conv1d: [batch_size, in_channels, seq_len]
        x = x.transpose(1, 2)
        
        # Apply conv layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            residual = x if i > 0 and x.shape[1] == conv.out_channels else None
            x = conv(x)
            x = bn(x)
            x = self.activation(x)
            if residual is not None:
                x = x + residual
            x = self.dropout(x)
        
        # x is now [batch_size, d_model, seq_len]
        
        # 1. Time-invariant representation via global pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # [batch_size, d_model]
        max_pool = self.global_max_pool(x).squeeze(-1)  # [batch_size, d_model]
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # [batch_size, d_model * 2]
        time_invariant_embedding = self.invariant_projection(pooled)  # [batch_size, d_model]
        
        # 2. Dynamic representation preserving temporal dimension
        # Transpose back to [batch_size, seq_len, d_model]
        dynamic_features = x.transpose(1, 2)  # [batch_size, seq_len, d_model]
        dynamic_embedding = self.dynamic_projection(dynamic_features)  # [batch_size, seq_len, d_model]
        
        return time_invariant_embedding, dynamic_embedding


class Model(nn.Module):
    """
    Non-stationary Transformer with Mixture of Experts
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        
        # MoE configuration
        self.use_moe = getattr(configs, 'use_moe', True)
        self.num_experts = getattr(configs, 'num_experts', 4)
        self.moe_layer_indices = getattr(configs, 'moe_layer_indices', [1, 3])  # Which encoder layers to replace with MoE
        self.moe_loss_weight = getattr(configs, 'moe_loss_weight', 0.01)
        self.log_routing_stats = getattr(configs, 'log_routing_stats', True)
        self.total_encoder_layers = configs.e_layers  # Store total number of encoder layers
        
        # Universal experts configuration
        self.num_universal_experts = getattr(configs, 'num_universal_experts', 1)
        self.universal_expert_weight = getattr(configs, 'universal_expert_weight', 0.3)

        # Time-series covariate configuration
        # Assume glucose is the last channel, rest are time-series covariates
        self.glucose_channels = 1  # Only glucose
        self.ts_covariate_channels = 1 if not configs.enable_context_aware else configs.enc_in  # All other channels
        
        # Embedding for glucose channel only
        self.enc_embedding = DataEmbedding(self.glucose_channels, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(self.glucose_channels, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        
        # Time-series covariate encoder
        if self.ts_covariate_channels > 0:
            self.ts_cov_encoder = TimeSeriesCovariateEncoder(
                in_channels=self.ts_covariate_channels,
                d_model=configs.d_model,
                seq_len=configs.seq_len,
                dropout=configs.dropout
            )
        
        # Encoder with optional MoE layers
        if self.use_moe:
            encoder_layers = []
            for l in range(configs.e_layers):
                if l in self.moe_layer_indices:
                    # Replace this layer with MoE
                    encoder_layers.append(None)  # Placeholder, will be handled separately
                else:
                    encoder_layers.append(
                        EncoderLayer(
                            AttentionLayer(
                                DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                            configs.d_model,
                            configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation
                        )
                    )
            
            self.encoder_layers = ModuleList([layer for layer in encoder_layers if layer is not None])
            
            # Create MoE layers
            self.moe_layers = nn.ModuleDict({
                str(idx): MixtureOfExperts(configs, self.num_experts, 
                                         num_universal_experts=self.num_universal_experts,
                                         universal_expert_weight=self.universal_expert_weight) 
                for idx in self.moe_layer_indices
            })
            
            # Create separate norm layer
            self.encoder_norm = torch.nn.LayerNorm(configs.d_model)
            
        else:
            # Standard encoder
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, self.glucose_channels, bias=True)
        )

        self.tau_learner = Projector(enc_in=self.glucose_channels, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=self.glucose_channels, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)

        self.z_mean = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model)
        )
        self.z_logvar = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model)
        )
        # Covariate encoder (time-invariant)
        self.cov_encoder = Trompt(
                channels=configs.d_model,
                out_channels=configs.d_model,
                num_prompts=128,
                num_layers=6,
                col_stats=configs.col_stats,
                col_names_dict=configs.col_names_dict,
        )
        self.z_out = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model)
        )
        
        # Covariate fusion layer to combine tabular and time-series covariates
        self.covariate_fusion = nn.Sequential(
            nn.Linear(configs.d_model * 7, configs.d_model),  # Combine tabular + time-series
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, configs.d_model)
        )

        # STRONG covariate conditioning for encoder/decoder
        self.enc_cov_scale_net = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.ReLU(),
            nn.Linear(configs.d_model // 2, configs.d_model),
            nn.Sigmoid()  # Ensure positive scales
        )
        self.enc_cov_shift_net = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.ReLU(),
            nn.Linear(configs.d_model // 2, configs.d_model)
        )
        
        self.dec_cov_scale_net = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.ReLU(),
            nn.Linear(configs.d_model // 2, configs.d_model),
            nn.Sigmoid()  # Ensure positive scales
        )
        self.dec_cov_shift_net = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.ReLU(),
            nn.Linear(configs.d_model // 2, configs.d_model)
        )
        
        # Cross-attention between sequence and covariates
        self.enc_cov_attention = nn.MultiheadAttention(
            embed_dim=configs.d_model,
            num_heads=configs.n_heads,
            dropout=configs.dropout,
            batch_first=True
        )
        self.dec_cov_attention = nn.MultiheadAttention(
            embed_dim=configs.d_model,
            num_heads=configs.n_heads,
            dropout=configs.dropout,
            batch_first=True
        )
        
        # Covariate importance learning
        self.cov_importance = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.Tanh(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.Sigmoid()
        )
        
        # Contrastive context learning for enriching covariate representations
        self.use_contrastive_learning = getattr(configs, 'use_contrastive_learning', True)
        self.contrastive_loss_weight = getattr(configs, 'contrastive_loss_weight', 0.1)
        self.glucose_dropout_rate= getattr(configs, 'glucose_dropout_rate', 0.4)
        self.glucose_dropout = nn.Dropout(self.glucose_dropout_rate)
        if self.use_contrastive_learning:
            self.contrastive_context_learner = ContrastiveContextLearning(
                d_model=configs.d_model,
                temperature=getattr(configs, 'contrastive_temperature', 0.1),
                use_momentum=getattr(configs, 'use_momentum_encoder', True),
                momentum=getattr(configs, 'momentum_factor', 0.999)
            )

    def KL_loss_normal(self, posterior_mean, posterior_logvar):
        KL = -0.5 * torch.mean(1 - posterior_mean ** 2 + posterior_logvar -
                               torch.exp(posterior_logvar), dim=1)
        return torch.mean(KL)

    def reparameterize(self, posterior_mean, posterior_logvar):
        posterior_var = posterior_logvar.exp()
        # take sample
        if self.training:
            posterior_mean = posterior_mean.repeat(100, 1, 1, 1)
            posterior_var = posterior_var.repeat(100, 1, 1, 1)
            eps = torch.zeros_like(posterior_var).normal_()
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
            z = z.mean(0)
        else:
            z = posterior_mean
        # z = posterior_mean
        return z

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, covariates=None):

        # IMPORTANT: Only use encoder data for time-series covariates to prevent future data leakage
        # Separate glucose (last channel) from time-series covariates
        x_glucose = x_enc[:, :, -self.glucose_channels:]  # [B, seq_len, 1] - glucose channel (historical only)
        x_glucose = self.glucose_dropout(x_glucose)
        x_ts_covariates = x_enc[:, :, -self.glucose_channels:]  if self.ts_covariate_channels == 1 else x_enc[:, :, :] # [B, seq_len, C-1] (historical only)
        
        # For decoder, ONLY extract glucose from x_dec (no time-series covariates to prevent leakage)
        x_dec_glucose = x_dec[:, :, -self.glucose_channels:]  # [B, label_len + pred_len, 1]
        
        # Process tabular covariates (time-invariant, no leakage concern)
        tabular_cov_embedding = self.cov_encoder(covariates)  # [B, d_model]
        tabular_cov_embedding = tabular_cov_embedding.reshape(tabular_cov_embedding.shape[0], -1) # [B, 192]
        # Process time-series covariates ONLY from historical encoder data (no future data)
        if self.ts_covariate_channels > 0 and x_ts_covariates is not None:



            
            # Use only historical time-series covariates from encoder to get both representations
            ts_cov_invariant, ts_cov_dynamic = self.ts_cov_encoder(x_ts_covariates)  # [B, d_model], [B, seq_len, d_model]
            
            # Fuse invariant tabular and time-series covariate embeddings
            combined_invariant_cov = self.covariate_fusion(
                torch.cat([tabular_cov_embedding, ts_cov_invariant], dim=1)
            )  # [B, d_model]
        else:
            # Use only tabular covariates if no time-series covariates
            combined_invariant_cov = tabular_cov_embedding  # [B, d_model]
            ts_cov_dynamic = None  # No dynamic time-series covariates
            
        # Apply contrastive learning to enrich context representations
        contrastive_loss = 0.0
        original_combined_invariant_cov = combined_invariant_cov.clone()  # Store for logging
        
        if self.use_contrastive_learning and self.training:
            contrastive_loss, enriched_invariant_cov = self.contrastive_context_learner(
                combined_invariant_cov, 
                augmentation_strength=getattr(self, 'augmentation_strength', 0.1)
            )
            # Use enriched representations for better conditioning
            combined_invariant_cov = enriched_invariant_cov
            
            # Log detailed contrastive learning statistics
            self._log_contrastive_learning_details(
                contrastive_loss, original_combined_invariant_cov, enriched_invariant_cov
            )
        elif self.use_contrastive_learning:
            # During inference, still apply the context encoder for consistency
            with torch.no_grad():
                _, enriched_invariant_cov = self.contrastive_context_learner(
                    combined_invariant_cov, augmentation_strength=0.0
                )
                combined_invariant_cov = enriched_invariant_cov
            
        x_raw = x_glucose.clone().detach()  # Use only glucose for raw processing

        # Normalization (only for glucose)
        mean_enc = x_glucose.mean(1, keepdim=True).detach()  # B x 1 x 1
        x_glucose_norm = x_glucose - mean_enc
        std_enc = torch.sqrt(torch.var(x_glucose_norm, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x 1
        x_glucose_norm = x_glucose_norm / std_enc
        
        # Create decoder input with normalized glucose only (NO time-series covariates)
        x_dec_new = torch.cat([x_glucose_norm[:, -self.label_len:, :], 
                              torch.zeros_like(x_dec_glucose[:, -self.pred_len:, :])],
                              dim=1).to(x_glucose.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x 1, B x 1 x 1 -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, mean_enc)  # B x S x 1, B x 1 x 1 -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x_glucose_norm, x_mark_enc)
        
        # STRONG covariate conditioning for encoder with proper cross-attention
        # Learn importance of invariant covariate features
        cov_importance = self.cov_importance(combined_invariant_cov)  # [batch, d_model]
        weighted_invariant_cov = combined_invariant_cov * cov_importance  # [batch, d_model]
        
        # Apply FiLM conditioning to encoder using invariant covariates
        enc_cov_scale = self.enc_cov_scale_net(weighted_invariant_cov)  # [batch, d_model]
        enc_cov_shift = self.enc_cov_shift_net(weighted_invariant_cov)  # [batch, d_model]
        enc_cov_scale = enc_cov_scale.unsqueeze(1)  # [batch, 1, d_model]
        enc_cov_shift = enc_cov_shift.unsqueeze(1)  # [batch, 1, d_model]
        enc_out = enc_cov_scale * enc_out + enc_cov_shift
        
        # Create covariate tokens for cross-attention
        # Static covariate token: [batch, d_model] -> [batch, 1, d_model]
        static_cov_token = weighted_invariant_cov.unsqueeze(1)  # [batch, 1, d_model]
        
        # Combine static and dynamic covariate tokens
        if ts_cov_dynamic is not None:
            # Dynamic covariate tokens: [batch, seq_len, d_model]
            # Concatenate along sequence dimension: [batch, 1 + seq_len, d_model]
            covariate_tokens = torch.cat([static_cov_token, ts_cov_dynamic], dim=1)
        else:
            # Only static covariate token: [batch, 1, d_model]
            covariate_tokens = static_cov_token
        
        # Cross-attention between encoder output and combined covariate tokens
        enc_attended, enc_attn_weights = self.enc_cov_attention(
            query=enc_out,           # [batch, seq_len, d_model]
            key=covariate_tokens,    # [batch, 1 + seq_len, d_model] or [batch, 1, d_model]
            value=covariate_tokens   # [batch, 1 + seq_len, d_model] or [batch, 1, d_model]
        )
        # Residual connection with learned gating
        # enc_attn_weights: [batch, seq_len, 1 + seq_len] or [batch, seq_len, 1]
        enc_gate = torch.sigmoid(torch.mean(enc_attn_weights, dim=-1, keepdim=True))  # [batch, seq_len, 1]
        enc_out = enc_out + enc_gate * enc_attended
        
        # Encoder processing with MoE (with strongly conditioned input)
        total_moe_loss = 0.0
        routing_info = {}
        
        if self.use_moe:
            # Process through encoder layers with MoE integration
            regular_layer_idx = 0
            attns = []
            
            for layer_idx in range(self.total_encoder_layers):
                if layer_idx in self.moe_layer_indices:
                    # Use MoE layer with weighted covariate embedding
                    moe_layer = self.moe_layers[str(layer_idx)]
                    enc_out, moe_loss, routing_weights = moe_layer(
                        enc_out, weighted_invariant_cov, attn_mask=enc_self_mask, tau=tau, delta=delta
                    )
                    total_moe_loss += moe_loss
                    routing_info[f'layer_{layer_idx}'] = routing_weights
                else:
                    # Use regular encoder layer
                    if regular_layer_idx < len(self.encoder_layers):
                        enc_out, attn = self.encoder_layers[regular_layer_idx](
                            enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta
                        )
                        if self.output_attention:
                            attns.append(attn)
                        regular_layer_idx += 1
            
            # Apply final norm
            enc_out = self.encoder_norm(enc_out)
            
        else:
            # Standard encoder processing
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        mean = self.z_mean(enc_out)
        logvar = self.z_logvar(enc_out)

        z_sample = self.reparameterize(mean, logvar)

        enc_out = self.z_out(z_sample)

        KL_z = self.KL_loss_normal(mean, logvar)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        
        # STRONG covariate conditioning for decoder (instead of simple addition)
        # Apply FiLM conditioning to decoder
        dec_cov_scale = self.dec_cov_scale_net(weighted_invariant_cov)  # [batch, d_model]
        dec_cov_shift = self.dec_cov_shift_net(weighted_invariant_cov)  # [batch, d_model]
        dec_cov_scale = dec_cov_scale.unsqueeze(1)  # [batch, 1, d_model]
        dec_cov_shift = dec_cov_shift.unsqueeze(1)  # [batch, 1, d_model]
        dec_out = dec_cov_scale * dec_out + dec_cov_shift
        
        # Cross-attention between decoder output and covariate information
        dec_attended, dec_attn_weights = self.dec_cov_attention(
            query=dec_out,  # [batch, seq_len, d_model]
            key=covariate_tokens,   # [batch, 1 + seq_len, d_model] or [batch, 1, d_model]
            value=covariate_tokens   # [batch, 1 + seq_len, d_model] or [batch, 1, d_model]
        )
        # Residual connection with learned gating
        dec_gate = torch.sigmoid(torch.mean(dec_attn_weights, dim=-1, keepdim=True))  # [batch, seq_len, 1]
        dec_out = dec_out + dec_gate * dec_attended
        
        # Apply decoder layers
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

        # De-normalization
        dec_out = dec_out * std_enc + mean_enc

        # Combine KL loss with MoE loss and contrastive loss for compatibility
        combined_KL_z = KL_z + self.moe_loss_weight * total_moe_loss
        if self.use_contrastive_learning:
            combined_KL_z = combined_KL_z + self.contrastive_loss_weight * contrastive_loss

        # Log routing statistics to wandb if available and enabled
        if self.use_moe and self.log_routing_stats and WANDB_AVAILABLE and routing_info:
            self._log_routing_statistics(routing_info, total_moe_loss)
            
        # Log contrastive learning statistics
        if self.use_contrastive_learning and WANDB_AVAILABLE and self.training:
            try:
                wandb.log({
                    "contrastive/loss": contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0,
                    "contrastive/weight": self.contrastive_loss_weight,
                    "contrastive/temperature": self.contrastive_context_learner.temperature if hasattr(self.contrastive_context_learner, 'temperature') else 0.1,
                })
            except Exception as e:
                print(f"Error logging contrastive learning statistics: {e}")

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :], dec_out, combined_KL_z, z_sample, combined_invariant_cov  # Return enriched covariates

    def get_routing_statistics(self, routing_info):
        """
        Get statistics about expert routing for analysis
        
        Args:
            routing_info: Dictionary containing routing weights for each MoE layer
            
        Returns:
            Dictionary with routing statistics
        """
        stats = {}
        
        for layer_name, routing_weights in routing_info.items():
            # routing_weights: [batch_size, num_experts]
            layer_stats = {
                'expert_usage': routing_weights.mean(dim=0).cpu().detach().numpy(),  # Average usage per expert
                'routing_entropy': self._compute_routing_entropy(routing_weights),
                'max_expert_weight': routing_weights.max(dim=1)[0].mean().item(),
                'min_expert_weight': routing_weights.min(dim=1)[0].mean().item(),
            }
            stats[layer_name] = layer_stats
            
        return stats
    
    def _compute_routing_entropy(self, routing_weights):
        """
        Compute entropy of routing distribution
        Higher entropy means more uniform expert usage
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        entropy = -(routing_weights * torch.log(routing_weights + eps)).sum(dim=1).mean()
        return entropy.item()
    
    def _log_routing_statistics(self, routing_info, moe_loss):
        """
        Log routing statistics to wandb during training
        
        Args:
            routing_info: Dictionary containing routing weights for each MoE layer
            moe_loss: Current MoE loss value
        """
        try:
            # Log overall MoE loss
            wandb.log({"moe/total_loss": moe_loss.item()})
            
            # Log statistics for each MoE layer
            for layer_name, routing_weights in routing_info.items():
                # Validate shape and fix if necessary
                if routing_weights.dim() != 2:
                    print(f"Warning: Unexpected routing weights shape for {layer_name}: {routing_weights.shape}")
                    continue
                    
                # Ensure correct shape: [batch_size, num_experts]
                if routing_weights.shape[1] != self.num_experts:
                    # If shape is [num_experts, batch_size], transpose it
                    if routing_weights.shape[0] == self.num_experts:
                        routing_weights = routing_weights.transpose(0, 1)
                        print(f"Fixed routing weights shape for {layer_name}: transposed to {routing_weights.shape}")
                    else:
                        print(f"Error: Cannot determine correct shape for {layer_name}: {routing_weights.shape}, expected [batch_size, {self.num_experts}]")
                        continue
                
                # routing_weights: [batch_size, num_experts]
                expert_usage = routing_weights.mean(dim=0).cpu().detach().numpy()  # [num_experts]
                
                # Verify we have the correct number of experts
                if len(expert_usage) != self.num_experts:
                    print(f"Error: Expert usage has {len(expert_usage)} elements, expected {self.num_experts} for {layer_name}")
                    continue
                
                # Separate universal and specialized expert statistics
                if self.num_universal_experts > 0:
                    universal_usage = expert_usage[:self.num_universal_experts]
                    specialized_usage = expert_usage[self.num_universal_experts:] if self.num_universal_experts < self.num_experts else []
                    
                    # Log universal expert usage
                    for i, usage in enumerate(universal_usage):
                        wandb.log({f"moe/{layer_name}/universal_expert_{i}_usage": usage})
                    
                    # Log universal expert statistics
                    wandb.log({
                        f"moe/{layer_name}/universal_usage_mean": universal_usage.mean(),
                        f"moe/{layer_name}/universal_usage_std": universal_usage.std(),
                        f"moe/{layer_name}/num_universal_experts": len(universal_usage),
                    })
                    
                    # Log specialized expert usage and statistics
                    if len(specialized_usage) > 0:
                        for i, usage in enumerate(specialized_usage):
                            wandb.log({f"moe/{layer_name}/specialized_expert_{i}_usage": usage})
                        
                        # Compute specialized expert statistics
                        specialized_routing_entropy = self._compute_routing_entropy(routing_weights[:, self.num_universal_experts:])
                        specialized_weights = routing_weights[:, self.num_universal_experts:]
                        
                        wandb.log({
                            f"moe/{layer_name}/specialized_routing_entropy": specialized_routing_entropy,
                            f"moe/{layer_name}/specialized_usage_mean": specialized_usage.mean(),
                            f"moe/{layer_name}/specialized_usage_std": specialized_usage.std(),
                            f"moe/{layer_name}/specialized_max_weight": specialized_weights.max(dim=1)[0].mean().item(),
                            f"moe/{layer_name}/specialized_min_weight": specialized_weights.min(dim=1)[0].mean().item(),
                            f"moe/{layer_name}/num_specialized_experts": len(specialized_usage),
                        })
                        
                        # Log specialized expert usage as histogram
                        if hasattr(wandb, 'Histogram'):
                            wandb.log({f"moe/{layer_name}/specialized_usage_hist": wandb.Histogram(specialized_usage)})
                else:
                    specialized_usage = expert_usage
                
                # Log all expert usage (universal + specialized)
                for i, usage in enumerate(expert_usage):
                    wandb.log({f"moe/{layer_name}/expert_{i}_usage": usage})
                
                # Log overall statistics
                routing_entropy = self._compute_routing_entropy(routing_weights)
                max_expert_weight = routing_weights.max(dim=1)[0].mean().item()
                min_expert_weight = routing_weights.min(dim=1)[0].mean().item()
                
                wandb.log({
                    f"moe/{layer_name}/overall_routing_entropy": routing_entropy,
                    f"moe/{layer_name}/overall_max_expert_weight": max_expert_weight,
                    f"moe/{layer_name}/overall_min_expert_weight": min_expert_weight,
                    f"moe/{layer_name}/overall_usage_std": expert_usage.std(),
                    f"moe/{layer_name}/overall_usage_mean": expert_usage.mean(),
                    f"moe/{layer_name}/total_experts": len(expert_usage),
                    f"moe/{layer_name}/universal_weight_ratio": self.universal_expert_weight,
                })
                
                # Log overall expert usage as histogram
                if hasattr(wandb, 'Histogram'):
                    wandb.log({f"moe/{layer_name}/overall_usage_hist": wandb.Histogram(expert_usage)})
                    
        except Exception as e:
            # Print error for debugging
            print(f"Error in MoE logging: {e}")
            print(f"Routing info shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in routing_info.items()]}")
            # Continue silently to not interrupt training
            pass

    def set_augmentation_strength(self, strength):
        """
        Dynamically set augmentation strength for contrastive learning
        
        Args:
            strength: Float between 0.0 and 1.0
        """
        self.augmentation_strength = max(0.0, min(1.0, strength))
    
    def get_contrastive_learning_stats(self):
        """
        Get statistics about contrastive learning for analysis
        
        Returns:
            Dictionary with contrastive learning statistics
        """
        stats = {}
        
        if self.use_contrastive_learning and hasattr(self, 'contrastive_context_learner'):
            stats['enabled'] = True
            stats['loss_weight'] = self.contrastive_loss_weight
            stats['temperature'] = self.contrastive_context_learner.temperature
            stats['use_momentum'] = self.contrastive_context_learner.use_momentum
            stats['momentum_factor'] = self.contrastive_context_learner.momentum
            stats['augmentation_strength'] = getattr(self, 'augmentation_strength', 0.1)
        else:
            stats['enabled'] = False
            
        return stats
    
    def _log_contrastive_learning_details(self, contrastive_loss, combined_invariant_cov, enriched_invariant_cov):
        """
        Log detailed contrastive learning statistics
        
        Args:
            contrastive_loss: Current contrastive loss value
            combined_invariant_cov: Original combined covariate embeddings
            enriched_invariant_cov: Enriched covariate embeddings after contrastive learning
        """
        if not (WANDB_AVAILABLE and self.training):
            return
            
        try:
            # Compute representation similarity between original and enriched
            cosine_sim = F.cosine_similarity(combined_invariant_cov, enriched_invariant_cov, dim=1)
            representation_change = torch.norm(enriched_invariant_cov - combined_invariant_cov, p=2, dim=1)
            
            # Compute representation diversity within batch
            batch_size = combined_invariant_cov.shape[0]
            if batch_size > 1:
                # Pairwise cosine similarities
                norm_enriched = F.normalize(enriched_invariant_cov, p=2, dim=1)
                similarity_matrix = torch.mm(norm_enriched, norm_enriched.t())
                # Remove diagonal elements
                mask = torch.eye(batch_size, device=similarity_matrix.device).bool()
                off_diagonal_sims = similarity_matrix[~mask]
                diversity_score = 1.0 - off_diagonal_sims.mean()
            else:
                diversity_score = torch.tensor(0.0)
            
            wandb.log({
                "contrastive/representation_similarity": cosine_sim.mean().item(),
                "contrastive/representation_change_norm": representation_change.mean().item(),
                "contrastive/batch_diversity_score": diversity_score.item(),
                "contrastive/enriched_embedding_norm": torch.norm(enriched_invariant_cov, p=2, dim=1).mean().item(),
                "contrastive/original_embedding_norm": torch.norm(combined_invariant_cov, p=2, dim=1).mean().item(),
            })
            
        except Exception as e:
            print(f"Error in detailed contrastive logging: {e}")

class ExpertNetwork(nn.Module):
    """
    Individual expert network - a smaller transformer encoder
    """
    def __init__(self, configs, expert_id=0):
        super(ExpertNetwork, self).__init__()
        self.expert_id = expert_id
        
        # Each expert has its own set of encoder layers (fewer than main model)
        self.expert_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.expert_layers if hasattr(configs, 'expert_layers') else 2)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Expert-specific projection layer
        self.expert_projection = nn.Linear(configs.d_model, configs.d_model)
        
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        Forward pass through expert network
        """
        expert_out, _ = self.expert_encoder(x, attn_mask=attn_mask, tau=tau, delta=delta)
        expert_out = self.expert_projection(expert_out)
        return expert_out


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer that uses covariate embedding for routing
    Supports universal experts that are always used plus specialized experts
    """
    def __init__(self, configs, num_experts=4, num_universal_experts=1, universal_expert_weight=0.3):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.num_universal_experts = min(num_universal_experts, num_experts)  # Can't have more universal than total
        self.num_specialized_experts = num_experts - self.num_universal_experts
        self.universal_expert_weight = universal_expert_weight
        self.configs = configs
        
        # Gating mechanism for sparsity (optional)
        self.use_sparse_gating = getattr(configs, 'use_sparse_gating', False)
        self.top_k = getattr(configs, 'top_k_experts', 2)  # Use top-k experts
        
        # Create expert networks (first num_universal_experts are universal, rest are specialized)
        self.experts = ModuleList([
            ExpertNetwork(configs, expert_id=i) for i in range(num_experts)
        ])
        
        # Router network that uses covariate embedding (only routes to specialized experts)
        if self.num_specialized_experts > 0:
            # Enhanced router with covariate importance learning
            self.cov_importance = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model),
                nn.Tanh(),
                nn.Linear(configs.d_model, configs.d_model),
                nn.Sigmoid()
            )
            
            # Multi-level routing with attention to different covariate aspects
            self.specialized_router = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model),
                nn.LayerNorm(configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model, configs.d_model // 2),
                nn.LayerNorm(configs.d_model // 2),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model // 2, self.num_specialized_experts)
                # No softmax here - will apply with temperature
            )
            
            # Hierarchical routing for expert groups
            self.expert_group_router = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model // 4),
                nn.ReLU(),
                nn.Linear(configs.d_model // 4, max(2, self.num_specialized_experts // 2)),
                nn.Softmax(dim=-1)
            )
        
            # Temperature parameter for routing sharpness
            self.routing_temperature = nn.Parameter(torch.ones(1))
            
            # FiLM conditioning for expert outputs
            self.expert_cov_scale_net = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model // 2),
                nn.ReLU(),
                nn.Linear(configs.d_model // 2, configs.d_model),
                nn.Sigmoid()
            )
            self.expert_cov_shift_net = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model // 2),
                nn.ReLU(),
                nn.Linear(configs.d_model // 2, configs.d_model)
            )

    def forward(self, x, cov_embedding, attn_mask=None, tau=None, delta=None):
        """
        Forward pass through mixture of experts with universal and specialized experts
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            cov_embedding: Covariate embedding [batch_size, d_model]
            attn_mask: Attention mask
            tau: Tau parameter for non-stationary attention
            delta: Delta parameter for non-stationary attention
        """
        batch_size, seq_len, d_model = x.shape
        
        # Initialize output
        mixed_output = torch.zeros_like(x)
        
        # 1. Universal experts - always used with fixed weight
        if self.num_universal_experts > 0:
            universal_weight = self.universal_expert_weight / self.num_universal_experts
            for i in range(self.num_universal_experts):
                expert_out = self.experts[i](x, attn_mask=attn_mask, tau=tau, delta=delta)
                mixed_output += universal_weight * expert_out
        
        # 2. Specialized experts - routed based on covariate embedding with strong conditioning
        routing_weights_for_logging = None
        load_balancing_loss = 0.0
        
        if self.num_specialized_experts > 0:
            # Learn importance of covariate features for expert routing
            cov_importance = self.cov_importance(cov_embedding)  # [batch, d_model]
            weighted_cov = cov_embedding * cov_importance  # [batch, d_model]
            
            # Compute routing weights with temperature scaling and hierarchical routing
            router_logits = self.specialized_router(weighted_cov)  # [batch, num_specialized_experts]
            router_logits = router_logits / torch.clamp(self.routing_temperature, min=0.1, max=5.0)
            
            # Hierarchical routing - modulate expert selection by group preferences
            group_weights = self.expert_group_router(weighted_cov)  # [batch, num_groups]
            # Expand group weights to match expert dimensions
            if group_weights.shape[1] < self.num_specialized_experts:
                experts_per_group = self.num_specialized_experts // group_weights.shape[1]
                group_expanded = group_weights.repeat_interleave(experts_per_group, dim=1)
                if group_expanded.shape[1] < self.num_specialized_experts:
                    remainder = self.num_specialized_experts - group_expanded.shape[1]
                    group_expanded = torch.cat([group_expanded, group_weights[:, :remainder]], dim=1)
                group_weights = group_expanded
            
            # Combine hierarchical and direct routing
            combined_logits = router_logits + 0.5 * torch.log(group_weights + 1e-8)
            specialized_routing_weights = F.softmax(combined_logits, dim=-1)
            
            # Apply sparse gating if enabled
            if self.use_sparse_gating:
                # Keep only top-k specialized experts
                top_k_weights, top_k_indices = torch.topk(specialized_routing_weights, min(self.top_k, self.num_specialized_experts), dim=-1)
                # Renormalize
                top_k_weights = F.softmax(top_k_weights, dim=-1)
                # Create sparse weight matrix
                sparse_weights = torch.zeros_like(specialized_routing_weights)
                sparse_weights.scatter_(-1, top_k_indices, top_k_weights)
                specialized_routing_weights = sparse_weights
            
            # Adjust specialized weights to use remaining weight budget
            remaining_weight = 1.0 - self.universal_expert_weight
            specialized_routing_weights = specialized_routing_weights * remaining_weight
            
            # Store for logging
            routing_weights_for_logging = specialized_routing_weights.clone()  # [batch_size, num_specialized_experts]
            
            # Get outputs from specialized experts with strong FiLM conditioning
            for i, specialist_idx in enumerate(range(self.num_universal_experts, self.num_experts)):
                expert_out = self.experts[specialist_idx](x, attn_mask=attn_mask, tau=tau, delta=delta)
                
                # Apply strong FiLM conditioning to expert output
                expert_cov_scale = self.expert_cov_scale_net(weighted_cov)  # [batch, d_model]
                expert_cov_shift = self.expert_cov_shift_net(weighted_cov)  # [batch, d_model]
                expert_cov_scale = expert_cov_scale.unsqueeze(1)  # [batch, 1, d_model]
                expert_cov_shift = expert_cov_shift.unsqueeze(1)  # [batch, 1, d_model]
                expert_out = expert_cov_scale * expert_out + expert_cov_shift
                
                # Apply routing weights: [batch_size, 1, 1] * [batch_size, seq_len, d_model]
                weighted_output = specialized_routing_weights[:, i:i+1].unsqueeze(-1) * expert_out
                mixed_output += weighted_output
            
            # Compute load balancing loss for specialized experts
            load_balancing_loss = self._compute_load_balancing_loss(specialized_routing_weights.transpose(0, 1))
        
        # Create combined routing weights for logging (universal + specialized)
        if routing_weights_for_logging is not None:
            # Create full routing weights including universal experts for logging
            full_routing_weights = torch.zeros(batch_size, self.num_experts, device=x.device)
            
            # Universal experts get fixed weights
            if self.num_universal_experts > 0:
                universal_weight_per_expert = self.universal_expert_weight / self.num_universal_experts
                full_routing_weights[:, :self.num_universal_experts] = universal_weight_per_expert
            
            # Specialized experts get routed weights
            if self.num_specialized_experts > 0:
                full_routing_weights[:, self.num_universal_experts:] = routing_weights_for_logging
                
            routing_weights_for_logging = full_routing_weights
        else:
            # Only universal experts
            routing_weights_for_logging = torch.zeros(batch_size, self.num_experts, device=x.device)
            if self.num_universal_experts > 0:
                universal_weight_per_expert = self.universal_expert_weight / self.num_universal_experts
                routing_weights_for_logging[:, :self.num_universal_experts] = universal_weight_per_expert
        
        return mixed_output, load_balancing_loss, routing_weights_for_logging
    
    def _compute_load_balancing_loss(self, routing_weights):
        """
        Compute load balancing loss to encourage uniform expert usage
        """
        # routing_weights: [num_specialized_experts, batch_size]
        expert_usage = routing_weights.mean(dim=1)  # Average usage per expert
        
        # Encourage uniform distribution across experts
        uniform_target = torch.ones_like(expert_usage) / self.num_specialized_experts
        load_loss = F.mse_loss(expert_usage, uniform_target)
        
        return load_loss

class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for learning better context representations
    """
    def __init__(self, temperature=0.1, negative_sampling='batch'):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.negative_sampling = negative_sampling  # 'batch' or 'random'
        
    def forward(self, context_embeddings, augmented_embeddings=None):
        """
        Compute InfoNCE loss for context representation learning
        
        Args:
            context_embeddings: [batch_size, d_model] - fused covariate embeddings
            augmented_embeddings: [batch_size, d_model] - optional augmented version for positive pairs
        """
        batch_size = context_embeddings.shape[0]
        device = context_embeddings.device
        
        if augmented_embeddings is None:
            # Create augmented embeddings by adding noise to the original embeddings
            noise_scale = 0.1
            augmented_embeddings = context_embeddings + noise_scale * torch.randn_like(context_embeddings)
        
        # Normalize embeddings for cosine similarity
        context_norm = F.normalize(context_embeddings, p=2, dim=1)  # [batch_size, d_model]
        augmented_norm = F.normalize(augmented_embeddings, p=2, dim=1)  # [batch_size, d_model]
        
        # Compute similarity matrix
        # Positive pairs: (context_i, augmented_i)
        # Negative pairs: (context_i, augmented_j) where i != j
        similarity_matrix = torch.mm(context_norm, augmented_norm.t()) / self.temperature  # [batch_size, batch_size]
        
        # Create labels - positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=device)
        
        # InfoNCE loss = negative log likelihood of positive pairs
        infonce_loss = F.cross_entropy(similarity_matrix, labels)
        
        return infonce_loss


class ContextAugmentation(nn.Module):
    """
    Context augmentation module for generating diverse representations
    """
    def __init__(self, d_model, dropout=0.1):
        super(ContextAugmentation, self).__init__()
        self.d_model = d_model
        
        # Learnable augmentation transformations
        self.augment_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Noise injection layers
        self.noise_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Feature masking for contrastive learning
        self.feature_mask_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Rotation transformation for invariance learning
        self.rotation_matrix = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        
        # Mixup coefficient learning
        self.mixup_weight = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, context_embeddings, augmentation_strength=0.1):
        """
        Generate augmented context embeddings with multiple augmentation strategies
        
        Args:
            context_embeddings: [batch_size, d_model]
            augmentation_strength: Strength of augmentation (0.0 to 1.0)
        """
        batch_size, d_model = context_embeddings.shape
        device = context_embeddings.device
        
        # Strategy 1: Learnable transformation
        transformed = self.augment_projection(context_embeddings)
        
        # Strategy 2: Adaptive noise injection
        noise_weights = self.noise_gate(context_embeddings)
        noise = torch.randn_like(context_embeddings) * augmentation_strength
        adaptive_noise = noise_weights * noise
        
        # Strategy 3: Feature masking (dropout-like but learnable)
        if self.training:
            mask_weights = self.feature_mask_gate(context_embeddings)
            feature_mask = torch.bernoulli(mask_weights * (1.0 - augmentation_strength))
            masked_features = context_embeddings * feature_mask
        else:
            masked_features = context_embeddings
            
        # Strategy 4: Learnable rotation for geometric augmentation
        # Normalize rotation matrix to be orthogonal (approximately)
        U, _, V = torch.svd(self.rotation_matrix)
        rotation_matrix = torch.mm(U, V.t())
        rotated_features = torch.mm(context_embeddings, rotation_matrix) * augmentation_strength
        
        # Strategy 5: Self-mixup within batch
        if batch_size > 1 and self.training:
            # Get random permutation for mixup
            perm_indices = torch.randperm(batch_size, device=device)
            mixed_contexts = context_embeddings[perm_indices]
            
            # Learnable mixup weights
            mixup_lambdas = self.mixup_weight(context_embeddings) * augmentation_strength
            mixed_features = mixup_lambdas * context_embeddings + (1 - mixup_lambdas) * mixed_contexts
        else:
            mixed_features = context_embeddings
        
        # Combine all augmentation strategies
        augmented = (context_embeddings + 
                    0.3 * transformed + 
                    adaptive_noise + 
                    0.2 * (masked_features - context_embeddings) +
                    0.1 * rotated_features +
                    0.2 * (mixed_features - context_embeddings))
        
        return augmented


class ContrastiveContextLearning(nn.Module):
    """
    Module for contrastive learning of context representations
    """
    def __init__(self, d_model, temperature=0.1, use_momentum=True, momentum=0.999):
        super(ContrastiveContextLearning, self).__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.use_momentum = use_momentum
        self.momentum = momentum
        
        # Context encoder for generating query representations
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Momentum encoder for generating key representations (if using momentum)
        if use_momentum:
            self.momentum_encoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
            # Initialize momentum encoder with same weights as context encoder
            self._initialize_momentum_encoder()
        
        # Context augmentation
        self.augmentation = ContextAugmentation(d_model)
        
        # InfoNCE loss
        self.infonce_loss = InfoNCELoss(temperature=temperature)
        
    def _initialize_momentum_encoder(self):
        """Initialize momentum encoder with context encoder weights"""
        for param_q, param_k in zip(self.context_encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    def _update_momentum_encoder(self):
        """Update momentum encoder with exponential moving average"""
        if not self.use_momentum:
            return
            
        for param_q, param_k in zip(self.context_encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    def forward(self, context_embeddings, augmentation_strength=0.1):
        """
        Perform contrastive learning on context embeddings
        
        Args:
            context_embeddings: [batch_size, d_model] - fused covariate embeddings
            augmentation_strength: Strength of data augmentation
            
        Returns:
            contrastive_loss: InfoNCE loss for representation learning
            enriched_embeddings: Enhanced context representations
        """
        batch_size = context_embeddings.shape[0]
        
        # Generate query representations
        query_embeddings = self.context_encoder(context_embeddings)  # [batch_size, d_model]
        
        # Generate augmented positive examples
        augmented_contexts = self.augmentation(context_embeddings, augmentation_strength)
        
        if self.use_momentum:
            # Generate key representations using momentum encoder
            with torch.no_grad():
                key_embeddings = self.momentum_encoder(augmented_contexts)
                # Update momentum encoder
                if self.training:
                    self._update_momentum_encoder()
        else:
            # Use the same encoder for keys
            key_embeddings = self.context_encoder(augmented_contexts)
        
        # Compute InfoNCE loss
        contrastive_loss = self.infonce_loss(query_embeddings, key_embeddings)
        
        # Enhanced representations combine original and learned features
        enriched_embeddings = context_embeddings + 0.2 * query_embeddings
        
        return contrastive_loss, enriched_embeddings
