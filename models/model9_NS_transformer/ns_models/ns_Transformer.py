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


class TimeSeriesCovariateEncoder(nn.Module):
    """
    1D CNN encoder for time-series covariates to get time-invariant embeddings
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
        self.projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 because we concatenate avg and max pool
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
        
        # Global pooling to get time-invariant features
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # [batch_size, d_model]
        max_pool = self.global_max_pool(x).squeeze(-1)  # [batch_size, d_model]
        
        # Concatenate and project
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # [batch_size, d_model * 2]
        time_invariant_embedding = self.projection(pooled)  # [batch_size, d_model]
        
        return time_invariant_embedding


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
        x_ts_covariates = x_enc[:, :, -self.glucose_channels:]  if self.ts_covariate_channels == 1 else x_enc[:, :, :] # [B, seq_len, C-1] (historical only)
        
        # For decoder, ONLY extract glucose from x_dec (no time-series covariates to prevent leakage)
        x_dec_glucose = x_dec[:, :, -self.glucose_channels:]  # [B, label_len + pred_len, 1]
        
        # Process tabular covariates (time-invariant, no leakage concern)
        tabular_cov_embedding = self.cov_encoder(covariates)  # [B, d_model]
        tabular_cov_embedding = tabular_cov_embedding.reshape(tabular_cov_embedding.shape[0], -1) # [B, 192]
        # Process time-series covariates ONLY from historical encoder data (no future data)
        if self.ts_covariate_channels > 0 and x_ts_covariates is not None:
            # Use only historical time-series covariates from encoder to get time-invariant embedding
            ts_cov_embedding = self.ts_cov_encoder(x_ts_covariates)  # [B, d_model]
            # Fuse tabular and time-series covariate embeddings
            combined_cov_embedding = self.covariate_fusion(
                torch.cat([tabular_cov_embedding, ts_cov_embedding], dim=1)
            )  # [B, d_model]
        else:
            # Use only tabular covariates if no time-series covariates
            combined_cov_embedding = tabular_cov_embedding
            
        x_raw = x_glucose.clone().detach()  # Use only glucose for raw processing

        # Normalization (only for glucose)
        mean_enc = x_glucose.mean(1, keepdim=True).detach()  # B x 1 x 1
        x_glucose_norm = x_glucose - mean_enc
        std_enc = torch.sqrt(torch.var(x_glucose_norm, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x 1
        x_glucose_norm = x_glucose_norm / std_enc
        
        # Create decoder input with normalized glucose only (NO time-series covariates)
        # This ensures decoder only processes glucose and uses covariate embeddings as conditioning
        x_dec_new = torch.cat([x_glucose_norm[:, -self.label_len:, :], 
                              torch.zeros_like(x_dec_glucose[:, -self.pred_len:, :])],
                              dim=1).to(x_glucose.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x 1, B x 1 x 1 -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, mean_enc)  # B x S x 1, B x 1 x 1 -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x_glucose_norm, x_mark_enc)
        # Add covariate embeddings as conditioning (time-invariant conditioning for VAE)
        enc_out = enc_out + combined_cov_embedding.unsqueeze(1)
        
        # Encoder processing with MoE
        total_moe_loss = 0.0
        routing_info = {}
        
        if self.use_moe:
            # Process through encoder layers with MoE integration
            regular_layer_idx = 0
            attns = []
            
            for layer_idx in range(self.total_encoder_layers):
                if layer_idx in self.moe_layer_indices:
                    # Use MoE layer
                    moe_layer = self.moe_layers[str(layer_idx)]
                    enc_out, moe_loss, routing_weights = moe_layer(
                        enc_out, combined_cov_embedding, attn_mask=enc_self_mask, tau=tau, delta=delta
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
        # Add same covariate conditioning to decoder (time-invariant conditioning for VAE)
        dec_out = dec_out + combined_cov_embedding.unsqueeze(1)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

        # De-normalization
        dec_out = dec_out * std_enc + mean_enc

        # Combine KL loss with MoE loss for compatibility
        combined_KL_z = KL_z + self.moe_loss_weight * total_moe_loss

        # Log routing statistics to wandb if available and enabled
        if self.use_moe and self.log_routing_stats and WANDB_AVAILABLE and routing_info:
            self._log_routing_statistics(routing_info, total_moe_loss)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :], dec_out, combined_KL_z, z_sample, combined_cov_embedding  # [B, L, D]

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
        
        # Create expert networks (first num_universal_experts are universal, rest are specialized)
        self.experts = ModuleList([
            ExpertNetwork(configs, expert_id=i) for i in range(num_experts)
        ])
        
        # Router network that uses covariate embedding (only routes to specialized experts)
        if self.num_specialized_experts > 0:
            self.specialized_router = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model // 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model // 2, self.num_specialized_experts),
                nn.Softmax(dim=-1)
            )
        
        # Gating mechanism for sparsity (optional, only for specialized experts)
        self.use_sparse_gating = getattr(configs, 'use_sparse_gating', False)
        self.top_k = getattr(configs, 'top_k_experts', 2)  # Use top-k experts

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
        
        # 2. Specialized experts - routed based on covariate embedding
        routing_weights_for_logging = None
        load_balancing_loss = 0.0
        
        if self.num_specialized_experts > 0:
            # Compute routing weights for specialized experts only
            specialized_routing_weights = self.specialized_router(cov_embedding)  # [batch_size, num_specialized_experts]
            
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
            
            # Get outputs from specialized experts and combine
            for i, specialist_idx in enumerate(range(self.num_universal_experts, self.num_experts)):
                expert_out = self.experts[specialist_idx](x, attn_mask=attn_mask, tau=tau, delta=delta)
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
