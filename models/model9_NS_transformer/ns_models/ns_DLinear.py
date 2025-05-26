import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List

from torch import Tensor
from torch.nn import Linear, Module, ModuleList

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.conv import TabTransformerConv
from torch_frame.nn.encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeWiseFeatureEncoder,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Routing statistics will not be logged.")


def configure_moe(configs, use_moe=True, num_experts=4, 
                  moe_loss_weight=0.01, log_routing_stats=True, num_universal_experts=1,
                  universal_expert_weight=0.3, d_model=64):
    """
    Helper function to configure Mixture of Experts parameters for DLinear
    
    Args:
        configs: Model configuration object
        use_moe: Whether to use MoE (default: True)
        num_experts: Number of expert networks (default: 4)
        moe_loss_weight: Weight for MoE loss in the combined KL loss (default: 0.01)
        log_routing_stats: Whether to log routing statistics to wandb (default: True)
        num_universal_experts: Number of universal experts that are always used (default: 1)
        universal_expert_weight: Weight given to universal experts (default: 0.3)
        d_model: Model dimension for embeddings (default: 64)
    """
    configs.use_moe = use_moe
    configs.num_experts = num_experts
    configs.moe_loss_weight = moe_loss_weight
    configs.log_routing_stats = log_routing_stats
    configs.num_universal_experts = num_universal_experts
    configs.universal_expert_weight = universal_expert_weight
    configs.d_model = d_model
    
    return configs


class ExampleTransformer(Module):
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
    

class HeadDropout(nn.Module):
    def __init__(self, p=0.5):
        super(HeadDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def forward(self, x):
        # If in evaluation mode, return the input as-is
        if not self.training:
            return x
        
        # Create a binary mask of the same shape as x
        binary_mask = (torch.rand_like(x) > self.p).float()
        
        # Set dropped values to negative infinity during training
        return x * binary_mask + (1 - binary_mask) * -1e20


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinearEncoder(nn.Module):
    """
    DLinear-based encoder that maps from seq_len to latent_len
    """
    def __init__(self, seq_len, latent_len, kernel_size=25):
        super(DLinearEncoder, self).__init__()
        self.seq_len = seq_len
        self.latent_len = latent_len
        
        # Decomposition
        self.decomposition = series_decomp(kernel_size)
        
        # Linear layers for encoding to latent space
        self.seasonal_encoder = nn.Linear(seq_len, latent_len)
        self.trend_encoder = nn.Linear(seq_len, latent_len)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, 1] - glucose time series
        Returns:
            encoded: [batch_size, latent_len, 1] - encoded latent time series
        """
        # Decompose
        seasonal, trend = self.decomposition(x)  # [B, seq_len, 1]
        
        # Permute for linear layers
        seasonal = seasonal.permute(0, 2, 1)  # [B, 1, seq_len]
        trend = trend.permute(0, 2, 1)  # [B, 1, seq_len]
        
        # Encode to latent space
        seasonal_encoded = self.seasonal_encoder(seasonal)  # [B, 1, latent_len]
        trend_encoded = self.trend_encoder(trend)  # [B, 1, latent_len]
        
        # Combine and permute back
        encoded = (seasonal_encoded + trend_encoded).permute(0, 2, 1)  # [B, latent_len, 1]
        
        return encoded


class DLinearDecoder(nn.Module):
    """
    DLinear-based decoder that maps from latent_len to pred_len
    """
    def __init__(self, latent_len, pred_len, kernel_size=25):
        super(DLinearDecoder, self).__init__()
        self.latent_len = latent_len
        self.pred_len = pred_len
        
        # Decomposition for latent space
        self.decomposition = series_decomp(kernel_size)
        
        # Linear layers for decoding to prediction space
        self.seasonal_decoder = nn.Linear(latent_len, pred_len)
        self.trend_decoder = nn.Linear(latent_len, pred_len)
        
    def forward(self, z):
        """
        Args:
            z: [batch_size, latent_len, 1] - latent time series
        Returns:
            decoded: [batch_size, pred_len, 1] - decoded prediction
        """
        # Decompose latent
        seasonal, trend = self.decomposition(z)  # [B, latent_len, 1]
        
        # Permute for linear layers
        seasonal = seasonal.permute(0, 2, 1)  # [B, 1, latent_len]
        trend = trend.permute(0, 2, 1)  # [B, 1, latent_len]
        
        # Decode to prediction space
        seasonal_decoded = self.seasonal_decoder(seasonal)  # [B, 1, pred_len]
        trend_decoded = self.trend_decoder(trend)  # [B, 1, pred_len]
        
        # Combine and permute back
        decoded = (seasonal_decoded + trend_decoded).permute(0, 2, 1)  # [B, pred_len, 1]
        
        return decoded


class ExpertNetwork(nn.Module):
    """
    Individual expert network for DLinear VAE - operates on latent space
    """
    def __init__(self, latent_len, pred_len, expert_id=0, hidden_dim=64):
        super(ExpertNetwork, self).__init__()
        self.expert_id = expert_id
        self.latent_len = latent_len
        self.pred_len = pred_len
        
        # Expert-specific decoder
        self.expert_decoder = DLinearDecoder(latent_len, pred_len)
        
        # Additional expert-specific processing
        self.expert_projection = nn.Sequential(
            nn.Linear(pred_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, pred_len)
        )
        
    def forward(self, z):
        """
        Forward pass through expert network
        Args:
            z: [batch_size, latent_len, 1] - latent time series
        Returns:
            expert_output: [batch_size, pred_len, 1]
        """
        # Decode latent to prediction
        decoded = self.expert_decoder(z)  # [B, pred_len, 1]
        
        # Expert-specific processing
        decoded_permuted = decoded.permute(0, 2, 1)  # [B, 1, pred_len]
        processed = self.expert_projection(decoded_permuted)  # [B, 1, pred_len]
        expert_output = processed.permute(0, 2, 1)  # [B, pred_len, 1]
        
        return expert_output


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer for DLinear VAE that operates on latent space
    """
    def __init__(self, latent_len, pred_len, num_experts=4, cov_dim=32, wearable_dim=32,
                 num_universal_experts=1, universal_expert_weight=0.3, dropout=0.1):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.num_universal_experts = min(num_universal_experts, num_experts)
        self.num_specialized_experts = num_experts - self.num_universal_experts
        self.universal_expert_weight = universal_expert_weight
        
        # Create expert networks
        self.experts = ModuleList([
            ExpertNetwork(latent_len, pred_len, expert_id=i) for i in range(num_experts)
        ])
        
        # Router network that uses both covariate and wearable embeddings (only routes to specialized experts)
        if self.num_specialized_experts > 0:
            router_input_dim = cov_dim + wearable_dim  # Combine both embeddings
            self.specialized_router = nn.Sequential(
                nn.Linear(router_input_dim, router_input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(router_input_dim // 2, self.num_specialized_experts),
                nn.Softmax(dim=-1)
            )

    def forward(self, z, cov_embedding, wearable_embedding):
        """
        Forward pass through mixture of experts
        
        Args:
            z: [batch_size, latent_len, 1] - latent time series
            cov_embedding: [batch_size, cov_dim] - covariate embedding
            wearable_embedding: [batch_size, wearable_dim] - wearable embedding
        """
        batch_size = z.shape[0]
        
        # Initialize output
        mixed_output = torch.zeros(batch_size, self.experts[0].pred_len, 1, 
                                 device=z.device, dtype=z.dtype)
        
        # 1. Universal experts - always used with fixed weight
        if self.num_universal_experts > 0:
            universal_weight = self.universal_expert_weight / self.num_universal_experts
            for i in range(self.num_universal_experts):
                expert_out = self.experts[i](z)
                mixed_output += universal_weight * expert_out
        
        # 2. Specialized experts - routed based on combined covariate and wearable embeddings
        routing_weights_for_logging = None
        load_balancing_loss = 0.0
        
        if self.num_specialized_experts > 0:
            # Combine covariate and wearable embeddings for routing
            combined_embedding = torch.cat([cov_embedding, wearable_embedding], dim=1)  # [batch_size, cov_dim + wearable_dim]
            
            # Compute routing weights for specialized experts only
            specialized_routing_weights = self.specialized_router(combined_embedding)  # [batch_size, num_specialized_experts]
            
            # Adjust specialized weights to use remaining weight budget
            remaining_weight = 1.0 - self.universal_expert_weight
            specialized_routing_weights = specialized_routing_weights * remaining_weight
            
            # Store for logging
            routing_weights_for_logging = specialized_routing_weights.clone()
            
            # Get outputs from specialized experts and combine
            for i, specialist_idx in enumerate(range(self.num_universal_experts, self.num_experts)):
                expert_out = self.experts[specialist_idx](z)
                # Apply routing weights: [batch_size, 1, 1] * [batch_size, pred_len, 1]
                weighted_output = specialized_routing_weights[:, i:i+1].unsqueeze(-1) * expert_out
                mixed_output += weighted_output
            
            # Compute load balancing loss for specialized experts
            load_balancing_loss = self._compute_load_balancing_loss(specialized_routing_weights.transpose(0, 1))
        
        # Create combined routing weights for logging (universal + specialized)
        if routing_weights_for_logging is not None:
            # Create full routing weights including universal experts for logging
            full_routing_weights = torch.zeros(batch_size, self.num_experts, device=z.device)
            
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
            routing_weights_for_logging = torch.zeros(batch_size, self.num_experts, device=z.device)
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


class Model(nn.Module):
    """
    Time Series VAE with DLinear Encoder/Decoder and Mixture of Experts
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.num_experts = getattr(configs, 'num_experts', 4)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = getattr(configs, 'label_len', self.seq_len) # Default label_len to seq_len if not specified
        self.channels = configs.enc_in
        
        # VAE configuration
        self.latent_len = getattr(configs, 'latent_len', self.seq_len // 2)  # Latent time series length
        latent_feature_dim = 1  # The feature dimension of our latent series [B, latent_len, 1]
        vae_hidden_dim = getattr(configs, 'vae_hidden_dim', 16) # Hidden dim for z_mean/z_logvar nets

        self.use_moe = getattr(configs, 'use_moe', True)
        self.moe_loss_weight = getattr(configs, 'moe_loss_weight', 0.01)
        self.log_routing_stats = getattr(configs, 'log_routing_stats', True)
        
        # Universal experts configuration
        self.num_universal_experts = getattr(configs, 'num_universal_experts', 1)
        self.universal_expert_weight = getattr(configs, 'universal_expert_weight', 0.3)

        # Time feature size
        self.expected_time_features = 4 if configs.freq.lower().endswith('h') else 5

        # Covariate encoder (time-invariant)
        self.cov_encoder = ExampleTransformer(
            channels=32,
            out_channels=32,
            num_layers=2,
            num_heads=8,
            col_stats=configs.col_stats,
            col_names_dict=configs.col_names_dict,
        )
        
        # Wearable feature encoder (time-invariant from time series)
        self.wearable_encoder = nn.Sequential(
            nn.Linear((self.channels - 1) * self.seq_len, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32)
        )

        # VAE Encoder: DLinear that maps glucose time series to latent time series
        self.encoder = DLinearEncoder(self.seq_len, self.latent_len)
        
        # VAE latent variable networks (Linear layers operating on the feature dim of latent series)
        self.z_mean_net = nn.Sequential(
            nn.Linear(latent_feature_dim, vae_hidden_dim),
            nn.ReLU(),
            nn.Linear(vae_hidden_dim, latent_feature_dim)
        )
        
        self.z_logvar_net = nn.Sequential(
            nn.Linear(latent_feature_dim, vae_hidden_dim),
            nn.ReLU(),
            nn.Linear(vae_hidden_dim, latent_feature_dim)
        )

        if self.use_moe:
            # Mixture of Experts for decoding
            self.moe_decoder = MixtureOfExperts(
                latent_len=self.latent_len,
                pred_len=self.pred_len,
                num_experts=self.num_experts,
                cov_dim=32,
                wearable_dim=32,
                num_universal_experts=self.num_universal_experts,
                universal_expert_weight=self.universal_expert_weight,
                dropout=getattr(configs, 'dropout', 0.1)
            )
        else:
            # Standard DLinear decoder
            self.decoder = DLinearDecoder(self.latent_len, self.pred_len)

        # Final conditioning layer that combines covariate information
        self.final_conditioning = nn.Sequential(
            nn.Linear(self.expected_time_features + 32 + 32, 64),  # time + wearable + cov
            nn.ReLU(),
            nn.Linear(64, self.pred_len)
        )

    def KL_loss_normal(self, posterior_mean, posterior_logvar):
        """Compute KL divergence loss for VAE (time series version)"""
        # posterior_mean, posterior_logvar: [batch_size, latent_len, 1]
        KL = -0.5 * torch.mean(1 - posterior_mean ** 2 + posterior_logvar -
                               torch.exp(posterior_logvar), dim=[1, 2])  # Sum over time and channel dims
        return torch.mean(KL)

    def reparameterize(self, posterior_mean, posterior_logvar):
        """Reparameterization trick for VAE (time series version)"""
        posterior_var = posterior_logvar.exp()
        if self.training:
            # Sample for time series
            eps = torch.randn_like(posterior_var)
            z = posterior_mean + posterior_var.sqrt() * eps
        else:
            z = posterior_mean
        return z

    def _compute_routing_entropy(self, routing_weights):
        """Compute entropy of routing distribution"""
        eps = 1e-8
        entropy = -(routing_weights * torch.log(routing_weights + eps)).sum(dim=1).mean()
        return entropy.item()

    def _log_routing_statistics(self, routing_info, moe_loss):
        """Log routing statistics to wandb during training"""
        if not WANDB_AVAILABLE:
            return
            
        try:
            # Log overall MoE loss
            wandb.log({"moe/total_loss": moe_loss.item()})
            
            # Log statistics for MoE layer
            for layer_name, routing_weights in routing_info.items():
                if routing_weights.dim() != 2:
                    continue
                    
                if routing_weights.shape[1] != self.num_experts:
                    if routing_weights.shape[0] == self.num_experts:
                        routing_weights = routing_weights.transpose(0, 1)
                    else:
                        continue
                
                expert_usage = routing_weights.mean(dim=0).cpu().detach().numpy()
                
                if len(expert_usage) != self.num_experts:
                    continue
                
                # Log expert usage
                for i, usage in enumerate(expert_usage):
                    wandb.log({f"moe/{layer_name}/expert_{i}_usage": usage})
                
                # Log overall statistics
                routing_entropy = self._compute_routing_entropy(routing_weights)
                wandb.log({
                    f"moe/{layer_name}/routing_entropy": routing_entropy,
                    f"moe/{layer_name}/usage_std": expert_usage.std(),
                    f"moe/{layer_name}/usage_mean": expert_usage.mean(),
                })
                    
        except Exception as e:
            print(f"Error in MoE logging: {e}")
            pass

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, covariates=None, 
                return_gating_weights=False, return_seperate_head=False):
        
        # Extract glucose and wearable data
        x_glucose = x_enc[:, :, -1].unsqueeze(-1)  # [B, seq_len, 1]
        x_wearable = x_enc[:, :, :-1]  # [B, seq_len, C-1]
        x_mark_initial = x_mark_enc[:, 0]  # [B, time_features]
        
        # Process covariates (time-invariant)
        cov_embedding = self.cov_encoder(covariates)  # [B, 32]
        
        # Process wearable features (time-invariant from time series)
        wearable_feature = self.wearable_encoder(
            x_wearable.reshape(-1, (self.channels - 1) * self.seq_len)
        )  # [B, 32]
        
        # VAE Encoding: glucose time series -> latent time series
        encoded = self.encoder(x_glucose)  # [B, latent_len, 1]
        
        # Get latent variables (time series) - Apply networks directly to the feature dimension
        z_mean = self.z_mean_net(encoded)  # Operates on the last dim: [B, latent_len, 1]
        z_logvar = self.z_logvar_net(encoded)  # Operates on the last dim: [B, latent_len, 1]
        
        # Reparameterization
        z_sample = self.reparameterize(z_mean, z_logvar)  # [B, latent_len, 1]
        
        # Compute KL loss
        KL_z = self.KL_loss_normal(z_mean, z_logvar)

        # Decoding with MoE or standard decoder
        total_moe_loss = 0.0
        routing_info = {}
        
        if self.use_moe:
            # Use Mixture of Experts for decoding
            decoded, moe_loss, routing_weights = self.moe_decoder(z_sample, cov_embedding, wearable_feature)
            total_moe_loss = moe_loss
            routing_info['moe_decoder'] = routing_weights
        else:
            # Standard decoding
            decoded = self.decoder(z_sample)  # [B, pred_len, 1]

        # Final conditioning with time-invariant features
        conditioning_features = torch.cat([x_mark_initial, wearable_feature, cov_embedding], dim=1)
        conditioning_adjustment = self.final_conditioning(conditioning_features)  # [B, pred_len]
        
        # Combine decoded output with conditioning
        prediction_output = decoded.squeeze(-1) + conditioning_adjustment  # [B, pred_len]
        prediction_output = prediction_output.unsqueeze(-1)  # [B, pred_len, 1]

        # Create full output sequence: [label_part + prediction_part]
        # Extract label part from decoder input (which contains the label part)
        label_part = x_dec[:, :self.label_len, -1:]  # [B, label_len, 1]
        
        # Concatenate label part with our prediction
        full_output = torch.cat([label_part, prediction_output], dim=1)  # [B, label_len + pred_len, 1]

        # Combine KL loss with MoE loss
        combined_KL_z = KL_z + self.moe_loss_weight * total_moe_loss

        # Log routing statistics
        if self.use_moe and self.log_routing_stats and routing_info:
            self._log_routing_statistics(routing_info, total_moe_loss)

        return full_output[:, -self.pred_len:, :], full_output, combined_KL_z, z_sample, cov_embedding

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