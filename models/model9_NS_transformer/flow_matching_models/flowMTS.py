import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
import yaml
import argparse
from .flow_matching_utils import *
from .model import ConditionalVelocityModel


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class Model(nn.Module):
    """
    Flow Matching Time Series Model
    Adapted from diffusion model to use ODE-based flow matching
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        with open(configs.diffusion_config_dir, "r") as f:
            config = yaml.unsafe_load(f)
            flow_config = dict2namespace(config)

        flow_config.diffusion.timesteps = configs.timesteps  # Reuse timesteps config
        
        self.args = configs
        self.flow_config = flow_config

        # Flow matching uses fewer timesteps typically, but we keep the same interface
        self.num_timesteps = flow_config.diffusion.timesteps
        self.vis_step = flow_config.diffusion.vis_step
        self.num_figs = flow_config.diffusion.num_figs
        self.dataset_object = None

        # Flow matching doesn't need complex noise schedules like diffusion
        # We just need time steps from 0 to 1
        timesteps = torch.linspace(0, 1, self.num_timesteps)
        self.register_buffer('timesteps', timesteps)
        
        # Flow matching model predicts velocity field
        self.flow_model = ConditionalVelocityModel(flow_config, self.args)

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.CART_input_x_embed_dim, configs.embed, configs.freq,
                                           configs.dropout)

    def forward(self, x, x_mark, y, y_t, y_0_hat, t, cov_embedding=None):
        """
        Forward pass for flow matching model
        
        Args:
            x: Input sequence
            x_mark: Input time marks
            y: Target sequence (unused in forward, kept for compatibility)
            y_t: Current state at time t
            y_0_hat: Conditional prediction (guidance)
            t: Time step
            cov_embedding: Covariate embedding for conditioning
            
        Returns:
            velocity: Predicted velocity field
            load_balancing_loss: Aggregated load balancing loss (0 if not using MoE)
            routing_info: Routing information from MoE layers
        """
        enc_out = self.enc_embedding(x, x_mark)
        # Predict velocity field and get MoE losses
        velocity, load_balancing_loss, routing_info = self.flow_model(enc_out, y_t, y_0_hat, t, cov_embedding=cov_embedding)

        return velocity, load_balancing_loss, routing_info
    
    def sample(self, x, x_mark, y_0_hat, y_T_mean, num_timesteps=None, 
               solver='dopri5', rtol=1e-5, atol=1e-5, cov_embedding=None):
        """
        Generate samples using ODE solver
        
        Args:
            x: Input conditions
            x_mark: Input time marks
            y_0_hat: Conditional prediction
            y_T_mean: Prior mean at T=1
            num_timesteps: Number of timesteps for ODE solver
            solver: ODE solver method
            rtol: Relative tolerance
            atol: Absolute tolerance
            cov_embedding: Covariate embedding for conditioning
            
        Returns:
            samples: Generated samples trajectory
        """
        if num_timesteps is None:
            num_timesteps = self.num_timesteps
            
        return sample_flow_matching(
            self, x, x_mark, y_0_hat, y_T_mean, num_timesteps,
            solver=solver, rtol=rtol, atol=atol, cov_embedding=cov_embedding
        )
    
    def compute_loss(self, x, x_mark, y_0, y_0_hat, t, cov_embedding=None,
                    noise_type="gaussian", interpolation_type="linear", moe_loss_weight=0.01):
        """
        Compute flow matching loss with optional MoE load balancing loss
        
        Args:
            x: Input conditions
            x_mark: Input time marks
            y_0: Target data (clean)
            y_0_hat: Conditional prediction
            t: Sampled timesteps [0, 1]
            cov_embedding: Covariate embedding for conditioning
            noise_type: Type of noise for y_1
            interpolation_type: Type of interpolation
            moe_loss_weight: Weight for MoE load balancing loss
            
        Returns:
            total_loss: Flow matching loss + weighted load balancing loss
            loss_dict: Dictionary containing individual loss components
        """
        return flow_matching_loss_with_moe(
            self, x, x_mark, y_0, y_0_hat, t, cov_embedding=cov_embedding,
            noise_type=noise_type, interpolation_type=interpolation_type,
            moe_loss_weight=moe_loss_weight
        ) 