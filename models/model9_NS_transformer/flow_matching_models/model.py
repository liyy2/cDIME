import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps, cov_dim=None):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()
        
        # Add covariate conditioning support
        self.use_cov_conditioning = cov_dim is not None
        if self.use_cov_conditioning:
            self.cov_projection = nn.Linear(cov_dim, num_out)

    def forward(self, x, t, cov_embedding=None):
        out = self.lin(x)
        gamma = self.embed(t)
        
        # Apply time conditioning
        out = gamma.view(t.size()[0], -1, self.num_out) * out
        
        # Apply covariate conditioning if available
        if self.use_cov_conditioning and cov_embedding is not None:
            cov_gamma = self.cov_projection(cov_embedding)  # [batch, num_out]
            cov_gamma = cov_gamma.unsqueeze(1)  # [batch, 1, num_out] for broadcasting
            out = out * (1 + cov_gamma)  # Multiplicative conditioning
            
        return out


class ConditionalVelocityModel(nn.Module):
    """
    Flow matching model that predicts velocity field instead of noise.
    Adapted from ConditionalGuidedModel for flow matching.
    """
    def __init__(self, config, MTS_args):
        super(ConditionalVelocityModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1  # Reuse timesteps config
        self.cat_x = config.model.cat_x
        self.cat_y_pred = config.model.cat_y_pred
        
        # Fix data_dim to match actual glucose-only channels
        # Since we now only work with glucose channel (1 channel), the input will be:
        # - y_t: [batch, pred_len, 1] (glucose only)
        # - y_0_hat: [batch, pred_len, 1] (glucose only)
        # So when concatenated: [batch, pred_len, 2]
        glucose_channels = 1  # Only glucose channel
        if self.cat_y_pred:
            data_dim = 2 * glucose_channels  # y_t + y_0_hat
        else:
            data_dim = glucose_channels  # only y_t
        
        # Add covariate conditioning support
        self.use_cov_conditioning = getattr(config.model, 'use_cov_conditioning', True)
        self.cov_dim = getattr(MTS_args, 'd_model', 512)  # Dimension of covariate embeddings

        # MoE configuration
        self.use_moe = getattr(config, 'use_moe', True)
        self.moe_num_experts = getattr(config.model, 'moe_num_experts', 4)
        self.moe_num_universal_experts = getattr(config.model, 'moe_num_universal_experts', 1)
        self.moe_universal_expert_weight = getattr(config.model, 'moe_universal_expert_weight', 0.3)
        self.moe_dropout = getattr(config, 'moe_dropout', 0.1)

        # Create ConditionalLinear layers with optional MoE
        cov_dim = self.cov_dim if self.use_cov_conditioning else None
        
        if self.use_moe:
            self.lin1 = ConditionalLinearMoE(data_dim, 128, n_steps, cov_dim=cov_dim,
                                           num_experts=self.moe_num_experts,
                                           num_universal_experts=self.moe_num_universal_experts,
                                           universal_expert_weight=self.moe_universal_expert_weight,
                                           dropout=self.moe_dropout)
            self.lin2 = ConditionalLinearMoE(128, 128, n_steps, cov_dim=cov_dim,
                                           num_experts=self.moe_num_experts,
                                           num_universal_experts=self.moe_num_universal_experts,
                                           universal_expert_weight=self.moe_universal_expert_weight,
                                           dropout=self.moe_dropout)
            self.lin3 = ConditionalLinearMoE(128, 128, n_steps, cov_dim=cov_dim,
                                           num_experts=self.moe_num_experts,
                                           num_universal_experts=self.moe_num_universal_experts,
                                           universal_expert_weight=self.moe_universal_expert_weight,
                                           dropout=self.moe_dropout)
        else:
            self.lin1 = ConditionalLinear(data_dim, 128, n_steps, cov_dim=cov_dim)
            self.lin2 = ConditionalLinear(128, 128, n_steps, cov_dim=cov_dim)
            self.lin3 = ConditionalLinear(128, 128, n_steps, cov_dim=cov_dim)
            
        self.lin4 = nn.Linear(128, glucose_channels)  # Output velocity for glucose channels only

    def forward(self, x, y_t, y_0_hat, t, cov_embedding=None):
        """
        Forward pass for velocity prediction
        
        Args:
            x: Input conditions (encoder output)
            y_t: Current state at time t
            y_0_hat: Conditional prediction (guidance)
            t: Time step
            cov_embedding: Covariate embedding for conditioning
            
        Returns:
            velocity: Predicted velocity field
            load_balancing_loss: Aggregated load balancing loss (0 if not using MoE)
        """
        # x size (batch * timesteps) * seq_len * data_dim (x is condition)
        # y_t size (batch * timesteps) * pred_len * glucose_channels (glucose only)
        # y_0_hat size (batch * timesteps) * pred_len * glucose_channels (glucose only)
        # cov_embedding size (batch * timesteps) * cov_dim (covariate conditioning)
        # velocity_pred batch * pred_len * glucose_channels
        
        if self.cat_x:
            if self.cat_y_pred:
                velocity_pred = torch.cat((y_t, y_0_hat), dim=-1)
            else:
                velocity_pred = torch.cat((y_t, x), dim=2)
        else:
            if self.cat_y_pred:
                velocity_pred = torch.cat((y_t, y_0_hat), dim=2)
            else:
                velocity_pred = y_t
        
        # Aggregate load balancing loss from MoE layers
        total_load_balancing_loss = 0.0
        routing_info = {}
            
        if self.use_moe:
            if y_t.device.type == 'mps': # mps is for macos
                velocity_pred, load_loss1, routing_weights1 = self.lin1(velocity_pred, t, cov_embedding=cov_embedding)
                velocity_pred = F.softplus(velocity_pred.cpu()).to(y_t.device)
                total_load_balancing_loss += load_loss1
                routing_info['lin1'] = routing_weights1

                velocity_pred, load_loss2, routing_weights2 = self.lin2(velocity_pred, t, cov_embedding=cov_embedding)
                velocity_pred = F.softplus(velocity_pred.cpu()).to(y_t.device)
                total_load_balancing_loss += load_loss2
                routing_info['lin2'] = routing_weights2

                velocity_pred, load_loss3, routing_weights3 = self.lin3(velocity_pred, t, cov_embedding=cov_embedding)
                velocity_pred = F.softplus(velocity_pred.cpu()).to(y_t.device)
                total_load_balancing_loss += load_loss3
                routing_info['lin3'] = routing_weights3
            else:
                velocity_pred, load_loss1, routing_weights1 = self.lin1(velocity_pred, t, cov_embedding=cov_embedding)
                velocity_pred = F.softplus(velocity_pred)
                total_load_balancing_loss += load_loss1
                routing_info['lin1'] = routing_weights1
                
                velocity_pred, load_loss2, routing_weights2 = self.lin2(velocity_pred, t, cov_embedding=cov_embedding)
                velocity_pred = F.softplus(velocity_pred)
                total_load_balancing_loss += load_loss2
                routing_info['lin2'] = routing_weights2
                
                velocity_pred, load_loss3, routing_weights3 = self.lin3(velocity_pred, t, cov_embedding=cov_embedding)
                velocity_pred = F.softplus(velocity_pred)
                total_load_balancing_loss += load_loss3
                routing_info['lin3'] = routing_weights3
        else:
            if y_t.device.type == 'mps': # mps is for macos
                velocity_pred = self.lin1(velocity_pred, t, cov_embedding=cov_embedding)
                velocity_pred = F.softplus(velocity_pred.cpu()).to(y_t.device)

                velocity_pred = self.lin2(velocity_pred, t, cov_embedding=cov_embedding)
                velocity_pred = F.softplus(velocity_pred.cpu()).to(y_t.device)

                velocity_pred = self.lin3(velocity_pred, t, cov_embedding=cov_embedding)
                velocity_pred = F.softplus(velocity_pred.cpu()).to(y_t.device)
            else:
                velocity_pred = F.softplus(self.lin1(velocity_pred, t, cov_embedding=cov_embedding))
                velocity_pred = F.softplus(self.lin2(velocity_pred, t, cov_embedding=cov_embedding))
                velocity_pred = F.softplus(self.lin3(velocity_pred, t, cov_embedding=cov_embedding))
            
        velocity_pred = self.lin4(velocity_pred)
        
        # Return both velocity and load balancing loss
        return velocity_pred, total_load_balancing_loss, routing_info


# deterministic feed forward neural network (same as diffusion)
class DeterministicFeedForwardNeuralNetwork(nn.Module):

    def __init__(self, dim_in, dim_out, hid_layers,
                 use_batchnorm=False, negative_slope=0.01, dropout_rate=0):
        super(DeterministicFeedForwardNeuralNetwork, self).__init__()
        self.dim_in = dim_in  # dimension of nn input
        self.dim_out = dim_out  # dimension of nn output
        self.hid_layers = hid_layers  # nn hidden layer architecture
        self.nn_layers = [self.dim_in] + self.hid_layers  # nn hidden layer architecture, except output layer
        self.use_batchnorm = use_batchnorm  # whether apply batch norm
        self.negative_slope = negative_slope  # negative slope for LeakyReLU
        self.dropout_rate = dropout_rate
        layers = self.create_nn_layers()
        self.network = nn.Sequential(*layers)

    def create_nn_layers(self):
        layers = []
        for idx in range(len(self.nn_layers) - 1):
            layers.append(nn.Linear(self.nn_layers[idx], self.nn_layers[idx + 1]))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.nn_layers[idx + 1]))
            layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
            layers.append(nn.Dropout(p=self.dropout_rate))
        layers.append(nn.Linear(self.nn_layers[-1], self.dim_out))
        return layers

    def forward(self, x):
        return self.network(x)


# early stopping scheme for hyperparameter tuning (same as diffusion)
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): Number of steps to wait after average improvement is below certain threshold.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement;
                           shall be a small positive value.
                           Default: 0
            best_score: value of the best metric on the validation set.
            best_epoch: epoch with the best metric on the validation set.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, val_cost, epoch, verbose=False):

        score = val_cost

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch + 1
        elif score > self.best_score - self.delta:
            self.counter += 1
            if verbose:
                print("EarlyStopping counter: {} out of {}...".format(
                    self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.counter = 0 


class ConditionalLinearMoE(nn.Module):
    """
    Mixture of Experts version of ConditionalLinear
    """
    def __init__(self, num_in, num_out, n_steps, cov_dim=None, num_experts=4, 
                 num_universal_experts=1, universal_expert_weight=0.3, dropout=0.1):
        super(ConditionalLinearMoE, self).__init__()
        self.num_out = num_out
        self.num_experts = num_experts
        self.num_universal_experts = min(num_universal_experts, num_experts)
        self.num_specialized_experts = num_experts - self.num_universal_experts
        self.universal_expert_weight = universal_expert_weight
        
        # Time embedding (shared across all experts)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()
        
        # Create expert networks
        self.experts = nn.ModuleList([
            ConditionalLinearExpert(num_in, num_out, expert_id=i) 
            for i in range(num_experts)
        ])
        
        # Add covariate conditioning support
        self.use_cov_conditioning = cov_dim is not None
        if self.use_cov_conditioning:
            self.cov_projection = nn.Linear(cov_dim, num_out)
            
            # Router network (only for specialized experts, uses covariate embedding)
            if self.num_specialized_experts > 0:
                self.specialized_router = nn.Sequential(
                    nn.Linear(cov_dim, cov_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(cov_dim // 2, self.num_specialized_experts),
                    nn.Softmax(dim=-1)
                )

    def forward(self, x, t, cov_embedding=None):
        """
        Forward pass through mixture of experts with conditional inputs
        
        Args:
            x: Input tensor [batch_size, seq_len, num_in]
            t: Time step tensor [batch_size]
            cov_embedding: Covariate embedding [batch_size, cov_dim]
        """
        batch_size = x.shape[0]
        
        # Get time conditioning (shared across all experts)
        gamma = self.embed(t)  # [batch_size, num_out]
        
        # Initialize output
        mixed_output = torch.zeros(batch_size, x.shape[1], self.num_out, 
                                 device=x.device, dtype=x.dtype)
        
        # 1. Universal experts - always used with fixed weight
        if self.num_universal_experts > 0:
            universal_weight = self.universal_expert_weight / self.num_universal_experts
            for i in range(self.num_universal_experts):
                expert_out = self.experts[i](x)
                # Apply time conditioning
                expert_out = gamma.view(batch_size, -1, self.num_out) * expert_out
                mixed_output += universal_weight * expert_out
        
        # 2. Specialized experts - routed based on covariate embedding
        routing_weights_for_logging = None
        load_balancing_loss = 0.0
        
        if self.num_specialized_experts > 0 and self.use_cov_conditioning and cov_embedding is not None:
            # Compute routing weights for specialized experts only
            specialized_routing_weights = self.specialized_router(cov_embedding)  # [batch_size, num_specialized_experts]
            
            # Adjust specialized weights to use remaining weight budget
            remaining_weight = 1.0 - self.universal_expert_weight
            specialized_routing_weights = specialized_routing_weights * remaining_weight
            
            # Store for logging
            routing_weights_for_logging = specialized_routing_weights.clone()
            
            # Get outputs from specialized experts and combine
            for i, specialist_idx in enumerate(range(self.num_universal_experts, self.num_experts)):
                expert_out = self.experts[specialist_idx](x)
                # Apply time conditioning
                expert_out = gamma.view(batch_size, -1, self.num_out) * expert_out
                # Apply routing weights: [batch_size, 1, 1] * [batch_size, seq_len, num_out]
                weighted_output = specialized_routing_weights[:, i:i+1].unsqueeze(-1) * expert_out
                mixed_output += weighted_output
            
            # Compute load balancing loss for specialized experts
            load_balancing_loss = self._compute_load_balancing_loss(specialized_routing_weights.transpose(0, 1))
        elif self.num_specialized_experts > 0:
            # No covariate conditioning available, use uniform weights for specialized experts
            remaining_weight = 1.0 - self.universal_expert_weight
            uniform_weight = remaining_weight / self.num_specialized_experts
            
            for specialist_idx in range(self.num_universal_experts, self.num_experts):
                expert_out = self.experts[specialist_idx](x)
                # Apply time conditioning
                expert_out = gamma.view(batch_size, -1, self.num_out) * expert_out
                mixed_output += uniform_weight * expert_out
        
        # Apply covariate conditioning if available
        if self.use_cov_conditioning and cov_embedding is not None:
            cov_gamma = self.cov_projection(cov_embedding)  # [batch, num_out]
            cov_gamma = cov_gamma.unsqueeze(1)  # [batch, 1, num_out] for broadcasting
            mixed_output = mixed_output * (1 + cov_gamma)  # Multiplicative conditioning
        
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
            # Only universal experts or no routing
            routing_weights_for_logging = torch.zeros(batch_size, self.num_experts, device=x.device)
            if self.num_universal_experts > 0:
                universal_weight_per_expert = self.universal_expert_weight / self.num_universal_experts
                routing_weights_for_logging[:, :self.num_universal_experts] = universal_weight_per_expert
            if self.num_specialized_experts > 0 and not self.use_cov_conditioning:
                remaining_weight = 1.0 - self.universal_expert_weight
                uniform_weight = remaining_weight / self.num_specialized_experts
                routing_weights_for_logging[:, self.num_universal_experts:] = uniform_weight
        
        return mixed_output, load_balancing_loss, routing_weights_for_logging
    
    def _compute_load_balancing_loss(self, routing_weights):
        """
        Compute load balancing loss to encourage uniform expert usage
        
        Args:
            routing_weights: [num_specialized_experts, batch_size]
        """
        expert_usage = routing_weights.mean(dim=1)  # Average usage per expert
        
        # Encourage uniform distribution across experts
        uniform_target = torch.ones_like(expert_usage) / self.num_specialized_experts
        load_loss = F.mse_loss(expert_usage, uniform_target)
        
        return load_loss


class ConditionalLinearExpert(nn.Module):
    """
    Individual expert network for ConditionalLinearMoE
    """
    def __init__(self, num_in, num_out, expert_id=0, hidden_dim=None):
        super(ConditionalLinearExpert, self).__init__()
        self.expert_id = expert_id
        
        if hidden_dim is None:
            # Simple linear expert
            self.expert = nn.Linear(num_in, num_out)
        else:
            # Expert with hidden layer for more capacity
            self.expert = nn.Sequential(
                nn.Linear(num_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_out)
            )
    
    def forward(self, x):
        return self.expert(x) 