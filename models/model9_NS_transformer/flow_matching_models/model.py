import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps, cov_dim=None):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        
        # Better time embedding initialization
        self.embed = nn.Embedding(n_steps, num_out)
        # Use Xavier normal initialization instead of uniform
        nn.init.xavier_normal_(self.embed.weight)
        
        # Add layer normalization for stability
        self.layer_norm = nn.LayerNorm(num_out)
        
        # STRONG covariate conditioning with FiLM (Feature-wise Linear Modulation)
        self.use_cov_conditioning = cov_dim is not None
        if self.use_cov_conditioning:
            # FiLM conditioning - separate scale and shift for each feature
            self.cov_scale_net = nn.Sequential(
                nn.Linear(cov_dim, cov_dim // 2),
                nn.ReLU(),
                nn.Linear(cov_dim // 2, num_out),
                nn.Sigmoid()  # Ensure positive scales
            )
            self.cov_shift_net = nn.Sequential(
                nn.Linear(cov_dim, cov_dim // 2),
                nn.ReLU(),
                nn.Linear(cov_dim // 2, num_out)
            )
            
            # Covariate feature importance learning
            self.cov_importance = nn.Sequential(
                nn.Linear(cov_dim, cov_dim),
                nn.Tanh(),
                nn.Linear(cov_dim, cov_dim),
                nn.Sigmoid()
            )
            
            # Initialize weights properly
            nn.init.xavier_normal_(self.cov_scale_net[0].weight)
            nn.init.xavier_normal_(self.cov_shift_net[0].weight)
            nn.init.xavier_normal_(self.cov_importance[0].weight)

    def forward(self, x, t, cov_embedding=None, dynamic_cov_tokens=None):
        out = self.lin(x)
        
        # Apply time conditioning (additive + multiplicative)
        gamma = self.embed(t)  # [batch, num_out]
        time_scale = torch.sigmoid(gamma)  # Use sigmoid to keep scale positive and bounded
        out = gamma.view(t.size()[0], -1, self.num_out) + time_scale.view(t.size()[0], -1, self.num_out) * out
        
        # STRONG covariate conditioning
        if self.use_cov_conditioning and cov_embedding is not None:
            batch_size = out.shape[0]
            
            # 1. Learn importance of different covariate features (static covariates only)
            cov_importance = self.cov_importance(cov_embedding)  # [batch, cov_dim]
            weighted_cov = cov_embedding * cov_importance  # [batch, cov_dim]
            
            # 2. FiLM conditioning - feature-wise linear modulation (using static covariates)
            cov_scale = self.cov_scale_net(weighted_cov)  # [batch, num_out]
            cov_shift = self.cov_shift_net(weighted_cov)  # [batch, num_out]
            
            # Apply FiLM: γ(cov) ⊙ x + β(cov)
            cov_scale = cov_scale.unsqueeze(1)  # [batch, 1, num_out]
            cov_shift = cov_shift.unsqueeze(1)  # [batch, 1, num_out]
            out = cov_scale * out + cov_shift
        
        # Apply layer normalization for stability
        out = self.layer_norm(out)
            
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
        
        # Increased model capacity: deeper and wider network
        hidden_dims = [64, 64, 64]  # Reduced to 3 layers
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Add gradient scaling factor (learnable parameter)
        self.gradient_scale = nn.Parameter(torch.ones(1))
        
        if self.use_moe:
            self.lin1 = ConditionalLinear(data_dim, hidden_dims[0], n_steps, cov_dim=cov_dim)
            self.lin2 = ConditionalLinearMoE(hidden_dims[0], hidden_dims[1], n_steps, cov_dim=cov_dim,
                                           num_experts=self.moe_num_experts,
                                           num_universal_experts=self.moe_num_universal_experts,
                                           universal_expert_weight=self.moe_universal_expert_weight,
                                           dropout=self.moe_dropout)
            self.lin3 = ConditionalLinear(hidden_dims[1], hidden_dims[2], n_steps, cov_dim=cov_dim)
        else:
            self.lin1 = ConditionalLinear(data_dim, hidden_dims[0], n_steps, cov_dim=cov_dim)
            self.lin2 = ConditionalLinear(hidden_dims[0], hidden_dims[1], n_steps, cov_dim=cov_dim)
            self.lin3 = ConditionalLinear(hidden_dims[1], hidden_dims[2], n_steps, cov_dim=cov_dim)
            
        self.lin4 = nn.Linear(hidden_dims[2], glucose_channels)  # Final output layer (non-MoE)

    def forward(self, x, y_t, y_0_hat, t, cov_embedding=None, dynamic_cov_tokens=None):
        """
        Forward pass for velocity prediction
        
        Args:
            x: Input conditions (encoder output)
            y_t: Current state at time t
            y_0_hat: Conditional prediction (guidance)
            t: Time step
            cov_embedding: Static covariate embedding for conditioning [batch, cov_dim]
            dynamic_cov_tokens: Dynamic covariate tokens [batch, seq_len, cov_dim]
            
        Returns:
            velocity: Predicted velocity field
            load_balancing_loss: Aggregated load balancing loss (0 if not using MoE)
        """
        # x size (batch * timesteps) * seq_len * data_dim (x is condition)
        # y_t size (batch * timesteps) * pred_len * glucose_channels (glucose only)
        # y_0_hat size (batch * timesteps) * pred_len * glucose_channels (glucose only)
        # cov_embedding size (batch * timesteps) * cov_dim (static covariate conditioning)
        # dynamic_cov_tokens size (batch * timesteps) * seq_len * cov_dim (dynamic covariate tokens)
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
                # lin1: Regular ConditionalLinear
                velocity_pred = self.lin1(velocity_pred, t, cov_embedding=cov_embedding, dynamic_cov_tokens=dynamic_cov_tokens)
                velocity_pred = F.gelu(velocity_pred.cpu()).to(y_t.device)
                velocity_pred = self.dropout(velocity_pred)

                # lin2: MoE layer
                velocity_pred, load_loss2, routing_weights2 = self.lin2(velocity_pred, t, cov_embedding=cov_embedding, dynamic_cov_tokens=dynamic_cov_tokens)
                velocity_pred = F.gelu(velocity_pred.cpu()).to(y_t.device)
                velocity_pred = self.dropout(velocity_pred)
                total_load_balancing_loss += load_loss2
                routing_info['lin2'] = routing_weights2

                # lin3: Regular ConditionalLinear
                velocity_pred = self.lin3(velocity_pred, t, cov_embedding=cov_embedding, dynamic_cov_tokens=dynamic_cov_tokens)
                velocity_pred = F.gelu(velocity_pred.cpu()).to(y_t.device)
            else:
                # lin1: Regular ConditionalLinear
                velocity_pred = F.gelu(self.lin1(velocity_pred, t, cov_embedding=cov_embedding, dynamic_cov_tokens=dynamic_cov_tokens))
                velocity_pred = self.dropout(velocity_pred)
                
                # lin2: MoE layer
                velocity_pred, load_loss2, routing_weights2 = self.lin2(velocity_pred, t, cov_embedding=cov_embedding, dynamic_cov_tokens=dynamic_cov_tokens)
                velocity_pred = F.gelu(velocity_pred)
                velocity_pred = self.dropout(velocity_pred)
                total_load_balancing_loss += load_loss2
                routing_info['lin2'] = routing_weights2
                
                # lin3: Regular ConditionalLinear
                velocity_pred = F.gelu(self.lin3(velocity_pred, t, cov_embedding=cov_embedding, dynamic_cov_tokens=dynamic_cov_tokens))
        else:
            if y_t.device.type == 'mps': # mps is for macos
                velocity_pred = self.lin1(velocity_pred, t, cov_embedding=cov_embedding, dynamic_cov_tokens=dynamic_cov_tokens)
                velocity_pred = F.gelu(velocity_pred.cpu()).to(y_t.device)
                velocity_pred = self.dropout(velocity_pred)

                velocity_pred = self.lin2(velocity_pred, t, cov_embedding=cov_embedding, dynamic_cov_tokens=dynamic_cov_tokens)
                velocity_pred = F.gelu(velocity_pred.cpu()).to(y_t.device)
                velocity_pred = self.dropout(velocity_pred)

                velocity_pred = self.lin3(velocity_pred, t, cov_embedding=cov_embedding, dynamic_cov_tokens=dynamic_cov_tokens)
                velocity_pred = F.gelu(velocity_pred.cpu()).to(y_t.device)
            else:
                velocity_pred = F.gelu(self.lin1(velocity_pred, t, cov_embedding=cov_embedding, dynamic_cov_tokens=dynamic_cov_tokens))
                velocity_pred = self.dropout(velocity_pred)
                velocity_pred = F.gelu(self.lin2(velocity_pred, t, cov_embedding=cov_embedding, dynamic_cov_tokens=dynamic_cov_tokens))
                velocity_pred = self.dropout(velocity_pred)
                velocity_pred = F.gelu(self.lin3(velocity_pred, t, cov_embedding=cov_embedding, dynamic_cov_tokens=dynamic_cov_tokens))
            
        # Apply gradient scaling and final projection
        velocity_pred = self.lin4(velocity_pred) * self.gradient_scale
        
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
            # STRONG covariate conditioning for MoE with FiLM
            self.cov_scale_net = nn.Sequential(
                nn.Linear(cov_dim, cov_dim // 2),
                nn.ReLU(),
                nn.Linear(cov_dim // 2, num_out),
                nn.Sigmoid()  # Ensure positive scales
            )
            self.cov_shift_net = nn.Sequential(
                nn.Linear(cov_dim, cov_dim // 2),
                nn.ReLU(),
                nn.Linear(cov_dim // 2, num_out)
            )
            
            # Covariate feature importance learning
            self.cov_importance = nn.Sequential(
                nn.Linear(cov_dim, cov_dim),
                nn.Tanh(),
                nn.Linear(cov_dim, cov_dim),
                nn.Sigmoid()
            )
            
            # Initialize weights properly
            nn.init.xavier_normal_(self.cov_scale_net[0].weight)
            nn.init.xavier_normal_(self.cov_shift_net[0].weight)
            nn.init.xavier_normal_(self.cov_importance[0].weight)
            
            # Improved router network with multiple routing strategies
            if self.num_specialized_experts > 0:
                # Simplified single-level routing
                self.specialized_router = nn.Sequential(
                    nn.Linear(cov_dim, cov_dim // 2),
                    nn.LayerNorm(cov_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(cov_dim // 2, self.num_specialized_experts)
                )
                
                # Temperature parameter for routing sharpness (learnable)
                self.routing_temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, t, cov_embedding=None, dynamic_cov_tokens=None):
        """
        Forward pass through mixture of experts with conditional inputs
        
        Args:
            x: Input tensor [batch_size, seq_len, num_in]
            t: Time step tensor [batch_size]
            cov_embedding: Static covariate embedding [batch_size, cov_dim]
            dynamic_cov_tokens: Dynamic covariate tokens [batch_size, seq_len, cov_dim]
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
        
        # 2. Specialized experts - routed based on static covariate embedding
        routing_weights_for_logging = None
        load_balancing_loss = 0.0
        
        if self.num_specialized_experts > 0 and self.use_cov_conditioning and cov_embedding is not None:
            # Learn importance of different covariate features for routing (static only)
            cov_importance = self.cov_importance(cov_embedding)  # [batch, cov_dim]
            weighted_cov = cov_embedding * cov_importance  # [batch, cov_dim]
            
            # Simplified routing - remove expensive hierarchical routing
            router_logits = self.specialized_router(weighted_cov)  # [batch, num_specialized_experts]
            router_logits = router_logits / torch.clamp(self.routing_temperature, min=0.1, max=5.0)  # Temperature scaling
            specialized_routing_weights = F.softmax(router_logits, dim=-1)  # [batch_size, num_specialized_experts]
            
            # Adjust specialized weights to use remaining weight budget
            remaining_weight = 1.0 - self.universal_expert_weight
            specialized_routing_weights = specialized_routing_weights * remaining_weight
            
            # Store for logging
            routing_weights_for_logging = specialized_routing_weights.clone()
            
            # Pre-compute FiLM parameters once (not per expert)
            cov_scale = self.cov_scale_net(weighted_cov)  # [batch, num_out]
            cov_shift = self.cov_shift_net(weighted_cov)  # [batch, num_out]
            cov_scale = cov_scale.unsqueeze(1)  # [batch, 1, num_out]
            cov_shift = cov_shift.unsqueeze(1)  # [batch, 1, num_out]
            
            # Get outputs from specialized experts and combine
            for i, specialist_idx in enumerate(range(self.num_universal_experts, self.num_experts)):
                expert_out = self.experts[specialist_idx](x)
                # Apply time conditioning
                expert_out = gamma.view(batch_size, -1, self.num_out) * expert_out
                
                # Apply pre-computed FiLM conditioning
                expert_out = cov_scale * expert_out + cov_shift
                
                # Apply routing weights: [batch_size, 1, 1] * [batch_size, seq_len, num_out]
                weighted_output = specialized_routing_weights[:, i:i+1].unsqueeze(-1) * expert_out
                mixed_output += weighted_output
            
            # Simple load balancing loss only
            load_balancing_loss = self._compute_load_balancing_loss(specialized_routing_weights.transpose(0, 1))
            
            # Remove expensive diversity loss computation
            # diversity_loss = self._compute_diversity_loss(specialized_routing_weights, weighted_cov)
            # load_balancing_loss += self.diversity_weight * diversity_loss
        elif self.num_specialized_experts > 0:
            # No covariate conditioning available, use uniform weights for specialized experts
            remaining_weight = 1.0 - self.universal_expert_weight
            uniform_weight = remaining_weight / self.num_specialized_experts
            
            for specialist_idx in range(self.num_universal_experts, self.num_experts):
                expert_out = self.experts[specialist_idx](x)
                # Apply time conditioning
                expert_out = gamma.view(batch_size, -1, self.num_out) * expert_out
                mixed_output += uniform_weight * expert_out
        
        # Apply strong covariate conditioning with FiLM after mixing experts
        if self.use_cov_conditioning and cov_embedding is not None and self.num_specialized_experts == 0:
            # Only apply if no specialized experts (to avoid double FiLM conditioning)
            cov_mult = torch.sigmoid(self.cov_scale_net(cov_embedding))  # [batch, num_out], bounded [0,1]
            cov_add = self.cov_shift_net(cov_embedding)  # [batch, num_out]
            
            cov_mult = cov_mult.unsqueeze(1)  # [batch, 1, num_out] for broadcasting
            cov_add = cov_add.unsqueeze(1)   # [batch, 1, num_out] for broadcasting
            
            out = cov_add + cov_mult * mixed_output  # Combined additive and multiplicative conditioning
        else:
            out = mixed_output
        
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
        
        return out, load_balancing_loss, routing_weights_for_logging
    
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

    def _compute_diversity_loss(self, routing_weights, weighted_cov):
        """
        Compute diversity loss to encourage different experts for different covariate patterns
        
        Args:
            routing_weights: [batch_size, num_specialized_experts]
            weighted_cov: [batch_size, cov_dim]
        """
        batch_size = weighted_cov.shape[0] 
        
        # Compute pairwise distances between weighted covariate vectors
        distances = torch.cdist(weighted_cov, weighted_cov)  # [batch_size, batch_size]
        
        # Compute diversity loss
        # Create upper triangular mask to avoid double counting
        mask = torch.triu(torch.ones(batch_size, batch_size, device=weighted_cov.device), diagonal=1)
        
        # Vectorized computation of diversity loss
        # routing_weights: [batch_size, num_specialized_experts]
        # We need to compute pairwise products of routing weights
        routing_outer = routing_weights.unsqueeze(1) * routing_weights.unsqueeze(0)  # [batch_size, batch_size, num_specialized_experts]
        
        # Apply mask and compute diversity loss
        masked_distances = distances * mask  # [batch_size, batch_size]
        weighted_distances = masked_distances.unsqueeze(-1) * routing_outer  # [batch_size, batch_size, num_specialized_experts]
        
        diversity_loss = weighted_distances.sum() / (batch_size * (batch_size - 1) / 2)
        
        return diversity_loss


class ConditionalLinearExpert(nn.Module):
    """
    Individual expert network for ConditionalLinearMoE with increased capacity
    """
    def __init__(self, num_in, num_out, expert_id=0, hidden_dim=None):
        super(ConditionalLinearExpert, self).__init__()
        self.expert_id = expert_id
        
        # Simplified single-layer expert for speed
        self.expert = nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.LayerNorm(num_out),
            nn.GELU()
        )
        
        # Simple residual connection
        self.use_residual = (num_in == num_out)
        if not self.use_residual:
            self.residual_proj = nn.Linear(num_in, num_out)
    
    def forward(self, x):
        out = self.expert(x)
        
        # Add residual connection
        if self.use_residual:
            out = out + x
        else:
            out = out + self.residual_proj(x)
            
        return out 