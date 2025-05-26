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

        # Create ConditionalLinear layers with covariate conditioning
        cov_dim = self.cov_dim if self.use_cov_conditioning else None
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
        return velocity_pred


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