import math
import torch
import numpy as np
from torchdiffeq import odeint


def extract(input, t, x):
    """Extract values from input tensor at timesteps t for batch x"""
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def sample_time_uniform(batch_size, num_timesteps, device):
    """Sample uniform timesteps for flow matching training"""
    return torch.randint(0, num_timesteps, (batch_size,), device=device)


def interpolate_data(y_0, y_1, t, interpolation_type="linear"):
    """
    Interpolate between y_0 and y_1 at time t
    
    Args:
        y_0: Starting data (clean target)
        y_1: Ending data (noise or prior)
        t: Interpolation time [0, 1]
        interpolation_type: Type of interpolation ("linear", "geodesic")
    
    Returns:
        y_t: Interpolated data
    """
    if interpolation_type == "linear":
        # Linear interpolation: y_t = (1-t) * y_0 + t * y_1
        t_expanded = t.view(-1, *([1] * (y_0.dim() - 1)))
        return (1 - t_expanded) * y_0 + t_expanded * y_1
    else:
        raise NotImplementedError(f"Interpolation type {interpolation_type} not implemented")


def compute_velocity_target(y_0, y_1, interpolation_type="linear"):
    """
    Compute the target velocity field for flow matching
    
    Args:
        y_0: Starting data (clean target)
        y_1: Ending data (noise or prior)
        interpolation_type: Type of interpolation
    
    Returns:
        velocity: Target velocity field
    """
    if interpolation_type == "linear":
        # For linear interpolation: v_t = y_1 - y_0
        return y_1 - y_0
    else:
        raise NotImplementedError(f"Interpolation type {interpolation_type} not implemented")


class ODEFunc(torch.nn.Module):
    """ODE function for neural ODE solver"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.x = None
        self.x_mark = None
        self.y_0_hat = None
        self.cov_embedding = None
    
    def set_conditioning(self, x, x_mark, y_0_hat, cov_embedding=None):
        """Set conditioning variables for the ODE"""
        self.x = x
        self.x_mark = x_mark
        self.y_0_hat = y_0_hat
        self.cov_embedding = cov_embedding
    
    def forward(self, t, y):
        """
        Forward pass for ODE solver
        
        Args:
            t: Current time (scalar or tensor)
            y: Current state
        
        Returns:
            dy/dt: Time derivative
        """
        # Convert time to tensor if needed
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=y.device, dtype=y.dtype)
        
        # Expand time to match batch size
        if t.dim() == 0:
            t = t.expand(y.shape[0])
        
        # Convert continuous time [0,1] to discrete timesteps for model
        # Scale from [0,1] to [0, num_timesteps-1] 
        t_discrete = (t * (self.model.num_timesteps - 1)).long().clamp(0, self.model.num_timesteps - 1)
        
        # Get velocity prediction from model (handle both old and new return formats)
        model_output = self.model(self.x, self.x_mark, 0, y, self.y_0_hat, t_discrete, cov_embedding=self.cov_embedding)
        
        if isinstance(model_output, tuple):
            # New format with MoE support - we only need the velocity for ODE integration
            velocity = model_output[0]
        else:
            # Old format (backward compatibility)
            velocity = model_output
        
        return velocity


def sample_flow_matching(model, x, x_mark, y_0_hat, y_T_mean, num_timesteps, 
                        solver='dopri5', rtol=1e-5, atol=1e-5, cov_embedding=None):
    """
    Sample from flow matching model using ODE solver
    
    Args:
        model: Flow matching model
        x: Input conditions
        x_mark: Input time marks
        y_0_hat: Conditional prediction
        y_T_mean: Prior mean at T=1
        num_timesteps: Number of timesteps for discretization
        solver: ODE solver method
        rtol: Relative tolerance for ODE solver
        atol: Absolute tolerance for ODE solver
        cov_embedding: Covariate embedding for conditioning
    
    Returns:
        y_0: Generated sample at t=0
    """
    device = next(model.parameters()).device
    
    # Start from noise at t=1
    z = torch.randn_like(y_T_mean).to(device)
    y_1 = z + y_T_mean
    
    # Set up ODE function
    ode_func = ODEFunc(model)
    ode_func.set_conditioning(x, x_mark, y_0_hat, cov_embedding)
    
    # Time points: from t=1 to t=0 (reverse time)
    t_span = torch.linspace(1.0, 0.0, num_timesteps, device=device)
    
    # Solve ODE
    with torch.no_grad():
        solution = odeint(ode_func, y_1, t_span, method=solver, rtol=rtol, atol=atol)
    
    # Return final state (at t=0)
    y_0 = solution[-1]
    
    return [solution[i] for i in range(len(solution))]  # Return full trajectory


def flow_matching_loss(model, x, x_mark, y_0, y_0_hat, t, cov_embedding=None, 
                      noise_type="gaussian", interpolation_type="linear"):
    """
    Compute flow matching loss
    
    Args:
        model: Flow matching model
        x: Input conditions
        x_mark: Input time marks
        y_0: Target data (clean)
        y_0_hat: Conditional prediction
        t: Sampled timesteps [0, 1]
        cov_embedding: Covariate embedding for conditioning
        noise_type: Type of noise for y_1
        interpolation_type: Type of interpolation
    
    Returns:
        loss: Flow matching loss
    """
    # Sample noise for y_1
    if noise_type == "gaussian":
        y_1 = torch.randn_like(y_0) + y_0_hat  # Noise around conditional prediction
    else:
        raise NotImplementedError(f"Noise type {noise_type} not implemented")
    
    # Interpolate to get y_t
    y_t = interpolate_data(y_0, y_1, t, interpolation_type)
    
    # Compute target velocity
    target_velocity = compute_velocity_target(y_0, y_1, interpolation_type)
    
    # Convert continuous time to discrete timesteps for model
    t_discrete = (t * (model.num_timesteps - 1)).long().clamp(0, model.num_timesteps - 1)
    
    # Predict velocity (handle both old and new return formats)
    model_output = model(x, x_mark, 0, y_t, y_0_hat, t_discrete, cov_embedding=cov_embedding)
    
    if isinstance(model_output, tuple):
        # New format with MoE support
        predicted_velocity = model_output[0]
    else:
        # Old format (backward compatibility)
        predicted_velocity = model_output
    
    # Compute loss (MSE between predicted and target velocity)
    loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
    
    return loss


def flow_matching_loss_with_moe(model, x, x_mark, y_0, y_0_hat, t, cov_embedding=None, 
                                noise_type="gaussian", interpolation_type="linear", 
                                moe_loss_weight=0.01):
    """
    Compute flow matching loss with MoE load balancing loss
    
    Args:
        model: Flow matching model
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
    # Sample noise for y_1
    if noise_type == "gaussian":
        y_1 = torch.randn_like(y_0) + y_0_hat  # Noise around conditional prediction
    else:
        raise NotImplementedError(f"Noise type {noise_type} not implemented")
    
    # Interpolate to get y_t
    y_t = interpolate_data(y_0, y_1, t, interpolation_type)
    
    # Compute target velocity
    target_velocity = compute_velocity_target(y_0, y_1, interpolation_type)
    
    # Convert continuous time to discrete timesteps for model
    t_discrete = (t * (model.num_timesteps - 1)).long().clamp(0, model.num_timesteps - 1)
    
    # Predict velocity and get MoE losses
    predicted_velocity, load_balancing_loss, routing_info = model(x, x_mark, 0, y_t, y_0_hat, t_discrete, cov_embedding=cov_embedding)
    
    # Compute main flow matching loss (MSE between predicted and target velocity)
    flow_loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
    
    # Combine losses
    total_loss = flow_loss + moe_loss_weight * load_balancing_loss
    
    # Return loss components for logging
    loss_dict = {
        'flow_loss': flow_loss,
        'load_balancing_loss': load_balancing_loss,
        'total_loss': total_loss,
        'routing_info': routing_info
    }
    
    return total_loss, loss_dict


# Evaluation with KLD (same as diffusion)
def kld(y1, y2, grid=(-20, 20), num_grid=400):
    y1, y2 = y1.numpy().flatten(), y2.numpy().flatten()
    p_y1, _ = np.histogram(y1, bins=num_grid, range=[grid[0], grid[1]], density=True)
    p_y1 += 1e-7
    p_y2, _ = np.histogram(y2, bins=num_grid, range=[grid[0], grid[1]], density=True)
    p_y2 += 1e-7
    return (p_y1 * np.log(p_y1 / p_y2)).sum() 