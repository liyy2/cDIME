# Time Series Flow Matching Models

This directory contains the implementation of Flow Matching for time series forecasting, which serves as an alternative to the diffusion-based approach. Flow matching uses ODEs instead of SDEs, providing more stable training and faster sampling.

## Overview

Flow matching is a generative modeling technique that learns to match probability flows using continuous normalizing flows. Unlike diffusion models that add noise through a stochastic process, flow matching learns a deterministic velocity field that transforms a simple distribution (e.g., Gaussian) to the target data distribution.

## Key Differences from Diffusion Models

| Aspect | Diffusion Models | Flow Matching |
|--------|------------------|---------------|
| **Mathematical Foundation** | Stochastic Differential Equations (SDEs) | Ordinary Differential Equations (ODEs) |
| **Training Objective** | Noise prediction (ε-prediction) | Velocity field prediction |
| **Time Sampling** | Discrete noise schedule (β_t) | Continuous time [0,1] |
| **Sampling Process** | Iterative denoising with many steps | ODE solving with adaptive steps |
| **Training Stability** | Requires careful noise schedule tuning | More stable, no schedule tuning |
| **Sampling Speed** | Slower (many diffusion steps) | Faster (adaptive ODE solver) |
| **Theoretical Guarantees** | Well-established but complex | Simpler theory, better guarantees |

## Files Structure

```
flow_matching_models/
├── __init__.py                 # Package initialization
├── flow_matching_utils.py      # Core flow matching utilities
├── model.py                   # Neural network architectures
├── flowMTS.py                 # Main flow matching model
└── README.md                  # This file
```

## Core Components

### 1. Flow Matching Utilities (`flow_matching_utils.py`)

- **`interpolate_data()`**: Linear interpolation between clean data and noise
- **`compute_velocity_target()`**: Computes target velocity field
- **`ODEFunc`**: ODE function wrapper for neural ODE solver
- **`sample_flow_matching()`**: ODE-based sampling using torchdiffeq
- **`flow_matching_loss()`**: Flow matching training loss

### 2. Model Architecture (`model.py`)

- **`ConditionalVelocityModel`**: Predicts velocity field instead of noise
- Same architecture as diffusion model but outputs velocity
- Supports covariate conditioning
- Time-dependent conditioning through embeddings

### 3. Main Model (`flowMTS.py`)

- **`Model`**: Main flow matching time series model
- Reuses diffusion config structure for compatibility
- Simplified noise scheduling (just linear time steps)
- Built-in sampling and loss computation methods

## Usage

### Basic Usage

```python
from models.time_series_flow_matching_model import TimeSeriesFlowMatchingModel

# Initialize model (same interface as diffusion model)
model = TimeSeriesFlowMatchingModel(args, train_loader, val_loader, test_loader)

# Train using PyTorch Lightning
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model)
```

### Key Parameters

- **`timesteps`**: Number of time discretization steps (can be fewer than diffusion)
- **`solver`**: ODE solver method ('dopri5', 'rk4', 'euler', etc.)
- **`rtol/atol`**: Relative/absolute tolerance for ODE solver
- **`interpolation_type`**: Interpolation method ('linear' supported)

### Example Training

```bash
python example_flow_matching.py \
    --model ns_Transformer \
    --timesteps 50 \
    --learning_rate 0.0001 \
    --train_epochs 100
```

## Mathematical Background

### Flow Matching Objective

The flow matching objective learns a velocity field `v_θ(x_t, t)` such that:

```
dx_t/dt = v_θ(x_t, t)
```

Where the training loss is:
```
L = E[||v_θ(x_t, t) - u_t||²]
```

And `u_t` is the target velocity:
```
u_t = x_1 - x_0  (for linear interpolation)
```

### Interpolation Process

Given clean data `x_0` and noise `x_1`, we interpolate:
```
x_t = (1-t) * x_0 + t * x_1
```

The model learns to predict the velocity that would generate this trajectory.

### Sampling Process

To generate new samples:
1. Start from noise: `x_1 ~ N(μ, σ²)`
2. Solve ODE backwards: `dx/dt = -v_θ(x_t, 1-t)`
3. Get clean sample: `x_0`

## Advantages

### 1. Training Stability
- No need to tune complex noise schedules
- More stable gradients during training
- Less sensitive to hyperparameters

### 2. Sampling Efficiency
- Adaptive ODE solvers can use fewer function evaluations
- No need for many fixed steps like in diffusion
- Can trade off speed vs accuracy with solver tolerance

### 3. Theoretical Guarantees
- Based on optimal transport theory
- Cleaner mathematical formulation
- Better understanding of the learning dynamics

### 4. Implementation Simplicity
- Simpler loss function (just MSE on velocities)
- No complex noise scheduling
- Direct optimization of the flow

## Configuration

The flow matching model reuses the diffusion configuration structure for compatibility:

```yaml
model:
  cat_x: true
  cat_y_pred: true
  use_cov_conditioning: true

diffusion:  # Reused for flow matching
  timesteps: 50
  
testing:
  n_z_samples: 100
  n_z_samples_depart: 5
```

## Dependencies

- `torch`: PyTorch framework
- `torchdiffeq`: Neural ODE solver
- `pytorch-lightning`: Training framework
- `numpy`: Numerical computations

Install with: `pip install torchdiffeq`

## Performance Considerations

### Training Time
- Similar to diffusion models
- May converge faster due to simpler objective

### Sampling Time
- Generally faster than diffusion models
- Can be controlled via ODE solver settings
- Use `rtol=1e-5, atol=1e-5` for good speed/quality trade-off

### Memory Usage
- Similar to diffusion models
- ODE solver may require additional memory for intermediate states

## Comparison with Diffusion Results

When switching from diffusion to flow matching, you can expect:
- **Similar or better sample quality**
- **Faster sampling (2-5x speedup)**
- **More stable training**
- **Easier hyperparameter tuning**

## Troubleshooting

### Common Issues

1. **ODE solver convergence**: Reduce `rtol/atol` or use a more stable solver
2. **Memory issues**: Reduce batch size or use checkpointing
3. **Training instability**: Check learning rate and loss weights

### Tips

- Start with `dopri5` solver for best accuracy
- Use `euler` solver for fastest sampling
- Monitor velocity magnitude during training
- Ensure proper time normalization [0,1]

## References

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Conditional Flow Matching](https://arxiv.org/abs/2302.00482)
- [Neural ODEs](https://arxiv.org/abs/1806.07366) 