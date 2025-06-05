# NS-Transformer with Time-Series Covariate Processing

## Overview

This document describes the modifications made to the NS-Transformer model to handle glucose prediction with separate processing of time-series covariates. The key changes enable the model to:

1. **Separate glucose from other time-series channels** - The last channel is treated as glucose, while other channels serve as time-series covariates
2. **Process time-series covariates with 1D CNN** - Extract time-invariant embeddings from time-series covariates
3. **Fuse multiple covariate types** - Combine tabular and time-series covariate embeddings
4. **Use glucose-only data for VAE conditioning** - Only glucose is used for the main VAE process

## Key Components

### 1. TimeSeriesCovariateEncoder

A new CNN-based encoder that processes time-series covariates to extract time-invariant embeddings:

```python
class TimeSeriesCovariateEncoder(nn.Module):
    def __init__(self, in_channels, d_model, seq_len, dropout=0.1):
        # Multi-scale 1D CNN layers for different temporal patterns
        # Global pooling for time-invariant representation
        # Projection layer for final embedding
```

**Features:**
- Multi-scale 1D convolutions (kernel sizes: 3, 5, 7) to capture different temporal patterns
- Batch normalization and residual connections
- Global average and max pooling for time-invariant features
- Final projection to model dimension

### 2. Modified Model Architecture

The main `Model` class has been updated with:

**New Attributes:**
- `glucose_channels = 1` - Number of glucose channels (always 1)
- `ts_covariate_channels` - Number of time-series covariate channels (enc_in - 1)
- `ts_cov_encoder` - TimeSeriesCovariateEncoder instance
- `covariate_fusion` - Neural network to fuse tabular and time-series embeddings

**Updated Components:**
- Embedding layers now process only glucose channels
- Projector layers (tau_learner, delta_learner) work with glucose-only data
- Covariate fusion layer combines different embedding types

## Data Flow

### Input Processing

```python
# Input: x_enc [batch_size, seq_len, enc_in]
# Where enc_in = ts_covariate_channels + glucose_channels

# Separate channels
x_glucose = x_enc[:, :, -1:]  # Last channel - glucose
x_ts_covariates = x_enc[:, :, :-1]  # Other channels - time-series covariates

# Process separately
glucose_embeddings = encoder(x_glucose)  # Main VAE path
ts_cov_embeddings = ts_cov_encoder(x_ts_covariates)  # Time-invariant
tabular_embeddings = tabular_encoder(tabular_covariates)  # Static features
```

### Covariate Fusion

```python
# Combine embeddings
if ts_covariate_channels > 0:
    combined_embedding = covariate_fusion(
        torch.cat([tabular_embeddings, ts_cov_embeddings], dim=1)
    )
else:
    combined_embedding = tabular_embeddings
```

### VAE Processing

The VAE components (mean/logvar networks, reparameterization) now work exclusively with glucose data:

- **Normalization**: Applied only to glucose channel
- **Tau/Delta learning**: Uses glucose statistics only
- **Latent space**: Represents glucose patterns specifically

## Usage Example

```python
from ns_models.ns_Transformer import Model, configure_moe

# Configuration
configs = create_config()
configs.enc_in = 8  # 7 time-series covariates + 1 glucose
configs.dec_in = 1  # glucose only
configs.c_out = 1   # glucose prediction

# Create model
model = Model(configs)

# Input data
x_enc = torch.randn(batch_size, seq_len, 8)  # 7 covariates + 1 glucose
# ... other inputs

# Forward pass
predictions, dec_out, kl_loss, latent = model(
    x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=tabular_covariates
)
```

## Benefits

1. **Specialized Processing**: Glucose and covariates are processed with appropriate architectures
2. **Time-Invariant Features**: CNN extracts robust features from time-series covariates
3. **Rich Conditioning**: VAE benefits from both tabular and time-series covariate information
4. **Scalability**: Can handle varying numbers of time-series covariates
5. **Interpretability**: Clear separation of glucose modeling from covariate processing

## Configuration Parameters

```python
# Model dimensions
configs.enc_in = total_channels  # ts_covariates + glucose
configs.dec_in = 1              # glucose only
configs.c_out = 1               # glucose prediction

# Time-series covariate encoder
configs.dropout = 0.1           # CNN dropout rate

# Covariate fusion
configs.d_model = 512           # Embedding dimension

# Tabular covariates
configs.col_stats = {...}       # Column statistics
configs.col_names_dict = {...}  # Column type mapping
```

## Compatibility

- Fully compatible with existing MoE functionality
- Works with all attention mechanisms (DSAttention, etc.)
- Maintains compatibility with wandb logging for routing statistics
- Supports both sparse and dense gating in MoE layers

## Testing

Run the provided example:

```bash
cd models/model9_NS_transformer
python usage_example.py
```

This will demonstrate the model with synthetic data showing the separation of glucose and time-series covariates. 