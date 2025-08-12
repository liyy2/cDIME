import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from scipy import stats
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import warnings
    warnings.filterwarnings('ignore')

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("viridis")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    return device, go, make_subplots, np, pd, plt, torch


@app.cell
def _():
    # Import project modules
    import sys
    sys.path.append('/home/yl2428/Time-LLM')

    from data_provider_pretrain.data_factory import data_provider
    from models.time_series_flow_matching_model import TimeSeriesFlowMatchingModel
    from models.model9_NS_transformer.ns_models.ns_Transformer import Model as NSTransformer
    import argparse
    from tqdm import tqdm
    import os
    import glob
    import re

    # Import sampling utilities
    from models.model9_NS_transformer.flow_matching_models.flow_matching_utils import sample_flow_matching
    from torch_frame import stype
    from copy import deepcopy

    return (
        TimeSeriesFlowMatchingModel,
        data_provider,
        deepcopy,
        glob,
        os,
        re,
        sample_flow_matching,
        stype,
    )


@app.cell
def _():
    # Define channel names for the 9 glucose time-series features
    CHANNEL_NAMES = [
        'bolus',           # 0: Insulin bolus
        'basal',           # 1: Basal insulin rate
        'heart_rate',      # 2: Heart rate
        'steps',           # 3: Step count
        'sleep',           # 4: Sleep quality/duration
        'fat_g',           # 5: Fat intake (grams)
        'protein_g',       # 6: Protein intake (grams)
        'carbohydrates_g', # 7: Carbohydrate intake (grams)
        'glucose'          # 8: Glucose level (target)
    ]

    # Channel categories for grouping
    CHANNEL_CATEGORIES = {
        'insulin': ['bolus', 'basal'],
        'activity': ['heart_rate', 'steps', 'sleep'],
        'nutrition': ['fat_g', 'protein_g', 'carbohydrates_g'],
        'target': ['glucose']
    }

    # Color mapping for channels
    CHANNEL_COLORS = {
        'bolus': '#FF6B6B',
        'basal': '#FF9999',
        'heart_rate': '#4ECDC4',
        'steps': '#45B7D1',
        'sleep': '#96CEB4',
        'fat_g': '#FFEAA7',
        'protein_g': '#DDA77A',
        'carbohydrates_g': '#FFB6C1',
        'glucose': '#9B59B6'
    }

    print(f"Defined {len(CHANNEL_NAMES)} channels:")
    for i, name in enumerate(CHANNEL_NAMES):
        category = [cat for cat, channels in CHANNEL_CATEGORIES.items() if name in channels][0]
        print(f"  {i}: {name:15} ({category})")

    return


@app.cell
def _(glob, os, re, torch):
    # Configuration from velocity_importance_advanced.ipynb
    class DotDict(dict):
        """A dictionary that supports both dot notation and dictionary access."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

        def __getattr__(self, attr):
            return self.get(attr)

        def __setattr__(self, key, value):
            self.__dict__[key] = value

        def __delattr__(self, item):
            self.__dict__.pop(item, None)

    # Flow matching configuration (exact from feature_importance_analysis.ipynb)
    flow_matching_config = DotDict({
        "num_nodes": 1,
        "task_name": "long_term_forecast",
        "is_training": 1,
        "model_id": "ETTh1_ETTh2_512_192",
        "model": "ns_Transformer",
        "precision": "32",
        "generative_model": "flow_matching",
        "data_pretrain": "Glucose",
        "root_path": "/home/yl2428/Time-LLM/dataset/glucose",
        "data_path": "output_Junt_16_3.csv",
        "data_path_pretrain": "output_Junt_16_3.csv",
        "features": "MS",
        "target": "OT",
        "freq": "t",
        "checkpoints": "/home/yl2428/checkpoints",
        "log_dir": "/home/yl2428/logs",
        "seq_len": 72,
        "label_len": 32,
        "pred_len": 48,
        "seasonal_patterns": "Monthly",
        "stride": 1,
        "enc_in": 9,
        "dec_in": 9,
        "c_out": 9,
        "d_model": 32,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 256,
        "moving_avg": 25,
        "factor": 3,
        "dropout": 0.1,
        "embed": "timeF",
        "activation": "gelu",
        "output_attention": False,
        "patch_len": 16,
        "prompt_domain": 0,
        "llm_model": "LLAMA",
        "llm_dim": 4096,
        "vae_hidden_dim": 16,
        "num_workers": 10,
        "itr": 1,
        "train_epochs": 100,
        "align_epochs": 10,
        "ema_decay": 0.995,
        "batch_size": 512,
        "eval_batch_size": 512,
        "patience": 40,
        "learning_rate": 0.0001,
        "des": "Exp",
        "loss": "MSE",
        "lradj": "COS",
        "pct_start": 0.2,
        "use_amp": False,
        "llm_layers": 32,
        "percent": 100,
        "num_individuals": -1,
        "enable_covariates": 1,
        "cov_type": "tensor",
        "gradient_accumulation_steps": 1,
        "use_deep_speed": 1,
        "wandb": 1,
        "wandb_group": None,
        "use_moe": 1,
        "num_experts": 8,
        "latent_len": 36,
        "top_k_experts": 4,
        "moe_layer_indices": [0, 1],
        "moe_loss_weight": 0.01,
        "log_routing_stats": 1,
        "num_universal_experts": 1,
        "universal_expert_weight": 0.3,
        "head_dropout": 0.1,
        "channel_independence": 0,
        "decomp_method": "moving_avg",
        "use_norm": 1,
        "down_sampling_layers": 2,
        "down_sampling_window": 1,
        "down_sampling_method": "avg",
        "use_future_temporal_feature": 0,
        "k_z": 1e-2,
        "k_cond": 0.001,
        "d_z": 8,
        "p_hidden_dims": [64, 64],
        "p_hidden_layers": 2,
        "diffusion_config_dir": "/home/yl2428/Time-LLM/models/model9_NS_transformer/configs/toy_8gauss.yml",
        "cond_pred_model_pertrain_dir": None,
        "CART_input_x_embed_dim": 32,
        "mse_timestep": 0,
        "MLP_diffusion_net": False,
        "timesteps": 50,
        "ode_solver": "dopri5",
        "ode_rtol": 1e-5,
        "ode_atol": 1e-5,
        "interpolation_type": "linear",
        "expert_layers": 2,
        "loader": "modal",
        "model_comment": "none",
        "enable_context_aware": 1,
        "glucose_dropout_rate": 0.4,
        "use_contrastive_learning": 1,
        "contrastive_loss_weight": 0.1,
        "contrastive_temperature": 0.1,
        "use_momentum_encoder": 1,
        "momentum_factor": 0.999,
        "n_flow_stages": 5,  # For velocity analysis
        "col_stats": {'SEX': {'COUNT': (['F', 'M'], [367, 135])}, 'RACE': {'COUNT': (['WHITE', 'NOT REPORTED', 'ASIAN', 'BLACK/AFRICAN AMERICAN', 'MULTIPLE', 'UNKNOWN', 'AMERICAN INDIAN/ALASKAN NATIVE'], [459, 11, 10, 10, 8, 2, 2])}, 'ETHNIC': {'COUNT': (['Not Hispanic or Latino', 'Hispanic or Latino', 'Do not wish to answer', "Don't know"], [472, 15, 13, 2])}, 'ARMCD': {'COUNT': (['RESISTANCE', 'INTERVAL', 'AEROBIC'], [172, 167, 163])}, 'insulin modality': {'COUNT': (['CLOSED LOOP INSULIN PUMP', 'INSULIN PUMP', 'MULTIPLE DAILY INJECTIONS'], [225, 189, 88])}, 'AGE': {'MEAN': 36.655378486055774, 'STD': 13.941209833786187, 'QUANTILES': [18.0, 25.0, 33.0, 45.75, 70.0]}, 'WEIGHT': {'MEAN': 161.39940239043824, 'STD': 30.624877585598654, 'QUANTILES': [103.0, 140.0, 155.0, 179.0, 280.0]}, 'HEIGHT': {'MEAN': 66.72509960159363, 'STD': 3.505847063905933, 'QUANTILES': [58.0, 64.0, 66.0, 69.0, 77.0]}, 'HbA1c': {'MEAN': 6.642828685258964, 'STD': 0.7633658734231158, 'QUANTILES': [4.8, 6.1, 6.6, 7.1, 10.0]}, 'DIABETES_ONSET': {'MEAN': 18.72725737051793, 'STD': 11.889102915798386, 'QUANTILES': [0.0833, 11.0, 16.0, 24.0, 66.0]}},
        "col_names_dict": {'categorical': ['ARMCD', 'ETHNIC', 'RACE', 'SEX', 'insulin modality'], 'numerical': ['AGE', 'DIABETES_ONSET', 'HEIGHT', 'HbA1c', 'WEIGHT']}
    })

    # Checkpoint loading functions
    def find_best_checkpoint(base_path="/home/yl2428/logs/ns_Transformer/flow_matching/comfy-dust-243", metric="val_loss"):
        """Find the best checkpoint based on validation loss."""
        print(f"Searching for checkpoints in: {base_path}")

        checkpoint_pattern = os.path.join(base_path, "checkpoints/epoch=*-step=*-val_loss=*.ckpt/checkpoint")
        print(checkpoint_pattern)
        checkpoint_dirs = glob.glob(checkpoint_pattern)

        if not checkpoint_dirs:
            print("No checkpoints found!")
            return None, None, None

        best_checkpoint = None
        best_metric = float('inf')
        best_run = None

        print(f"Found {len(checkpoint_dirs)} checkpoints:")

        for checkpoint_dir in checkpoint_dirs:
            pattern = r'epoch=(\d+)-step=(\d+)-val_loss=([\d.]+)\.ckpt'
            match = re.search(pattern, checkpoint_dir)

            if match:
                epoch, step, val_loss = match.groups()
                val_loss = float(val_loss)
                run_name = checkpoint_dir.split('/')[-4]

                print(f"  - {run_name}: epoch={epoch}, step={step}, val_loss={val_loss:.4f}")

                if val_loss < best_metric:
                    best_metric = val_loss
                    best_checkpoint = checkpoint_dir
                    best_run = run_name

        if best_checkpoint:
            print(f"\nBest checkpoint: {best_run}")
            print(f"  - Path: {best_checkpoint}")
            print(f"  - Val Loss: {best_metric:.4f}")

        return best_checkpoint, best_metric, best_run

    def load_deepspeed_checkpoint(model, checkpoint_path, device):
        """Load DeepSpeed checkpoint into the model."""
        print(f"Loading DeepSpeed checkpoint from: {checkpoint_path}")

        model_states_path = os.path.join(checkpoint_path, "mp_rank_00_model_states.pt")

        if not os.path.exists(model_states_path):
            raise FileNotFoundError(f"Model states file not found: {model_states_path}")

        print(f"Using device: {device}")

        checkpoint = torch.load(model_states_path, map_location=device)

        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key
            if key.startswith('_forward_module.'):
                clean_key = key.replace('_forward_module.', '')
            elif key.startswith('module.'):
                clean_key = key.replace('module.', '')

            if isinstance(value, torch.Tensor):
                value = value.to(device)

            cleaned_state_dict[clean_key] = value

        try:
            model = model.to(device)
            missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)

            if missing_keys:
                print(f"Missing keys: {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")

            print("✓ Model weights loaded successfully!")

        except Exception as e:
            print(f"Warning: Some keys couldn't be loaded: {e}")
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in cleaned_state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)

            model = model.to(device)
            model.load_state_dict(model_dict)
            print(f"✓ Loaded {len(pretrained_dict)}/{len(cleaned_state_dict)} parameters")

        model = model.to(device)
        print(f"✓ All model components moved to {device}")

        return model

    # Use the config as args
    args = flow_matching_config
    print(f"Configuration loaded: {args.model} with d_model={args.d_model}, batch_size={args.batch_size}")
    print(f"Diffusion config path: {args.diffusion_config_dir}")

    return (
        find_best_checkpoint,
        flow_matching_config,
        load_deepspeed_checkpoint,
    )


@app.cell
def _(
    TimeSeriesFlowMatchingModel,
    data_provider,
    device,
    find_best_checkpoint,
    flow_matching_config,
    load_deepspeed_checkpoint,
):
    # Load model and data
    print("Loading flow matching model and data...")

    # Load data
    flow_args = flow_matching_config
    train_data_fm, train_loader_fm, flow_args = data_provider(
        flow_args, flow_args.data_pretrain, flow_args.data_path_pretrain, True, 'train'
    )
    vali_data_fm, vali_loader_fm, flow_args = data_provider(
        flow_args, flow_args.data_pretrain, flow_args.data_path_pretrain, True, 'val'
    )
    test_data_fm, test_loader_fm, flow_args = data_provider(
        flow_args, flow_args.data_pretrain, flow_args.data_path_pretrain, False, 'test'
    )

    # Initialize model
    flow_matching_model = TimeSeriesFlowMatchingModel(flow_args, train_loader_fm, vali_loader_fm, test_loader_fm)

    # Find and load best checkpoint
    checkpoint_path, best_metric, run_name = find_best_checkpoint()

    if checkpoint_path:
        flow_matching_model = load_deepspeed_checkpoint(flow_matching_model, checkpoint_path, device)
        flow_matching_model.eval()
        print(f"✓ Model loaded from {run_name} with val_loss: {best_metric:.4f}")
    else:
        print("No checkpoint found - using untrained model")
        flow_matching_model.eval()

    # Set model reference
    model = flow_matching_model
    test_loader = test_loader_fm

    print(f"Model on device: {next(flow_matching_model.parameters()).device}")
    print(f"Model type: {flow_args.model}")
    print(f"Covariates enabled: {flow_args.enable_covariates}")
    print(f"Batch size: {flow_args.batch_size}")
    print(f"Number of experts (from config): {flow_args.num_experts}")
    print("✓ Data and model loaded successfully!")

    return model, test_loader


@app.cell
def _(deepcopy, generate_predictions, model, np, stype, test_loader):
    # Perturbation functions and execution
    
    # Define perturbation functions
    def perturb_hba1c_covariates(batch_cov, individual_indices, percentage_increase):
        """
        Perturbs HbA1c for specified individuals in the batch_cov.
        HbA1c is at index 3 of the numerical features.
        Returns a new batch_cov object with perturbations; does not modify the input.
        """
        if stype.numerical not in batch_cov.feat_dict:
            print(f"Warning: stype.numerical not found in batch_cov.feat_dict")
            return batch_cov

        # Clone the numerical features tensor
        original_numerical_tensor = batch_cov.feat_dict[stype.numerical]
        perturbed_numerical_tensor = original_numerical_tensor.clone()

        for idx in individual_indices:
            if 0 <= idx < perturbed_numerical_tensor.shape[0]:
                # Modify HbA1c (index 3)
                current_hba1c_val = perturbed_numerical_tensor[idx, 3]
                perturbed_numerical_tensor[idx, 3] = current_hba1c_val * (1 + percentage_increase / 100.0)
                print(f"Individual {idx}: HbA1c changed from {current_hba1c_val.item():.2f} to {perturbed_numerical_tensor[idx, 3].item():.2f}")

        # Create new batch_cov with perturbed values
        new_batch_cov = deepcopy(batch_cov)
        new_batch_cov.feat_dict[stype.numerical] = perturbed_numerical_tensor

        return new_batch_cov

    def perturb_steps_timeseries(batch_x, individual_indices, percentage_increase):
        """
        Perturbs steps for specified individuals in batch_x.
        Steps are at channel 3 (0-indexed) in the time series.
        Returns a new batch_x tensor with perturbations.
        """
        # Constants for step calculation
        STEPS_SCALE = 20.84327263
        STEPS_OFFSET = 6.53019535
        STEPS_CHANNEL_IDX = 3  # Channel 3 in 9-channel system

        perturbed_batch_x = batch_x.clone()

        for idx in individual_indices:
            # Extract scaled steps series for this individual
            scaled_steps = perturbed_batch_x[idx, :, STEPS_CHANNEL_IDX]

            # Convert to true steps values
            true_steps = scaled_steps * STEPS_SCALE + STEPS_OFFSET

            # Apply percentage increase
            perturbed_true_steps = true_steps * (1 + percentage_increase / 100.0)

            # Convert back to scaled values
            perturbed_scaled_steps = (perturbed_true_steps - STEPS_OFFSET) / STEPS_SCALE

            # Update the tensor
            perturbed_batch_x[idx, :, STEPS_CHANNEL_IDX] = perturbed_scaled_steps

            print(f"Individual {idx}: Steps increased by {percentage_increase}% (mean: {true_steps.mean().item():.1f} -> {perturbed_true_steps.mean().item():.1f})")

        return perturbed_batch_x

    print("Perturbation functions defined:")
    print("  - perturb_hba1c_covariates: Modifies HbA1c in covariate tensor")
    print("  - perturb_steps_timeseries: Modifies steps in time series data")
    
    # Execute perturbation analysis
    print("\nExecuting perturbation analysis...")

    # Get a batch from test loader
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx == 0:  # Use first batch
            break

    # Unpack batch
    if len(batch) == 2:
        (batch_x, batch_y, batch_x_mark, batch_y_mark), batch_cov = batch
    else:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_cov = None

    # Select individuals to perturb (first 10 or random subset)
    batch_size_actual = batch_x.shape[0]
    num_individuals_to_perturb = min(10, batch_size_actual)
    individuals_to_perturb = np.random.choice(batch_size_actual, num_individuals_to_perturb, replace=False)

    print(f"Selected {num_individuals_to_perturb} individuals for perturbation: {individuals_to_perturb}")

    # Store original HbA1c values (using stype from earlier import)
    original_hba1c_values = []
    if batch_cov is not None:
        if stype.numerical in batch_cov.feat_dict:
            for idx in individuals_to_perturb:
                original_hba1c_values.append(batch_cov.feat_dict[stype.numerical][idx, 3].item())

    # Generate predictions BEFORE perturbation
    print("\nGenerating predictions BEFORE perturbation...")
    predictions_before = generate_predictions(
        model, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cov, n_samples=50
    )

    # Apply perturbations
    print("\nApplying perturbations...")
    percentage_increase_hba1c = 20  # 20% increase in HbA1c
    percentage_increase_steps = 30  # 30% increase in steps

    # Perturb HbA1c
    perturbed_batch_cov = perturb_hba1c_covariates(batch_cov, individuals_to_perturb, percentage_increase_hba1c)

    # Perturb steps
    perturbed_batch_x = perturb_steps_timeseries(batch_x, individuals_to_perturb, percentage_increase_steps)

    # Generate predictions AFTER perturbation
    print("\nGenerating predictions AFTER perturbation...")
    predictions_after = generate_predictions(
        model, perturbed_batch_x, batch_y, batch_x_mark, batch_y_mark, perturbed_batch_cov, n_samples=50
    )

    # Store results
    perturbation_results = {
        'individuals': individuals_to_perturb,
        'predictions_before': predictions_before,
        'predictions_after': predictions_after,
        'batch_x': batch_x,
        'batch_y': batch_y,
        'original_hba1c': original_hba1c_values,
        'percentage_increase_hba1c': percentage_increase_hba1c,
        'percentage_increase_steps': percentage_increase_steps
    }

    print(f"\n✓ Perturbation analysis complete!")
    print(f"  - Predictions shape: {predictions_before.shape}")
    print(f"  - HbA1c increased by {percentage_increase_hba1c}%")
    print(f"  - Steps increased by {percentage_increase_steps}%")

    return perturbation_results, perturb_hba1c_covariates, perturb_steps_timeseries


@app.cell
def _(model, sample_flow_matching, torch):
    # Sampling function for generating predictions

    def generate_predictions(model, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cov, 
                            n_samples=50, device='cuda'):
        """
        Generate predictions from the flow matching model.
        Returns sampled predictions.
        """
        model.eval()

        # Move data to device
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        if batch_cov is not None:
            batch_cov = batch_cov.to(device)

        with torch.no_grad():
            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -model.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :model.args.label_len, :], dec_inp], dim=1).float().to(device)

            # Get condition model output
            y_0_hat_batch, KL_loss, z_sample, cov_embedding = model.condition_model_forward(
                batch_x, batch_x_mark, dec_inp, batch_y_mark, covariates=batch_cov
            )

            # Prepare for flow matching
            f_dim = -1 if model.args.features == 'MS' else 0
            batch_y_target = batch_y[:, :, f_dim:]
            y_0_hat_batch = y_0_hat_batch[:, :, f_dim:]

            # Sample noise
            batch_size = batch_x.shape[0]
            y_T_mean = torch.randn_like(y_0_hat_batch).to(device)

            # Collect samples
            all_samples = []

            for _ in range(n_samples):
                # Sample using flow matching
                y_sampled = sample_flow_matching(
                    model.model, batch_x, batch_x_mark, y_0_hat_batch, y_T_mean,
                    model.args.timesteps, solver=model.args.ode_solver,
                    rtol=model.args.ode_rtol, atol=model.args.ode_atol,
                    cov_embedding=cov_embedding
                )
                # Check if y_sampled is a list and extract the tensor
                if isinstance(y_sampled, list):
                    y_sampled = y_sampled[0]  # Take first element if it's a list
                all_samples.append(y_sampled.cpu())

            # Stack samples: [batch_size, n_samples, pred_len, n_features]
            sampled_output = torch.stack(all_samples, dim=1)

        return sampled_output

    print("Sampling function defined: generate_predictions")
    print(f"  - Configured for {model.args.timesteps} timesteps")
    print(f"  - Using ODE solver: {model.args.ode_solver}")

    return (generate_predictions,)




@app.cell
def _(np, perturbation_results, plt):
    # Individual trajectory visualization
    print("Creating individual trajectory visualizations...")

    individuals = perturbation_results['individuals']
    pred_before = perturbation_results['predictions_before']
    pred_after = perturbation_results['predictions_after']
    batch_x_vis = perturbation_results['batch_x']
    batch_y_vis = perturbation_results['batch_y']

    # Focus on glucose channel (last channel, index 8)
    glucose_channel_idx = 8

    # Create figure with subplots for first 6 individuals
    n_plots = min(6, len(individuals))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for plot_idx in range(n_plots):
        idx = individuals[plot_idx]
        ax = axes[plot_idx]

        # Get data for this individual
        history_glucose = batch_x_vis[idx, :, glucose_channel_idx].cpu().numpy()
        true_future_glucose = batch_y_vis[idx, :, glucose_channel_idx].cpu().numpy()

        # Get predictions (glucose channel)
        preds_before = pred_before[idx, :, :, glucose_channel_idx-1].numpy()  # -1 because f_dim removes one
        preds_after = pred_after[idx, :, :, glucose_channel_idx-1].numpy()

        # Calculate statistics
        mean_before = np.mean(preds_before, axis=0)
        std_before = np.std(preds_before, axis=0)
        mean_after = np.mean(preds_after, axis=0)
        std_after = np.std(preds_after, axis=0)

        # Create time axes
        seq_len = len(history_glucose)
        pred_len = len(mean_before)
        time_history = np.arange(-seq_len, 0)
        time_pred = np.arange(0, pred_len)

        # Plot history
        ax.plot(time_history, history_glucose, 'k-', label='History', linewidth=1.5)

        # Plot ground truth
        ax.plot(time_pred, true_future_glucose[:pred_len], 'g--', label='Ground Truth', linewidth=2)

        # Plot predictions before perturbation
        ax.plot(time_pred, mean_before, 'b-', label='Before Perturbation', linewidth=1.5)
        ax.fill_between(time_pred, mean_before - std_before, mean_before + std_before, 
                        color='blue', alpha=0.2)

        # Plot predictions after perturbation
        ax.plot(time_pred, mean_after, 'r-', label='After Perturbation', linewidth=1.5)
        ax.fill_between(time_pred, mean_after - std_after, mean_after + std_after, 
                        color='red', alpha=0.2)

        # Add labels and title
        if perturbation_results['original_hba1c']:
            orig_hba1c = perturbation_results['original_hba1c'][plot_idx]
            ax.set_title(f'Individual {idx} (HbA1c: {orig_hba1c:.1f})', fontsize=10)
        else:
            ax.set_title(f'Individual {idx}', fontsize=10)

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Glucose Level')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

        if plot_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Perturbation Analysis: Effect of HbA1c & Steps Increase on Glucose Predictions', fontsize=14)
    plt.tight_layout()
    plt.show()

    print("✓ Individual trajectory plots created")

    return


@app.cell
def _(np, pd, perturbation_results):
    # Summary statistics
    print("\nCalculating summary statistics...")

    individuals_stats = perturbation_results['individuals']
    pred_before_stats = perturbation_results['predictions_before']
    pred_after_stats = perturbation_results['predictions_after']

    # Focus on glucose channel
    glucose_idx = 8 - 1  # Adjusted for f_dim

    summary_data = []

    for i, idx in enumerate(individuals_stats):
        # Get predictions for this individual
        preds_before_i = pred_before_stats[idx, :, :, glucose_idx].numpy()
        preds_after_i = pred_after_stats[idx, :, :, glucose_idx].numpy()

        # Calculate statistics
        mean_before = np.mean(preds_before_i)
        mean_after = np.mean(preds_after_i)
        std_before = np.mean(np.std(preds_before_i, axis=0))
        std_after = np.mean(np.std(preds_after_i, axis=0))

        # Calculate changes
        mean_change = mean_after - mean_before
        std_change = std_after - std_before
        mean_change_pct = (mean_change / abs(mean_before)) * 100 if mean_before != 0 else 0
        std_change_pct = (std_change / std_before) * 100 if std_before != 0 else 0

        summary_data.append({
            'Individual': idx,
            'Mean_Before': mean_before,
            'Mean_After': mean_after,
            'Mean_Change': mean_change,
            'Mean_Change_%': mean_change_pct,
            'Std_Before': std_before,
            'Std_After': std_after,
            'Std_Change': std_change,
            'Std_Change_%': std_change_pct
        })

    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Print summary
    print("\n" + "="*70)
    print("PERTURBATION EFFECT SUMMARY")
    print("="*70)
    print(f"\nHbA1c increased by: {perturbation_results['percentage_increase_hba1c']}%")
    print(f"Steps increased by: {perturbation_results['percentage_increase_steps']}%")
    print("\nIndividual Effects on Glucose Predictions:")
    print("-"*70)

    for _, row in summary_df.iterrows():
        print(f"\nIndividual {int(row['Individual'])}:")
        print(f"  Mean prediction change: {row['Mean_Change']:.3f} ({row['Mean_Change_%']:.1f}%)")
        print(f"  Uncertainty change: {row['Std_Change']:.3f} ({row['Std_Change_%']:.1f}%)")

        if row['Std_Change'] > 0:
            print(f"  → Uncertainty INCREASED")
        else:
            print(f"  → Uncertainty DECREASED")

    # Aggregate statistics
    print("\n" + "="*70)
    print("AGGREGATE STATISTICS")
    print("="*70)
    print(f"Average mean change: {summary_df['Mean_Change'].mean():.3f} ± {summary_df['Mean_Change'].std():.3f}")
    print(f"Average mean change %: {summary_df['Mean_Change_%'].mean():.1f}% ± {summary_df['Mean_Change_%'].std():.1f}%")
    print(f"Average uncertainty change: {summary_df['Std_Change'].mean():.3f} ± {summary_df['Std_Change'].std():.3f}")

    # Count increases vs decreases
    uncertainty_increased = (summary_df['Std_Change'] > 0).sum()
    uncertainty_decreased = (summary_df['Std_Change'] <= 0).sum()
    print(f"\nUncertainty increased for {uncertainty_increased}/{len(summary_df)} individuals")
    print(f"Uncertainty decreased for {uncertainty_decreased}/{len(summary_df)} individuals")

    return (summary_df,)


@app.cell
def _(go, make_subplots, np, perturbation_results):
    # Comparative analysis with interactive plots
    print("Creating comparative analysis visualizations...")

    individuals_comp = perturbation_results['individuals']
    pred_before_comp = perturbation_results['predictions_before']
    pred_after_comp = perturbation_results['predictions_after']
    batch_y_comp = perturbation_results['batch_y']

    glucose_idx_comp = 8 - 1  # Adjusted for f_dim

    # Create subplot figure
    n_individuals = len(individuals_comp)
    fig_comp = make_subplots(
        rows=n_individuals, 
        cols=1,
        subplot_titles=[f'Individual {idx}' for idx in individuals_comp],
        vertical_spacing=0.05
    )

    for i, idx in enumerate(individuals_comp):
        row = i + 1

        # Get data
        true_glucose = batch_y_comp[idx, :, 8].cpu().numpy()  # Use original channel index
        preds_before_i = pred_before_comp[idx, :, :, glucose_idx_comp].numpy()
        preds_after_i = pred_after_comp[idx, :, :, glucose_idx_comp].numpy()

        # Calculate means
        mean_before_i = np.mean(preds_before_i, axis=0)
        mean_after_i = np.mean(preds_after_i, axis=0)

        # Time axis
        time_axis = np.arange(len(mean_before_i))

        # Add traces
        fig_comp.add_trace(
            go.Scatter(x=time_axis, y=true_glucose[:len(mean_before_i)], 
                      mode='lines', name='Ground Truth',
                      line=dict(color='green', dash='dash'),
                      showlegend=(i==0)),
            row=row, col=1
        )

        fig_comp.add_trace(
            go.Scatter(x=time_axis, y=mean_before_i,
                      mode='lines', name='Before Perturbation',
                      line=dict(color='blue'),
                      showlegend=(i==0)),
            row=row, col=1
        )

        fig_comp.add_trace(
            go.Scatter(x=time_axis, y=mean_after_i,
                      mode='lines', name='After Perturbation',
                      line=dict(color='red'),
                      showlegend=(i==0)),
            row=row, col=1
        )

        # Add difference
        difference = mean_after_i - mean_before_i
        fig_comp.add_trace(
            go.Scatter(x=time_axis, y=difference,
                      mode='lines', name='Difference',
                      line=dict(color='purple', dash='dot'),
                      showlegend=(i==0)),
            row=row, col=1
        )

    fig_comp.update_layout(
        height=200*n_individuals,
        title_text="Comparative Analysis: Perturbation Effects Across Individuals",
        showlegend=True
    )

    fig_comp.update_xaxes(title_text="Time Steps")
    fig_comp.update_yaxes(title_text="Glucose Level")

    print("✓ Comparative analysis visualization created")

    return (fig_comp,)


@app.cell
def _(fig_comp):
    # Display comparative analysis
    fig_comp
    return


@app.cell
def _(np, perturbation_results, plt, summary_df):
    # Visualize the individual with most significant perturbation change
    print("Identifying and visualizing most affected individual...")

    # Find individual with largest absolute mean change
    max_mean_change_idx = np.argmax(np.abs(summary_df['Mean_Change'].values))
    max_mean_change_individual = int(summary_df.iloc[max_mean_change_idx]['Individual'])

    # Find individual with largest uncertainty change
    max_std_change_idx = np.argmax(np.abs(summary_df['Std_Change'].values))
    max_std_change_individual = int(summary_df.iloc[max_std_change_idx]['Individual'])

    # Get the position in the individuals array
    individuals_array = perturbation_results['individuals']
    mean_change_pos = np.where(individuals_array == max_mean_change_individual)[0][0]
    std_change_pos = np.where(individuals_array == max_std_change_individual)[0][0]

    # Create detailed visualization for these individuals
    fig_detail, axes_detail = plt.subplots(2, 2, figsize=(16, 12))

    # Helper function to plot detailed analysis
    def plot_detailed_perturbation(ax_main, ax_diff, individual_idx, pos_in_array, title_suffix):
        # Get all data for this individual
        batch_x_detail = perturbation_results['batch_x']
        batch_y_detail = perturbation_results['batch_y']
        pred_before_detail = perturbation_results['predictions_before']
        pred_after_detail = perturbation_results['predictions_after']

        # Get time series data (all channels)
        history_all_channels = batch_x_detail[individual_idx, :, :].cpu().numpy()

        # Get glucose predictions
        glucose_idx_detail = 8 - 1  # Adjusted for f_dim
        preds_before_detail = pred_before_detail[individual_idx, :, :, glucose_idx_detail].numpy()
        preds_after_detail = pred_after_detail[individual_idx, :, :, glucose_idx_detail].numpy()

        # Calculate statistics
        mean_before_detail = np.mean(preds_before_detail, axis=0)
        std_before_detail = np.std(preds_before_detail, axis=0)
        mean_after_detail = np.mean(preds_after_detail, axis=0)
        std_after_detail = np.std(preds_after_detail, axis=0)

        # Get true future
        true_future_detail = batch_y_detail[individual_idx, :, 8].cpu().numpy()

        # Time axes
        seq_len_detail = history_all_channels.shape[0]
        pred_len_detail = mean_before_detail.shape[0]
        time_history_detail = np.arange(-seq_len_detail, 0)
        time_pred_detail = np.arange(0, pred_len_detail)

        # Main plot: Glucose with perturbation effects
        ax_main.plot(time_history_detail, history_all_channels[:, 8], 'k-', 
                    label='Historical Glucose', linewidth=2, alpha=0.8)

        # Show steps history (channel 3) on secondary y-axis
        ax_steps = ax_main.twinx()
        ax_steps.plot(time_history_detail, history_all_channels[:, 3], 
                     color='orange', alpha=0.3, label='Historical Steps (scaled)')
        ax_steps.set_ylabel('Steps (scaled)', color='orange')
        ax_steps.tick_params(axis='y', labelcolor='orange')

        # Plot predictions
        ax_main.plot(time_pred_detail, true_future_detail[:pred_len_detail], 
                    'g--', label='Ground Truth', linewidth=2.5)

        # Before perturbation
        ax_main.plot(time_pred_detail, mean_before_detail, 'b-', 
                    label='Before (Original)', linewidth=2)
        ax_main.fill_between(time_pred_detail, 
                            mean_before_detail - std_before_detail,
                            mean_before_detail + std_before_detail,
                            color='blue', alpha=0.15, label='±1 STD (Before)')

        # After perturbation
        ax_main.plot(time_pred_detail, mean_after_detail, 'r-', 
                    label='After (Perturbed)', linewidth=2)
        ax_main.fill_between(time_pred_detail,
                            mean_after_detail - std_after_detail,
                            mean_after_detail + std_after_detail,
                            color='red', alpha=0.15, label='±1 STD (After)')

        # Add vertical line at prediction start
        ax_main.axvline(x=0, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)

        # Labels and title
        stats_row = summary_df.iloc[pos_in_array]
        if perturbation_results['original_hba1c']:
            orig_hba1c_detail = perturbation_results['original_hba1c'][pos_in_array]
            new_hba1c_detail = orig_hba1c_detail * (1 + perturbation_results['percentage_increase_hba1c']/100)
            ax_main.set_title(f'Individual {individual_idx} - {title_suffix}\n' + 
                            f'HbA1c: {orig_hba1c_detail:.1f} → {new_hba1c_detail:.1f} | ' +
                            f'Mean Δ: {stats_row["Mean_Change"]:.3f} ({stats_row["Mean_Change_%"]:.1f}%) | ' +
                            f'STD Δ: {stats_row["Std_Change"]:.3f} ({stats_row["Std_Change_%"]:.1f}%)',
                            fontsize=11)
        else:
            ax_main.set_title(f'Individual {individual_idx} - {title_suffix}', fontsize=11)

        ax_main.set_xlabel('Time Steps')
        ax_main.set_ylabel('Glucose Level')
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(loc='upper left', fontsize=9)

        # Difference plot
        difference_detail = mean_after_detail - mean_before_detail
        std_diff_detail = std_after_detail - std_before_detail

        ax_diff.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax_diff.plot(time_pred_detail, difference_detail, 'purple', 
                    linewidth=2, label='Mean Difference')
        ax_diff.fill_between(time_pred_detail, difference_detail - std_diff_detail,
                            difference_detail + std_diff_detail,
                            color='purple', alpha=0.2)

        # Show uncertainty change
        ax_diff2 = ax_diff.twinx()
        ax_diff2.plot(time_pred_detail, std_diff_detail, 
                     'orange', linewidth=2, alpha=0.7, label='STD Difference')
        ax_diff2.set_ylabel('STD Difference', color='orange')
        ax_diff2.tick_params(axis='y', labelcolor='orange')

        ax_diff.set_xlabel('Time Steps')
        ax_diff.set_ylabel('Mean Difference', color='purple')
        ax_diff.tick_params(axis='y', labelcolor='purple')
        ax_diff.grid(True, alpha=0.3)
        ax_diff.set_title(f'Perturbation Effect Over Time', fontsize=10)

    # Plot for individual with max mean change
    plot_detailed_perturbation(axes_detail[0, 0], axes_detail[0, 1], 
                              max_mean_change_individual, mean_change_pos,
                              "Maximum Mean Change")

    # Plot for individual with max uncertainty change
    plot_detailed_perturbation(axes_detail[1, 0], axes_detail[1, 1],
                              max_std_change_individual, std_change_pos, 
                              "Maximum Uncertainty Change")

    plt.suptitle(f'Detailed Perturbation Analysis: Most Affected Individuals\n' + 
                f'(HbA1c +{perturbation_results["percentage_increase_hba1c"]}%, ' +
                f'Steps +{perturbation_results["percentage_increase_steps"]}%)',
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    print(f"\n✓ Detailed visualization created for:")
    print(f"  - Individual {max_mean_change_individual}: Largest mean change ({summary_df.iloc[max_mean_change_idx]['Mean_Change_%']:.1f}%)")
    print(f"  - Individual {max_std_change_individual}: Largest uncertainty change ({summary_df.iloc[max_std_change_idx]['Std_Change_%']:.1f}%)")

    return


@app.cell
def _(go, np, summary_df):
    # Create effect magnitude heatmap
    print("Creating effect magnitude visualization...")

    # Prepare data for heatmap
    metrics = ['Mean_Change_%', 'Std_Change_%']

    # Create heatmap data
    heatmap_data = []
    for metric in metrics:
        heatmap_data.append(summary_df[metric].values)

    heatmap_data = np.array(heatmap_data)

    # Create heatmap
    fig_heatmap_effect = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"Ind {int(idx)}" for idx in summary_df['Individual']],
        y=['Mean Change (%)', 'Uncertainty Change (%)'],
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(heatmap_data, 1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Change (%)")
    ))

    fig_heatmap_effect.update_layout(
        title='Perturbation Effect Magnitude Across Individuals',
        xaxis_title='Individuals',
        yaxis_title='Metrics',
        width=800,
        height=300
    )

    print("✓ Effect magnitude heatmap created")

    return (fig_heatmap_effect,)


@app.cell
def _(fig_heatmap_effect):
    # Display effect heatmap
    fig_heatmap_effect
    return


@app.cell
def _(perturbation_results, summary_df):
    # Export results
    print("\nExporting results...")

    # Prepare export data
    export_data = summary_df.copy()
    export_data['HbA1c_Increase_%'] = perturbation_results['percentage_increase_hba1c']
    export_data['Steps_Increase_%'] = perturbation_results['percentage_increase_steps']

    # Add original HbA1c values if available
    if perturbation_results['original_hba1c']:
        hba1c_dict = {perturbation_results['individuals'][i]: perturbation_results['original_hba1c'][i] 
                      for i in range(len(perturbation_results['individuals']))}
        export_data['Original_HbA1c'] = export_data['Individual'].map(hba1c_dict)

    # Save to CSV
    output_filename = 'perturbation_analysis_results.csv'
    export_data.to_csv(output_filename, index=False)

    print(f"✓ Results saved to {output_filename}")
    print(f"  - {len(export_data)} individuals analyzed")
    print(f"  - {len(export_data.columns)} metrics recorded")

    # Display first few rows
    print("\nFirst 5 rows of exported data:")
    print(export_data.head().to_string())

    print("\n" + "="*70)
    print("PERTURBATION ANALYSIS COMPLETE")
    print("="*70)
    print("Summary:")
    print(f"  - Analyzed {len(perturbation_results['individuals'])} individuals")
    print(f"  - HbA1c increased by {perturbation_results['percentage_increase_hba1c']}%")
    print(f"  - Steps increased by {perturbation_results['percentage_increase_steps']}%")
    print(f"  - Average glucose prediction change: {export_data['Mean_Change_%'].mean():.1f}%")
    print(f"  - Results exported to: {output_filename}")

    return


if __name__ == "__main__":
    app.run()
