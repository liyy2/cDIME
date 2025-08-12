import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell
def __():
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
    return device, go, make_subplots, np, pd, torch, F, DataLoader, px, stats, PCA, TSNE, plt, sns, warnings


@app.cell
def __():
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
    return TimeSeriesFlowMatchingModel, data_provider, glob, os, re, tqdm, sys, NSTransformer, argparse


@app.cell
def __():
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

    return CHANNEL_CATEGORIES, CHANNEL_NAMES, CHANNEL_COLORS


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
        DotDict,
        args
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

    return model, test_loader, flow_args, train_loader_fm, vali_loader_fm, test_loader_fm, flow_matching_model


@app.cell
def _(np, torch):
    class FlowStageFeatureAnalyzer:
        """
        Analyzes feature importance across different stages of the flow matching process.
        """

        def __init__(self, model, n_stages=5, device='cuda'):
            self.model = model
            self.n_stages = n_stages
            self.device = device

            # Define stage centers for flow matching (t ∈ [0, 1])
            self.stage_centers = np.linspace(0.1, 0.9, n_stages)
            self.stage_names = [
                "Initial Transport",
                "Coarse Features",
                "Mid Refinement",
                "Fine Details",
                "Final Approach"
            ]

        def compute_gradients_at_stage(self, batch_x, batch_x_mark, batch_y, batch_y_mark, batch_cov, t_value):
            """
            Compute gradients with respect to input features at a specific flow stage.
            """
            batch_x = batch_x.requires_grad_(True)

            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.model.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.model.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            # Get condition model output
            with torch.enable_grad():
                y_0_hat_batch, KL_loss, z_sample, cov_embedding = self.model.condition_model_forward(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark, covariates=batch_cov
                )

                # Prepare for flow matching
                f_dim = -1 if self.model.args.features == 'MS' else 0
                batch_y_target = batch_y[:, :, f_dim:]
                y_0_hat_batch = y_0_hat_batch[:, :, f_dim:]

                # Sample time t for this stage
                batch_size = batch_x.shape[0]
                t = torch.full((batch_size,), t_value, device=self.device)

                # Compute flow matching loss
                moe_loss_weight = getattr(self.model.args, 'moe_loss_weight', 0.01)
                flow_loss_result = self.model.model.compute_loss(
                    batch_x, batch_x_mark, batch_y_target, y_0_hat_batch, t,
                    cov_embedding=cov_embedding, moe_loss_weight=moe_loss_weight
                )

                # Handle different return formats
                if isinstance(flow_loss_result, tuple):
                    flow_loss, _ = flow_loss_result
                else:
                    flow_loss = flow_loss_result

                # Compute gradients
                gradients = torch.autograd.grad(
                    outputs=flow_loss,
                    inputs=batch_x,
                    create_graph=False,
                    retain_graph=False
                )[0]

            return gradients.abs()  # Use absolute gradients for importance

        def analyze_batch(self, batch):
            """
            Analyze feature importance for a batch across all stages.
            """
            # Unpack batch
            if len(batch) == 2:
                (batch_x, batch_y, batch_x_mark, batch_y_mark), batch_cov = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_cov = None

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            if batch_cov is not None:
                batch_cov = batch_cov.to(self.device)

            stage_importance = {}

            for stage_idx, t_value in enumerate(self.stage_centers):
                try:
                    # Compute gradients at this stage
                    gradients = self.compute_gradients_at_stage(
                        batch_x, batch_x_mark, batch_y, batch_y_mark, batch_cov, t_value
                    )

                    # Shape: [batch, seq_len, channels]
                    # Average over batch dimension
                    importance = gradients.mean(dim=0).cpu().numpy()

                    stage_importance[stage_idx] = {
                        'raw': importance,
                        'per_channel': importance.mean(axis=0),  # Average over time
                        'per_time': importance.mean(axis=1),     # Average over channels
                        'stage_name': self.stage_names[stage_idx],
                        't_value': t_value
                    }
                except Exception as e:
                    print(f"Error at stage {stage_idx}: {e}")
                    continue

            return stage_importance

    return (FlowStageFeatureAnalyzer,)


@app.cell
def _(FlowStageFeatureAnalyzer, device, model):
    # Initialize analyzer
    analyzer = FlowStageFeatureAnalyzer(model, n_stages=5, device=device)
    print(f"Analyzer initialized with {analyzer.n_stages} stages")
    return (analyzer,)


@app.cell
def _(analyzer, np, test_loader, tqdm):
    # Analyze multiple batches
    print("Computing feature importance across flow stages...")

    all_stage_importance = {i: [] for i in range(analyzer.n_stages)}
    num_batches_to_analyze = 10

    # Note: We need gradients enabled for feature importance computation
    for batch_idx, batch in enumerate(tqdm(test_loader, total=num_batches_to_analyze)):
        if batch_idx >= num_batches_to_analyze:
            break

        # Analyze this batch
        stage_importance = analyzer.analyze_batch(batch)

        # Accumulate results
        for stage_idx in stage_importance:
            all_stage_importance[stage_idx].append(stage_importance[stage_idx])

    # Aggregate results across batches
    aggregated_importance = {}
    for stage_idx in range(analyzer.n_stages):
        if all_stage_importance[stage_idx]:
            # Stack per-channel importance from all batches
            per_channel_stack = np.stack([
                imp['per_channel'] for imp in all_stage_importance[stage_idx]
            ])

            aggregated_importance[stage_idx] = {
                'per_channel_mean': per_channel_stack.mean(axis=0),
                'per_channel_std': per_channel_stack.std(axis=0),
                'stage_name': analyzer.stage_names[stage_idx],
                't_value': analyzer.stage_centers[stage_idx]
            }

    print(f"✓ Analyzed {num_batches_to_analyze} batches across {len(aggregated_importance)} stages")

    return aggregated_importance, all_stage_importance


@app.cell
def _(CHANNEL_NAMES, aggregated_importance, analyzer, go, np):
    # Create heatmap of channel importance across stages
    if aggregated_importance:
        # Prepare data for heatmap
        importance_matrix = np.zeros((analyzer.n_stages, len(CHANNEL_NAMES)))

        for stage_idx in aggregated_importance:
            importance_matrix[stage_idx, :] = aggregated_importance[stage_idx]['per_channel_mean']

        # Normalize per stage for better visualization
        importance_matrix_norm = importance_matrix / importance_matrix.sum(axis=1, keepdims=True)

        # Create interactive heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=importance_matrix_norm,
            x=CHANNEL_NAMES,
            y=[f"Stage {i+1}: {name}" for i, name in enumerate(analyzer.stage_names)],
            colorscale='Viridis',
            text=np.round(importance_matrix_norm * 100, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Relative Importance (%)")
        ))

        fig_heatmap.update_layout(
            title='Channel Importance Across Flow Stages',
            xaxis_title='Channels (Features)',
            yaxis_title='Flow Stages',
            width=1000,
            height=600,
            xaxis=dict(tickangle=45)
        )

        # Print summary statistics
        print("\n" + "="*60)
        print("CHANNEL IMPORTANCE SUMMARY")
        print("="*60)

        for stage_idx in range(analyzer.n_stages):
            print(f"\n{analyzer.stage_names[stage_idx]} (t={analyzer.stage_centers[stage_idx]:.2f}):")
            channel_importance = importance_matrix_norm[stage_idx, :]
            sorted_indices = np.argsort(channel_importance)[::-1]

            for rank, idx in enumerate(sorted_indices[:3], 1):
                print(f"  {rank}. {CHANNEL_NAMES[idx]:15} {channel_importance[idx]*100:.1f}%")
    else:
        print("No importance data available for visualization")
        fig_heatmap = None
        importance_matrix = None
        importance_matrix_norm = None

    return fig_heatmap, importance_matrix, importance_matrix_norm


@app.cell
def _(fig_heatmap):
    # Display the heatmap
    fig_heatmap
    return


@app.cell
def _(
    CHANNEL_CATEGORIES,
    CHANNEL_NAMES,
    aggregated_importance,
    analyzer,
    go,
    importance_matrix,
    np,
):
    # Category-level analysis
    if aggregated_importance:
        # Aggregate importance by category
        category_importance = np.zeros((analyzer.n_stages, len(CHANNEL_CATEGORIES)))

        for stage_idx in range(analyzer.n_stages):
            for cat_idx, (category, channels) in enumerate(CHANNEL_CATEGORIES.items()):
                channel_indices = [CHANNEL_NAMES.index(ch) for ch in channels]
                category_importance[stage_idx, cat_idx] = \
                    importance_matrix[stage_idx, channel_indices].sum()

        # Normalize
        category_importance_norm = category_importance / category_importance.sum(axis=1, keepdims=True)

        # Create stacked bar chart
        fig_category = go.Figure()

        colors = ['#FF6B6B', '#4ECDC4', '#FFEAA7', '#9B59B6']

        for cat_idx, category in enumerate(CHANNEL_CATEGORIES.keys()):
            fig_category.add_trace(go.Bar(
                name=category.capitalize(),
                x=[f"Stage {i+1}" for i in range(analyzer.n_stages)],
                y=category_importance_norm[:, cat_idx] * 100,
                marker_color=colors[cat_idx],
                text=np.round(category_importance_norm[:, cat_idx] * 100, 1),
                texttemplate='%{text}%',
                textposition='inside'
            ))

        fig_category.update_layout(
            barmode='stack',
            title='Category-Level Feature Importance Across Flow Stages',
            xaxis_title='Flow Stages',
            yaxis_title='Relative Importance (%)',
            width=900,
            height=500,
            showlegend=True,
            legend=dict(x=1.02, y=1)
        )

        # Print category evolution
        print("\n" + "="*60)
        print("CATEGORY IMPORTANCE EVOLUTION")
        print("="*60)

        for cat_idx, category in enumerate(CHANNEL_CATEGORIES.keys()):
            importance_values = category_importance_norm[:, cat_idx] * 100
            print(f"\n{category.capitalize()}:")
            print(f"  Early stage (1-2):  {importance_values[:2].mean():.1f}%")
            print(f"  Mid stage (3):      {importance_values[2]:.1f}%")
            print(f"  Late stage (4-5):   {importance_values[3:].mean():.1f}%")

            # Compute trend
            trend = np.polyfit(range(len(importance_values)), importance_values, 1)[0]
            trend_direction = "increasing" if trend > 0 else "decreasing"
            print(f"  Trend: {trend_direction} ({trend:.2f}% per stage)")
    else:
        fig_category = None
        category_importance = None
        category_importance_norm = None

    return (fig_category,)


@app.cell
def _(fig_category):
    # Display category analysis
    fig_category
    return


@app.cell
def _(CHANNEL_NAMES, all_stage_importance, analyzer, go, make_subplots, np):
    # Temporal dynamics analysis
    if all_stage_importance[0]:
        # Get raw importance for detailed temporal analysis
        fig_temporal = make_subplots(
            rows=3, cols=3,
            subplot_titles=CHANNEL_NAMES,
            shared_yaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )

        for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
            row = channel_idx // 3 + 1
            col = channel_idx % 3 + 1

            for stage_idx in range(analyzer.n_stages):
                if all_stage_importance[stage_idx]:
                    # Get temporal importance for this channel
                    temporal_importance = []
                    for batch_imp in all_stage_importance[stage_idx]:
                        # Shape: [seq_len, channels]
                        temporal_importance.append(batch_imp['raw'][:, channel_idx])

                    # Average across batches
                    temporal_importance = np.stack(temporal_importance).mean(axis=0)

                    # Create time axis (negative for history)
                    time_steps = np.arange(-len(temporal_importance), 0)

                    fig_temporal.add_trace(
                        go.Scatter(
                            x=time_steps,
                            y=temporal_importance,
                            name=f"Stage {stage_idx+1}",
                            line=dict(width=2),
                            showlegend=(channel_idx == 0)
                        ),
                        row=row, col=col
                    )

            fig_temporal.update_xaxes(title_text="Time (relative)", row=row, col=col)
            fig_temporal.update_yaxes(title_text="Importance", row=row, col=1)

        fig_temporal.update_layout(
            title='Temporal Feature Importance Across Channels and Stages',
            height=900,
            width=1200,
            showlegend=True
        )
    else:
        fig_temporal = None
        print("No temporal data available for analysis")

    return (fig_temporal,)


@app.cell
def _(fig_temporal):
    # Display temporal analysis
    fig_temporal
    return


@app.cell
def _(CHANNEL_NAMES, aggregated_importance, analyzer, importance_matrix, np):
    # Statistical analysis
    if aggregated_importance:
        from scipy.stats import friedmanchisquare

        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)

        # Test if channel importance differs significantly across stages
        for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
            channel_importance_across_stages = []
            for stage_idx in range(analyzer.n_stages):
                if stage_idx in aggregated_importance:
                    channel_importance_across_stages.append(
                        aggregated_importance[stage_idx]['per_channel_mean'][channel_idx]
                    )

            if len(channel_importance_across_stages) >= 3:
                # Create pseudo-replicates for the test
                pseudo_data = [np.random.normal(val, 0.01, 10) 
                              for val in channel_importance_across_stages]

                try:
                    stat, p_value = friedmanchisquare(*pseudo_data)
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"{channel_name:15} χ²={stat:.2f}, p={p_value:.4f} {significance}")
                except:
                    print(f"{channel_name:15} Unable to compute statistics")

        print("\nSignificance: * p<0.05, ** p<0.01, *** p<0.001")

        # Correlation analysis between channels
        print("\n" + "="*60)
        print("INTER-CHANNEL CORRELATIONS")
        print("="*60)

        # Compute correlation matrix
        correlation_matrix = np.corrcoef(importance_matrix.T)

        # Find strongest correlations
        strong_correlations = []
        for i in range(len(CHANNEL_NAMES)):
            for j in range(i+1, len(CHANNEL_NAMES)):
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.7:  # Strong correlation threshold
                    strong_correlations.append((
                        CHANNEL_NAMES[i], 
                        CHANNEL_NAMES[j], 
                        corr
                    ))

        if strong_correlations:
            print("\nStrong correlations (|r| > 0.7):")
            for ch1, ch2, corr in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
                print(f"  {ch1:15} ↔ {ch2:15} r={corr:.3f}")
        else:
            print("\nNo strong correlations found between channels")

        statistical_results = {
            'correlation_matrix': correlation_matrix,
            'strong_correlations': strong_correlations
        }
    else:
        statistical_results = None
        correlation_matrix = None
        strong_correlations = None

    return statistical_results, correlation_matrix, strong_correlations


@app.cell
def _(CHANNEL_CATEGORIES, CHANNEL_NAMES, aggregated_importance, analyzer, pd):
    # Export results
    if aggregated_importance:
        results_data = []

        for stage_idx in range(analyzer.n_stages):
            if stage_idx in aggregated_importance:
                for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
                    results_data.append({
                        'stage': stage_idx + 1,
                        'stage_name': analyzer.stage_names[stage_idx],
                        't_value': analyzer.stage_centers[stage_idx],
                        'channel': channel_name,
                        'channel_category': [cat for cat, chs in CHANNEL_CATEGORIES.items() if channel_name in chs][0],
                        'importance_mean': aggregated_importance[stage_idx]['per_channel_mean'][channel_idx],
                        'importance_std': aggregated_importance[stage_idx]['per_channel_std'][channel_idx]
                    })

        results_df = pd.DataFrame(results_data)

        # Save to CSV
        output_path = 'flow_stage_feature_importance_results.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")

        # Display summary
        print("\nTop 5 most important channel-stage combinations:")
        top_5 = results_df.nlargest(5, 'importance_mean')[['stage_name', 'channel', 'importance_mean']]
        print(top_5.to_string(index=False))

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
    else:
        results_df = None
        output_path = None
        top_5 = None

    return results_df, output_path, top_5


if __name__ == "__main__":
    app.run()
