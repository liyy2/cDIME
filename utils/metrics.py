import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def CV_ERROR(pred, true):
    """
    Calculate the Coefficient of Variation Error.
    
    CV Error = (RMSE / mean(true)) * 100
    
    This provides a normalized measure of prediction error relative to the magnitude
    of the true values, expressed as a percentage.
    
    Args:
        pred: predictions
        true: ground truth values
        
    Returns:
        CV error as a percentage
    """
    rmse = RMSE(pred, true)
    mean_true = np.mean(true)
    if mean_true == 0:
        return float('inf')  # Avoid division by zero
    return (rmse / abs(mean_true)) * 100


def denormalize_glucose(data, mean=144.91148743, std=55.13396884):
    """
    Denormalize glucose values back to mg/dL scale.
    Default values are from the glucose dataset statistics.
    """
    return data * std + mean


def hypoglycemia_sensitivity_specificity(pred, true, threshold_mg_dl, time_horizon_steps, 
                                       glucose_mean=144.91148743, glucose_std=55.13396884):
    """
    Calculate sensitivity and specificity for hypoglycemia prediction.
    
    Args:
        pred: predictions (normalized glucose values)
        true: ground truth (normalized glucose values) 
        threshold_mg_dl: glucose threshold in mg/dL (e.g., 70 for Level 1, 54 for Level 2)
        time_horizon_steps: number of timesteps to look ahead (e.g., 3 for 15min, 6 for 30min, 12 for 60min)
        glucose_mean: mean glucose value for denormalization
        glucose_std: std glucose value for denormalization
    
    Returns:
        sensitivity, specificity, positive_predictive_value, negative_predictive_value
    """
    # Denormalize predictions and ground truth to mg/dL
    pred_denorm = denormalize_glucose(pred, glucose_mean, glucose_std)
    true_denorm = denormalize_glucose(true, glucose_mean, glucose_std)
    
    # Ensure we have enough timesteps for the time horizon
    if pred_denorm.shape[-2] < time_horizon_steps:
        time_horizon_steps = pred_denorm.shape[-2]
    
    # Extract predictions and ground truth at the specified time horizon
    # pred_at_horizon: predictions at time_horizon_steps ahead
    # true_at_horizon: ground truth at time_horizon_steps ahead
    pred_at_horizon = pred_denorm[:, time_horizon_steps-1, -1]  # Last channel is glucose
    true_at_horizon = true_denorm[:, time_horizon_steps-1, -1]  # Last channel is glucose
    
    # Binary classification: hypoglycemia (below threshold) vs normal
    pred_hypo = pred_at_horizon < threshold_mg_dl
    true_hypo = true_at_horizon < threshold_mg_dl
    
    # Calculate confusion matrix elements
    true_positives = np.sum((pred_hypo == True) & (true_hypo == True))
    false_positives = np.sum((pred_hypo == True) & (true_hypo == False))
    true_negatives = np.sum((pred_hypo == False) & (true_hypo == False))
    false_negatives = np.sum((pred_hypo == False) & (true_hypo == True))
    
    # Calculate metrics
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
    ppv = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    npv = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0.0
    
    return sensitivity, specificity, ppv, npv


def hypoglycemia_metrics(pred, true, glucose_mean=144.91148743, glucose_std=55.13396884):
    """
    Calculate comprehensive hypoglycemia detection metrics for Level 1 and Level 2 
    at 15, 30, and 60 minute horizons.
    
    Args:
        pred: predictions (normalized glucose values) - shape: [batch, time, features]
        true: ground truth (normalized glucose values) - shape: [batch, time, features]
        glucose_mean: mean glucose value for denormalization
        glucose_std: std glucose value for denormalization
    
    Returns:
        Dictionary with sensitivity/specificity metrics for each condition
    """
    results = {}
    
    # Define thresholds and time horizons
    thresholds = {
        'Level1': 70.0,  # mg/dL
        'Level2': 54.0   # mg/dL
    }
    
    time_horizons = {
        '15min': 3,   # 15 minutes = 3 timesteps at 5-min intervals
        '30min': 6,   # 30 minutes = 6 timesteps at 5-min intervals
        '60min': 12   # 60 minutes = 12 timesteps at 5-min intervals
    }
    
    for threshold_name, threshold_value in thresholds.items():
        for horizon_name, horizon_steps in time_horizons.items():
            try:
                sensitivity, specificity, ppv, npv = hypoglycemia_sensitivity_specificity(
                    pred, true, threshold_value, horizon_steps, glucose_mean, glucose_std
                )
                
                results[f'{threshold_name}_{horizon_name}_sensitivity'] = sensitivity
                results[f'{threshold_name}_{horizon_name}_specificity'] = specificity
                results[f'{threshold_name}_{horizon_name}_ppv'] = ppv
                results[f'{threshold_name}_{horizon_name}_npv'] = npv
                
            except Exception as e:
                # Handle cases where metrics cannot be calculated
                results[f'{threshold_name}_{horizon_name}_sensitivity'] = 0.0
                results[f'{threshold_name}_{horizon_name}_specificity'] = 0.0
                results[f'{threshold_name}_{horizon_name}_ppv'] = 0.0
                results[f'{threshold_name}_{horizon_name}_npv'] = 0.0
    
    return results


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    cv_error = CV_ERROR(pred, true)

    return mae, mse, rmse, mape, mspe, cv_error


def comprehensive_metric(pred, true, glucose_mean=144.91148743, glucose_std=55.13396884):
    """
    Calculate both traditional forecasting metrics and hypoglycemia detection metrics.
    
    Args:
        pred: predictions (normalized glucose values)
        true: ground truth (normalized glucose values)
        glucose_mean: mean glucose value for denormalization
        glucose_std: std glucose value for denormalization
        
    Returns:
        Dictionary containing all metrics
    """
    # Traditional metrics
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    cv_error = CV_ERROR(pred, true)
    
    # Hypoglycemia metrics
    hypo_metrics = hypoglycemia_metrics(pred, true, glucose_mean, glucose_std)
    
    # Combine all metrics
    results = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'mspe': mspe,
        'cv_error': cv_error,
        **hypo_metrics
    }
    
    return results
