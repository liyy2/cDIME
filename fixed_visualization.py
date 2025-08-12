"""
Fixed visualization that correctly shows the data alignment.
The key insight: seq_y[:label_len] OVERLAPS with seq_x[-label_len:]
"""

def plot_individual_with_history_FIXED(samples, individual_idx, batch_x_orig, batch_y_orig, 
                                       label, color='blue', channel_idx=8, args=None):
    """
    CORRECTED plot showing the true data relationship without artificial gaps.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Get the data
    history = batch_x_orig[individual_idx, :, channel_idx].cpu().numpy()  # Length: seq_len (72)
    full_batch_y = batch_y_orig[individual_idx, :, channel_idx].cpu().numpy()  # Length: label_len + pred_len (80)
    
    # Split batch_y into its components
    label_portion = full_batch_y[:args.label_len]  # First 32 points (overlap with history)
    true_future = full_batch_y[args.label_len:]    # Last 48 points (true future)
    
    # CRITICAL: Verify the overlap
    history_overlap = history[-args.label_len:]  # Last 32 points of history
    overlap_diff = np.abs(history_overlap - label_portion).mean()
    
    # Create proper time axis
    # We'll set the END of history as t=0 (present moment)
    time_history = np.arange(-args.seq_len, 0)  # -72 to -1
    time_future = np.arange(0, args.pred_len)    # 0 to 47
    
    # Extract predictions for the future portion only
    if samples.ndim == 4:
        preds = samples[:, individual_idx, :, channel_idx]
    elif samples.ndim == 3:
        preds = samples[individual_idx, :, :]
    else:
        preds = samples[individual_idx, :]
    
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    
    # Compute statistics
    mean_pred = preds.mean(axis=0)
    ci_lower = np.percentile(preds, 2.5, axis=0)
    ci_upper = np.percentile(preds, 97.5, axis=0)
    
    # Plot historical data (full history)
    ax.plot(time_history, history, 'k-', label='Historical Input', linewidth=2, alpha=0.8)
    
    # Highlight the overlap region in history
    time_overlap = np.arange(-args.label_len, 0)
    ax.fill_between(time_overlap, history[-args.label_len:] - 0.5, history[-args.label_len:] + 0.5,
                    color='yellow', alpha=0.2, label=f'Decoder Context (last {args.label_len} steps)')
    
    # Mark the present moment
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Present (t=0)')
    
    # Plot ground truth future
    ax.plot(time_future, true_future, 'g--', label='Ground Truth Future', linewidth=2, alpha=0.7)
    
    # Plot predictions
    ax.plot(time_future, mean_pred, color=color, label=f'{label} Prediction', linewidth=2.5)
    ax.fill_between(time_future, ci_lower, ci_upper, 
                    color=color, alpha=0.2, label=f'{label} 95% CI')
    
    # Add overlap verification text
    if overlap_diff < 0.01:
        verification_text = f"✓ Data alignment verified (overlap diff: {overlap_diff:.4f})"
        text_color = 'green'
    else:
        verification_text = f"⚠ Potential alignment issue (overlap diff: {overlap_diff:.4f})"
        text_color = 'red'
    
    ax.text(0.02, 0.98, verification_text, transform=ax.transAxes, 
           verticalalignment='top', color=text_color, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add annotation explaining the structure
    ax.annotate('', xy=(time_overlap[0], history[-args.label_len]), 
               xytext=(time_overlap[0], history[-args.label_len] - 5),
               arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
    ax.text(time_overlap[len(time_overlap)//2], history[-args.label_len] - 7, 
           'Decoder sees this context', ha='center', color='orange', fontweight='bold')
    
    # Formatting
    ax.set_title(f'Individual {individual_idx}: {label} (Corrected Visualization)')
    ax.set_xlabel('Time Steps (relative to present)')
    ax.set_ylabel('Glucose Level')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add text box with data structure explanation
    info_text = (f"Data Structure:\n"
                f"• History (batch_x): t=-{args.seq_len} to t=0\n"
                f"• Decoder context: last {args.label_len} of history\n"
                f"• Predictions: t=0 to t={args.pred_len-1}")
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
           fontsize=8, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    return fig

# Print explanation
print("="*80)
print("VISUALIZATION FIX EXPLANATION")
print("="*80)
print("""
The original visualization might have shown a 'gap' because it was plotting:
- batch_x as history
- batch_y[:label_len] as a separate segment

But actually, batch_y[:label_len] OVERLAPS with batch_x[-label_len:].
They are the SAME data points!

The corrected visualization:
1. Shows the full history (batch_x) from t=-72 to t=0
2. Highlights the decoder context region (last 32 points)
3. Shows predictions starting from t=0
4. Verifies that the overlap is correct

There is NO gap in the data. The model architecture intentionally uses
an overlapping region to provide context to the decoder.
""")
print("="*80)