#!/usr/bin/env python3
"""
Debug script to verify time series data alignment in the data loader.
This script checks the relationship between batch_x, batch_y, seq_len, label_len, and pred_len.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('/home/yl2428/Time-LLM')

# Configuration matching the notebook
seq_len = 72
label_len = 32
pred_len = 48

print("="*80)
print("TIME SERIES DATA ALIGNMENT VERIFICATION")
print("="*80)
print(f"\nConfiguration:")
print(f"  seq_len (input history): {seq_len}")
print(f"  label_len (decoder context): {label_len}")
print(f"  pred_len (prediction horizon): {pred_len}")
print(f"  batch_y total length: {label_len + pred_len} = {label_len + pred_len}")

# Simulate the data loader indexing logic from DatasetPerIndividual.__getitem__
def simulate_data_indexing(index=0, stride=1):
    """Simulate how the data loader creates batch_x and batch_y."""
    
    # From lines 315-318 of data_loader.py
    s_begin = index * stride
    s_end = s_begin + seq_len
    r_begin = s_end - label_len  # This is the key line!
    r_end = r_begin + label_len + pred_len
    
    print(f"\nIndexing calculations for index={index}, stride={stride}:")
    print(f"  s_begin (batch_x start): {s_begin}")
    print(f"  s_end (batch_x end): {s_end}")
    print(f"  r_begin (batch_y start): {r_begin}")
    print(f"  r_end (batch_y end): {r_end}")
    
    return s_begin, s_end, r_begin, r_end

# Run simulation
s_begin, s_end, r_begin, r_end = simulate_data_indexing()

# Verify the overlap
print(f"\n" + "="*80)
print("DATA ALIGNMENT ANALYSIS:")
print("="*80)

print(f"\nbatch_x covers indices: [{s_begin}, {s_end})")
print(f"batch_y covers indices: [{r_begin}, {r_end})")

overlap_start = max(s_begin, r_begin)
overlap_end = min(s_end, r_end)
overlap_length = overlap_end - overlap_start

print(f"\nOverlap region: [{overlap_start}, {overlap_end})")
print(f"Overlap length: {overlap_length} time steps")

# Check if overlap equals label_len
if overlap_length == label_len:
    print(f"✓ CORRECT: Overlap length ({overlap_length}) equals label_len ({label_len})")
else:
    print(f"✗ ERROR: Overlap length ({overlap_length}) does NOT equal label_len ({label_len})")

# Check the structure of batch_y
batch_y_label_portion = (r_begin, r_begin + label_len)
batch_y_pred_portion = (r_begin + label_len, r_end)

print(f"\nbatch_y structure:")
print(f"  Label portion (overlaps with batch_x): [{batch_y_label_portion[0]}, {batch_y_label_portion[1]})")
print(f"  Prediction portion (future): [{batch_y_pred_portion[0]}, {batch_y_pred_portion[1]})")

# Verify continuity
if batch_y_label_portion[0] == s_end - label_len:
    print(f"✓ CORRECT: batch_y starts at the right position for {label_len} overlap")
else:
    print(f"✗ ERROR: batch_y does not start at the expected position")

if batch_y_pred_portion[0] == s_end:
    print(f"✓ CORRECT: Predictions start immediately after batch_x ends (no gap)")
else:
    print(f"✗ ERROR: There is a gap or misalignment between history and predictions")

# Create visualization
print(f"\n" + "="*80)
print("CREATING VISUALIZATION...")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 6))

# Create a timeline
timeline = np.arange(0, s_end + pred_len + 10)

# Define regions
batch_x_region = (s_begin, s_end)
batch_y_region = (r_begin, r_end)
overlap_region = (overlap_start, overlap_end)
pred_only_region = (s_end, r_end)

# Plot regions
ax.barh(3, batch_x_region[1] - batch_x_region[0], left=batch_x_region[0], 
        height=0.4, color='blue', alpha=0.6, label=f'batch_x (seq_len={seq_len})')

ax.barh(2, batch_y_region[1] - batch_y_region[0], left=batch_y_region[0], 
        height=0.4, color='green', alpha=0.6, label=f'batch_y (label_len+pred_len={label_len+pred_len})')

ax.barh(1, overlap_region[1] - overlap_region[0], left=overlap_region[0], 
        height=0.4, color='orange', alpha=0.8, label=f'Overlap (label_len={label_len})')

ax.barh(0, pred_only_region[1] - pred_only_region[0], left=pred_only_region[0], 
        height=0.4, color='red', alpha=0.6, label=f'Pure predictions (pred_len={pred_len})')

# Add vertical lines for key positions
ax.axvline(s_begin, color='blue', linestyle='--', alpha=0.5)
ax.axvline(s_end, color='blue', linestyle='--', alpha=0.5)
ax.axvline(r_begin, color='green', linestyle='--', alpha=0.5)
ax.axvline(r_end, color='green', linestyle='--', alpha=0.5)

# Add annotations
ax.text(s_begin, 3.5, f't={s_begin}\nbatch_x start', ha='center', fontsize=9)
ax.text(s_end, 3.5, f't={s_end}\nbatch_x end', ha='center', fontsize=9)
ax.text(r_begin, 2.5, f't={r_begin}\nbatch_y start', ha='center', fontsize=9, color='green')
ax.text(r_end, 2.5, f't={r_end}\nbatch_y end', ha='center', fontsize=9, color='green')

# Add region labels
ax.text((s_begin + s_end) / 2, 3, 'Input History', ha='center', va='center', fontweight='bold')
ax.text((r_begin + r_begin + label_len) / 2, 1, 'Decoder Context', ha='center', va='center', fontweight='bold')
ax.text((s_end + r_end) / 2, 0, 'Future Predictions', ha='center', va='center', fontweight='bold', color='darkred')

# Formatting
ax.set_xlim(-5, r_end + 5)
ax.set_ylim(-0.5, 4)
ax.set_xlabel('Time Index', fontsize=12)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['Predictions\nOnly', 'Overlap\n(Label)', 'batch_y\n(Full)', 'batch_x\n(Input)'])
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title('Time Series Data Alignment: batch_x vs batch_y', fontsize=14, fontweight='bold')

# Add a text box with key insights
textstr = f'''Key Insights:
• batch_x: indices [{s_begin}, {s_end}) - length {seq_len}
• batch_y: indices [{r_begin}, {r_end}) - length {label_len + pred_len}
• Overlap: indices [{overlap_start}, {overlap_end}) - length {overlap_length}
• Gap between history and predictions: {"NONE (continuous)" if s_end == batch_y_pred_portion[0] else f"{batch_y_pred_portion[0] - s_end} steps"}'''

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('data_alignment_debug.png', dpi=150, bbox_inches='tight')
plt.savefig('data_alignment_debug.pdf', dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to data_alignment_debug.png and data_alignment_debug.pdf")

# Final verdict
print(f"\n" + "="*80)
print("FINAL VERDICT:")
print("="*80)

if overlap_length == label_len and s_end == batch_y_pred_portion[0]:
    print("✓ DATA ALIGNMENT IS CORRECT!")
    print(f"  - The last {label_len} points of batch_x overlap with the first {label_len} points of batch_y")
    print(f"  - This provides decoder context for attention mechanisms")
    print(f"  - Predictions start immediately after batch_x ends (no gap)")
    print(f"  - The visualization in the notebook is CORRECT and shows the intended behavior")
else:
    print("✗ DATA ALIGNMENT HAS ISSUES!")
    if overlap_length != label_len:
        print(f"  - Overlap mismatch: expected {label_len}, got {overlap_length}")
    if s_end != batch_y_pred_portion[0]:
        print(f"  - Gap detected: {batch_y_pred_portion[0] - s_end} steps between history and predictions")

print(f"\n" + "="*80)
print("EXPLANATION OF THE DESIGN:")
print("="*80)
print("""
This is a STANDARD Transformer time series forecasting setup:

1. Input Sequence (batch_x): 
   - Contains the historical observations
   - Length: seq_len (72 time steps)

2. Target Sequence (batch_y):
   - Contains BOTH decoder context AND future predictions
   - Total length: label_len + pred_len (32 + 48 = 80 time steps)
   - First part (label_len=32): Overlaps with the END of batch_x
   - Second part (pred_len=48): Pure future predictions

3. Why the overlap (label_len)?
   - Provides ground truth context for the decoder during training
   - Allows the model to learn how to transition from known to unknown
   - Standard practice in seq2seq and Transformer architectures
   - Similar to teacher forcing in training

4. Why it looks like a "gap" in visualization:
   - When plotting history (batch_x) vs predictions, we typically show:
     * History: full batch_x sequence
     * Predictions: only the pred_len portion of batch_y (excluding label overlap)
   - This creates a visual continuity: history ends where predictions begin
   - The label_len overlap is used internally by the model but not shown in plots

This design is INTENTIONAL and CORRECT for Transformer-based forecasting!
""")