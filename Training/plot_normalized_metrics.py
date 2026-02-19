"""
Generate normalized error metrics bar chart for model comparison (with feature selection).
Uses CV test metrics from the training pipeline.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# CV test metrics from training results (with feature selection)
# These are the average fold test metrics from the heatmap
models = ['Cubist', 'GBM', 'MARS', 'RF', 'SVM', 'XGB']

metrics = {
    'RMSE':  [7.038, 7.787, 9.344, 7.768, 8.540, 7.809],
    'MSE':   [51.518, 63.172, 87.760, 61.658, 74.837, 62.907],
    'MAE':   [4.661, 5.049, 6.695, 5.210, 5.124, 5.284],
    'MAPE':  [18.89, 46.56, 85.54, 27.11, 39.26, 43.54],
}

# Normalize each metric to 0-100 scale
normalized = {}
for metric, values in metrics.items():
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        normalized[metric] = [0] * len(values)
    else:
        normalized[metric] = [(v - min_val) / (max_val - min_val) * 100 for v in values]

# Plot setup
x = np.arange(len(models))
width = 0.18
metric_names = list(metrics.keys())
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

fig, ax = plt.subplots(figsize=(12, 7))

for i, (metric, color) in enumerate(zip(metric_names, colors)):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, normalized[metric], width, label=metric, color=color)

# Dotted lines for minimum (best) value per metric
line_styles = ['--', '--', '--', '--']
for i, (metric, color) in enumerate(zip(metric_names, colors)):
    min_norm = min(normalized[metric])
    ax.axhline(y=min_norm, color=color, linestyle='--', linewidth=1, alpha=0.6)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Normalized Metric Value', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 105)
ax.legend(title='Metric', loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()

plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(plot_dir, exist_ok=True)
out_path = os.path.join(plot_dir, 'normalized_error_metrics_with_fs.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
