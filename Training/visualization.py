"""
Model performance visualization: bar charts and heatmaps.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_model_performance(results, with_fs):
    """Create bar chart visualizations for model performance and save to plots/."""
    performance_data = []

    for model_type, model_data in results.items():
        for model_name, metrics in model_data.items():
            avg_metrics = metrics.get('avg_metrics', {})
            performance_data.append({
                'Model Type': model_type,
                'Model': model_name,
                'R2': avg_metrics.get('R2', np.nan),
                'RMSE': avg_metrics.get('RMSE', np.nan),
                'MSE': avg_metrics.get('MSE', np.nan),
                'MAE': avg_metrics.get('MAE', np.nan),
                'MAPE': avg_metrics.get('MAPE', np.nan)
            })

    df_performance = pd.DataFrame(performance_data)

    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    if 'regression' in results:
        regression_data = df_performance[df_performance['Model Type'] == 'regression']
        suffix = "with_fs" if with_fs else "without_fs"

        for metric in ['R2', 'RMSE', 'MSE', 'MAE', 'MAPE']:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Model', y=metric, data=regression_data)
            plt.title(f'{metric} Comparison - Regression Models', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'regression_{metric.lower()}_comparison_{suffix}.png'), dpi=300)
            plt.close()

    # Heatmap of normalized metrics
    heatmap_data = df_performance.set_index('Model').drop(columns='Model Type')
    existing_metrics = [col for col in ['R2', 'RMSE', 'MSE', 'MAE', 'MAPE'] if col in heatmap_data.columns]
    heatmap_data = heatmap_data[existing_metrics]

    normalized_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    normalized_data.columns.name = "Metrics"

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        normalized_data,
        annot=heatmap_data,
        cmap='YlGnBu',
        fmt='.3f',
        cbar_kws={'label': 'Normalized Value'}
    )
    plt.title('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'model_performance_heatmap.png'), dpi=300)
    plt.close()

    # Save CV test metrics to CSV
    csv_path = os.path.join(plot_dir, 'cv_test_metrics.csv')
    df_performance.to_csv(csv_path, index=False)
    print(f"CV test metrics saved: {csv_path}")

    # Normalized grouped bar chart (error metrics only)
    if 'regression' in results:
        regression_data = df_performance[df_performance['Model Type'] == 'regression'].copy()
        suffix = "with_fs" if with_fs else "without_fs"
        error_metrics = ['RMSE', 'MSE', 'MAE', 'MAPE']
        models = regression_data['Model'].tolist()

        norm_data = {}
        for metric in error_metrics:
            vals = regression_data[metric].values
            min_v, max_v = vals.min(), vals.max()
            if max_v == min_v:
                norm_data[metric] = np.zeros(len(vals))
            else:
                norm_data[metric] = (vals - min_v) / (max_v - min_v) * 100

        x = np.arange(len(models))
        width = 0.18
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

        fig, ax = plt.subplots(figsize=(12, 7))
        for i, (metric, color) in enumerate(zip(error_metrics, colors)):
            offset = (i - 1.5) * width
            ax.bar(x + offset, norm_data[metric], width, label=metric, color=color)
            min_norm = norm_data[metric].min()
            ax.axhline(y=min_norm, color=color, linestyle='--', linewidth=1, alpha=0.6)

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Normalized Metric Value', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylim(0, 105)
        ax.legend(title='Metric', loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'normalized_error_metrics_{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return df_performance
