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

        for metric in ['R2', 'RMSE', 'MSE', 'MAE']:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Model', y=metric, data=regression_data)
            plt.title(f'{metric} Comparison - Regression Models', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'regression_{metric.lower()}_comparison_{suffix}.png'), dpi=300)
            plt.close()

    # Heatmap of normalized metrics
    heatmap_data = df_performance.set_index('Model').drop(columns='Model Type')
    existing_metrics = [col for col in ['R2', 'RMSE', 'MSE', 'MAE'] if col in heatmap_data.columns]
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

    return df_performance
