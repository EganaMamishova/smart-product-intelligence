import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curves(history, metric_name, save_path):
    """
    Plots training and validation learning curves for a given metric.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history[metric_name], label=f'Train {metric_name}')
    if f'val_{metric_name}' in history.history:
        plt.plot(history.history[f'val_{metric_name}'], label=f'Val {metric_name}')
    plt.title(f'Model Training {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
