import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple

# Set academic style for plots
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
sns.set_context("paper", font_scale=1.5)
sns.set_palette("colorblind")

def _save_figure(fig: plt.Figure, save_path: str):
    """Helper to save figure in both PNG and PDF formats."""
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        base_path, _ = os.path.splitext(save_path)
        fig.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{base_path}.pdf", bbox_inches='tight')

def plot_training_curves(history: Dict[str, List[float]], title: str, save_path: str) -> plt.Figure:
    """Plot training and validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['loss'], label='Train Loss', color='blue')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], label='Val Loss', color='red')
        best_epoch = np.argmin(history['val_loss']) + 1
        ax1.axvline(best_epoch, color='gray', linestyle='--', label=f'Best (Ep: {best_epoch})')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    
    # Accuracy plot
    if 'accuracy' in history:
        ax2.plot(epochs, history['accuracy'], label='Train Acc', color='blue')
        if 'val_accuracy' in history:
            ax2.plot(epochs, history['val_accuracy'], label='Val Acc', color='red')
            best_epoch_acc = np.argmax(history['val_accuracy']) + 1
            ax2.axvline(best_epoch_acc, color='gray', linestyle='--', label=f'Best (Ep: {best_epoch_acc})')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curve')
        ax2.legend()
        
    fig.suptitle(title)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, save_path: str, normalize: bool = True) -> plt.Figure:
    """Plot confusion matrix using seaborn heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    fmt = 'd'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig

def plot_cross_environment_comparison(results_dict: Dict[str, float], save_path: str, error_bars: Optional[Dict[str, float]] = None) -> plt.Figure:
    """Plot bar chart comparing accuracy across environments."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    envs = list(results_dict.keys())
    accuracies = [results_dict[env] for env in envs]
    
    yerr = [error_bars.get(env, 0) for env in envs] if error_bars else None
    
    bars = ax.bar(envs, accuracies, yerr=yerr, capsize=5, color=sns.color_palette("colorblind")[0])
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Cross-Environment Performance Comparison')
    ax.set_ylim(0, max(1.0, max(accuracies) * 1.1))
    
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig

def plot_finetuning_curve(shots_list: List[int], accuracies_dict: Dict[str, np.ndarray], 
                         baseline_acc: float, zero_shot_acc: float, save_path: str, title: str) -> plt.Figure:
    """Plot few-shot fine-tuning curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_xscale('log')
    
    # Plot baseline & zero-shot
    ax.axhline(baseline_acc, color='black', linestyle='--', label='Target Baseline (Upper Bound)')
    ax.axhline(zero_shot_acc, color='gray', linestyle=':', label='Zero-Shot (Lower Bound)')
    
    # Plot curves (assumes dict maps method name to 2D array: (shots_idx, runs))
    colors = sns.color_palette("colorblind")[1:]
    
    for (method, acc_data), color in zip(accuracies_dict.items(), colors):
        means = np.mean(acc_data, axis=1)
        stds = np.std(acc_data, axis=1)
        cis = 1.96 * stds / np.sqrt(acc_data.shape[1])
        
        ax.plot(shots_list, means, marker='o', label=method, color=color)
        ax.fill_between(shots_list, means - cis, means + cis, alpha=0.2, color=color)
        
        # Mark 90% recovery of baseline if applicable
        target_recovery = zero_shot_acc + 0.9 * (baseline_acc - zero_shot_acc)
        idx_above = np.where(means >= target_recovery)[0]
        if len(idx_above) > 0:
            rec_shot = shots_list[idx_above[0]]
            rec_acc = means[idx_above[0]]
            ax.scatter([rec_shot], [rec_acc], color='red', s=100, zorder=5, marker='*')
            ax.annotate('90% Recovery', (rec_shot, rec_acc), xytext=(10, -15), 
                        textcoords='offset points', color='red')

    ax.set_xlabel('Number of Shots per Class')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(shots_list)
    ax.set_xticklabels([str(s) for s in shots_list])
    ax.legend(loc='best')
    
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig

def plot_all_confusion_matrices(results: Dict[str, Dict[int, np.ndarray]], class_names: List[str], save_path: str) -> plt.Figure:
    """Grid of confusion matrices for multiple conditions."""
    envs = list(results.keys())
    shots = list(next(iter(results.values())).keys())
    
    fig, axes = plt.subplots(len(envs), len(shots), figsize=(5 * len(shots), 5 * len(envs)))
    if len(envs) == 1: axes = np.expand_dims(axes, 0)
    if len(shots) == 1: axes = np.expand_dims(axes, 1)
    
    for i, env in enumerate(envs):
        for j, shot in enumerate(shots):
            cm = results[env][shot]
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_norm, annot=False, cmap='Blues', cbar=False, ax=axes[i, j],
                        xticklabels=class_names, yticklabels=class_names)
            axes[i, j].set_title(f"{env} - {shot} Shots")
            if j == 0:
                axes[i, j].set_ylabel('True Label')
            if i == len(envs) - 1:
                axes[i, j].set_xlabel('Predicted Label')
                
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig

def plot_per_class_accuracy(results: Dict[str, List[float]], class_names: List[str], save_path: str) -> plt.Figure:
    """Grouped bar chart showing per-class accuracy for different conditions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    conditions = list(results.keys())
    x = np.arange(len(class_names))
    width = 0.8 / len(conditions)
    
    for i, condition in enumerate(conditions):
        accs = results[condition]
        offset = (i - len(conditions)/2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=condition)
        
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig

def plot_accuracy_recovery_curve(shots_list: List[int], recovery_fractions: np.ndarray, save_path: str) -> plt.Figure:
    """Plot fraction of baseline performance recovered vs shots."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_xscale('log')
    ax.plot(shots_list, recovery_fractions, marker='s', linewidth=2, color='purple')
    
    # Mark thresholds
    thresholds = [0.50, 0.75, 0.90, 0.95]
    for t in thresholds:
        ax.axhline(t, color='gray', linestyle='--', alpha=0.5)
        ax.text(shots_list[0], t, f' {int(t*100)}%', va='bottom', color='gray')
        
    ax.set_xlabel('Number of Shots per Class')
    ax.set_ylabel('Fraction of Baseline Performance Recovered')
    ax.set_title('Accuracy Recovery vs. Few-Shot Data')
    ax.set_xticks(shots_list)
    ax.set_xticklabels([str(s) for s in shots_list])
    ax.set_ylim(0, 1.05)
    
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig
