"""
Experiment orchestrator for Wi-Fi CSI Activity Recognition.

This module runs all four experiments:
  1. Baseline: train and evaluate within source environment (A1)
  2. Cross-environment: test baseline model on unseen environments (A2, A3)
  3. Fine-tuning: adapt pre-trained model with limited target data
  4. From-scratch: train new model with limited target data (control)

It also generates all plots, tables, and saves results.
"""

import os
import sys
import time
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import ExperimentConfig, BASE_DIR
from src.data_loader import (
    load_environment_data, create_dataloaders, CSIDataset, sample_k_shot
)
from src.model import CSIActivityCNN, build_model, save_model, load_model
from src.trainer import Trainer, set_seed, compute_metrics
from src.visualisation import (
    plot_training_curves, plot_confusion_matrix,
    plot_cross_environment_comparison, plot_finetuning_curve,
    plot_per_class_accuracy, plot_accuracy_recovery_curve
)
from src.utils import ResultsAggregator, save_results, ensure_dir, NumpyEncoder
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Baseline — Within-environment training and evaluation
# ═══════════════════════════════════════════════════════════════════

def run_baseline_experiment(config):
    """
    Train and evaluate the CNN within the source environment (A1).

    Steps:
        1. Load all A1 data
        2. Split into train (85%) / validation (15%)
        3. Train CNN with early stopping for up to 200 epochs
        4. Evaluate on A1 validation set
        5. Save model, training curves, and confusion matrix

    Returns:
        dict with keys: accuracy, model_path, history, per_class_accuracy
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Baseline Training on Source Environment (A1)")
    print("=" * 70)

    set_seed(config.seed)
    device = config.device

    # --- Load data ---
    print(f"\n[1/4] Loading {config.data.source_env} data...")
    X, y = load_environment_data(
        BASE_DIR, config.data.source_env, config.data.data_folder,
        config.data.label_map
    )
    if len(X) == 0:
        raise RuntimeError(f"No data loaded for {config.data.source_env}. Check dataset path.")

    print(f"  Samples: {len(X)}, Shape: {X.shape[1:]}")
    for cls_name, cls_id in config.data.label_map.items():
        n = np.sum(y == cls_id)
        print(f"    {cls_name}: {n} samples")

    # --- Create train/val split ---
    print(f"\n[2/4] Creating train/val split (val={config.train.validation_split:.0%})...")
    train_loader, val_loader, idx_train, idx_val = create_dataloaders(
        X, y,
        batch_size=config.train.batch_size,
        val_split=config.train.validation_split,
        seed=config.seed
    )
    print(f"  Train: {len(idx_train)} samples, Val: {len(idx_val)} samples")

    # --- Build & train model ---
    print(f"\n[3/4] Building and training model...")
    model = build_model(num_classes=config.data.num_classes, device=device)

    trainer_config = {
        'lr': config.train.learning_rate,
        'weight_decay': config.train.weight_decay,
        'step_size': config.train.step_size,
        'gamma': config.train.gamma,
    }
    trainer = Trainer(model, device, trainer_config)

    model_path = str(config.models_dir / 'baseline_A1.pt')
    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=config.train.num_epochs,
        patience=config.train.early_stopping_patience,
        model_save_path=model_path
    )

    # --- Evaluate ---
    print(f"\n[4/4] Evaluating on A1 validation set...")
    class_names = list(config.data.label_map.keys())
    metrics = trainer.evaluate(val_loader, class_names=class_names)

    print(f"\n  Baseline Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"\n  Per-class accuracy:")
    for cls, acc in metrics['per_class_accuracy'].items():
        print(f"    {cls}: {acc:.4f}")
    print(f"\n  Classification Report:\n{metrics['classification_report_str']}")

    # --- Save plots ---
    plot_dir = str(config.results_dir / 'plots' / 'baseline')
    ensure_dir(plot_dir)

    # Adapt history keys for plotting
    plot_history = {
        'loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'accuracy': history['train_acc'],
        'val_accuracy': history['val_acc'],
    }
    plot_training_curves(
        plot_history,
        title='Baseline Training on A1',
        save_path=os.path.join(plot_dir, 'training_curves')
    )

    plot_confusion_matrix(
        metrics['confusion_matrix'], class_names,
        title='Baseline Confusion Matrix (A1 Val)',
        save_path=os.path.join(plot_dir, 'confusion_matrix'),
        normalize=True
    )
    plt.close('all')

    result = {
        'accuracy': float(metrics['accuracy']),
        'model_path': model_path,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'per_class_accuracy': {k: float(v) for k, v in metrics['per_class_accuracy'].items()},
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
    }

    save_results(result, str(config.results_dir / 'baseline_results.json'))
    print(f"\n[OK] Baseline experiment complete. Model saved to {model_path}")
    return result


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Cross-environment — Zero-shot transfer
# ═══════════════════════════════════════════════════════════════════

def run_cross_environment_experiment(config, model_path):
    """
    Test the A1-trained model directly on unseen environments (A2, A3)
    without any adaptation (zero-shot transfer).

    Returns:
        dict mapping env_name -> {accuracy, per_class_accuracy, confusion_matrix, accuracy_drop}
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Cross-Environment Evaluation (Zero-Shot Transfer)")
    print("=" * 70)

    set_seed(config.seed)
    device = config.device

    # Load baseline model
    print(f"\nLoading baseline model from {model_path}...")
    model = load_model(model_path, num_classes=config.data.num_classes, device=device)
    model.eval()

    class_names = list(config.data.label_map.keys())
    trainer_config = {'lr': 1e-3, 'weight_decay': 1e-4, 'step_size': 50, 'gamma': 0.5}
    trainer = Trainer(model, device, trainer_config)

    results = {}
    plot_dir = str(config.results_dir / 'plots' / 'cross_env')
    ensure_dir(plot_dir)

    for env in config.data.target_envs:
        print(f"\n--- Evaluating on {env} ---")
        X, y = load_environment_data(
            BASE_DIR, env, config.data.data_folder, config.data.label_map
        )
        if len(X) == 0:
            print(f"  Warning: No data for {env}, skipping.")
            continue

        dataset = CSIDataset(X, y)
        test_loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=False)

        metrics = trainer.evaluate(test_loader, class_names=class_names)

        print(f"  Accuracy on {env}: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        print(f"  Per-class accuracy:")
        for cls, acc in metrics['per_class_accuracy'].items():
            print(f"    {cls}: {acc:.4f}")

        plot_confusion_matrix(
            metrics['confusion_matrix'], class_names,
            title=f'Zero-Shot on {env} (trained A1)',
            save_path=os.path.join(plot_dir, f'confusion_matrix_{env}'),
            normalize=True
        )
        plt.close('all')

        results[env] = {
            'accuracy': float(metrics['accuracy']),
            'per_class_accuracy': {k: float(v) for k, v in metrics['per_class_accuracy'].items()},
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
        }

    save_results(results, str(config.results_dir / 'cross_env_results.json'))
    print(f"\n[OK] Cross-environment experiment complete.")
    return results


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Fine-tuning with limited target data
# ═══════════════════════════════════════════════════════════════════

def run_finetuning_experiment(config, model_path, target_env):
    """
    Fine-tune the pre-trained model using k shots per class from the target
    environment.

    For each k in shots_per_class:
        For each seed in range(num_seeds):
            For each draw in range(num_sample_draws):
                - Sample k examples per class
                - Split into mini-train (80%) / mini-val (20%)
                - Load pre-trained model
                - Fine-tune with small lr
                - Evaluate on REMAINING target data
                - Record accuracy

    Returns:
        dict mapping k -> {mean_acc, std_acc, ci_lower, ci_upper, all_runs}
    """
    print(f"\n" + "=" * 70)
    print(f"  EXPERIMENT 3: Fine-Tuning on {target_env}")
    print("=" * 70)

    device = config.device
    class_names = list(config.data.label_map.keys())

    # Load full target data
    print(f"\nLoading full {target_env} data...")
    X_full, y_full = load_environment_data(
        BASE_DIR, target_env, config.data.data_folder, config.data.label_map
    )
    if len(X_full) == 0:
        raise RuntimeError(f"No data for {target_env}")

    results = {}
    total_configs = len(config.finetune.shots_per_class)

    for idx, k in enumerate(config.finetune.shots_per_class):
        print(f"\n--- k={k} shots/class ({idx+1}/{total_configs}) ---")
        k_accuracies = []

        for seed_idx in range(config.finetune.num_seeds):
            for draw_idx in range(config.finetune.num_sample_draws):
                run_seed = config.seed + seed_idx * 1000 + draw_idx

                try:
                    set_seed(run_seed)

                    # Sample k per class
                    X_sampled, y_sampled = sample_k_shot(
                        X_full, y_full, k, config.data.num_classes, seed=run_seed
                    )

                    # Identify remaining samples for testing
                    sampled_indices = set()
                    for c in range(config.data.num_classes):
                        c_indices = np.where(y_full == c)[0]
                        np.random.seed(run_seed)
                        selected = np.random.choice(c_indices, min(k, len(c_indices)), replace=False)
                        sampled_indices.update(selected.tolist())

                    remaining_mask = np.ones(len(y_full), dtype=bool)
                    remaining_mask[list(sampled_indices)] = False
                    X_remaining = X_full[remaining_mask]
                    y_remaining = y_full[remaining_mask]

                    # If too few samples for train/val split, use all for training
                    n_sampled = len(y_sampled)
                    if n_sampled < 8:  # Too few for meaningful split
                        train_dataset = CSIDataset(X_sampled, y_sampled)
                        train_loader = DataLoader(train_dataset, batch_size=max(1, n_sampled), shuffle=True)
                        val_loader = train_loader  # Use same for validation in very low data
                    else:
                        val_frac = 0.2
                        train_loader, val_loader, _, _ = create_dataloaders(
                            X_sampled, y_sampled,
                            batch_size=min(config.train.batch_size, n_sampled),
                            val_split=val_frac,
                            seed=run_seed
                        )

                    # Test loader from remaining data
                    test_dataset = CSIDataset(X_remaining, y_remaining)
                    test_loader = DataLoader(
                        test_dataset, batch_size=config.train.batch_size, shuffle=False
                    )

                    # Load pre-trained model
                    model = load_model(
                        model_path, num_classes=config.data.num_classes, device=device
                    )

                    # Fine-tune
                    ft_config = {
                        'lr': config.finetune.ft_lr,
                        'weight_decay': config.finetune.ft_weight_decay,
                        'step_size': 30,
                        'gamma': 0.5,
                    }
                    trainer = Trainer(model, device, ft_config)
                    trainer.fine_tune(
                        train_loader, val_loader,
                        num_epochs=config.finetune.ft_epochs,
                        lr=config.finetune.ft_lr,
                        patience=config.finetune.ft_patience,
                        freeze_features=config.finetune.freeze_features
                    )

                    # Evaluate on remaining data
                    metrics = trainer.evaluate(test_loader, class_names=class_names)
                    k_accuracies.append(metrics['accuracy'])

                except Exception as e:
                    print(f"    Error (seed={seed_idx}, draw={draw_idx}): {e}")
                    import traceback
                    traceback.print_exc()

        if k_accuracies:
            mean_acc = float(np.mean(k_accuracies))
            std_acc = float(np.std(k_accuracies))
            n = len(k_accuracies)
            ci = 1.96 * std_acc / np.sqrt(n) if n > 1 else 0.0

            results[k] = {
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'ci_lower': mean_acc - ci,
                'ci_upper': mean_acc + ci,
                'all_runs': [float(a) for a in k_accuracies],
                'n_runs': n,
            }
            print(f"  k={k}: acc = {mean_acc:.4f} ± {std_acc:.4f} (95% CI: [{mean_acc-ci:.4f}, {mean_acc+ci:.4f}])")

    save_results(
        {str(k): v for k, v in results.items()},
        str(config.results_dir / f'finetuning_{target_env}_results.json')
    )
    print(f"\n[OK] Fine-tuning experiment on {target_env} complete.")
    return results


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Train from scratch (control)
# ═══════════════════════════════════════════════════════════════════

def run_scratch_experiment(config, target_env):
    """
    Train a randomly-initialized model with the same limited target data.
    This is the control experiment that isolates the benefit of pre-training.

    Same loop structure as fine-tuning but model starts from random weights.

    Returns:
        dict mapping k -> {mean_acc, std_acc, ci_lower, ci_upper, all_runs}
    """
    print(f"\n" + "=" * 70)
    print(f"  EXPERIMENT 4: Train from Scratch on {target_env} (Control)")
    print("=" * 70)

    device = config.device
    class_names = list(config.data.label_map.keys())

    # Load full target data
    print(f"\nLoading full {target_env} data...")
    X_full, y_full = load_environment_data(
        BASE_DIR, target_env, config.data.data_folder, config.data.label_map
    )
    if len(X_full) == 0:
        raise RuntimeError(f"No data for {target_env}")

    results = {}
    total_configs = len(config.finetune.shots_per_class)

    for idx, k in enumerate(config.finetune.shots_per_class):
        print(f"\n--- k={k} shots/class ({idx+1}/{total_configs}) ---")
        k_accuracies = []

        for seed_idx in range(config.finetune.num_seeds):
            for draw_idx in range(config.finetune.num_sample_draws):
                run_seed = config.seed + seed_idx * 1000 + draw_idx

                try:
                    set_seed(run_seed)

                    # Sample k per class (same sampling as fine-tuning)
                    X_sampled, y_sampled = sample_k_shot(
                        X_full, y_full, k, config.data.num_classes, seed=run_seed
                    )

                    # Identify remaining for testing
                    sampled_indices = set()
                    for c in range(config.data.num_classes):
                        c_indices = np.where(y_full == c)[0]
                        np.random.seed(run_seed)
                        selected = np.random.choice(c_indices, min(k, len(c_indices)), replace=False)
                        sampled_indices.update(selected.tolist())

                    remaining_mask = np.ones(len(y_full), dtype=bool)
                    remaining_mask[list(sampled_indices)] = False
                    X_remaining = X_full[remaining_mask]
                    y_remaining = y_full[remaining_mask]

                    n_sampled = len(y_sampled)
                    if n_sampled < 8:
                        train_dataset = CSIDataset(X_sampled, y_sampled)
                        train_loader = DataLoader(train_dataset, batch_size=max(1, n_sampled), shuffle=True)
                        val_loader = train_loader
                    else:
                        val_frac = 0.2
                        train_loader, val_loader, _, _ = create_dataloaders(
                            X_sampled, y_sampled,
                            batch_size=min(config.train.batch_size, n_sampled),
                            val_split=val_frac,
                            seed=run_seed
                        )

                    test_dataset = CSIDataset(X_remaining, y_remaining)
                    test_loader = DataLoader(
                        test_dataset, batch_size=config.train.batch_size, shuffle=False
                    )

                    # Build NEW random model (no pre-training)
                    model = build_model(num_classes=config.data.num_classes, device=device)

                    scratch_config = {
                        'lr': config.finetune.ft_lr,
                        'weight_decay': config.finetune.ft_weight_decay,
                        'step_size': 30,
                        'gamma': 0.5,
                    }
                    trainer = Trainer(model, device, scratch_config)
                    trainer.fit(
                        train_loader, val_loader,
                        num_epochs=config.finetune.ft_epochs,
                        patience=config.finetune.ft_patience
                    )

                    # Evaluate
                    metrics = trainer.evaluate(test_loader, class_names=class_names)
                    k_accuracies.append(metrics['accuracy'])

                except Exception as e:
                    print(f"    Error (seed={seed_idx}, draw={draw_idx}): {e}")
                    import traceback
                    traceback.print_exc()

        if k_accuracies:
            mean_acc = float(np.mean(k_accuracies))
            std_acc = float(np.std(k_accuracies))
            n = len(k_accuracies)
            ci = 1.96 * std_acc / np.sqrt(n) if n > 1 else 0.0

            results[k] = {
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'ci_lower': mean_acc - ci,
                'ci_upper': mean_acc + ci,
                'all_runs': [float(a) for a in k_accuracies],
                'n_runs': n,
            }
            print(f"  k={k}: acc = {mean_acc:.4f} ± {std_acc:.4f}")

    save_results(
        {str(k): v for k, v in results.items()},
        str(config.results_dir / f'scratch_{target_env}_results.json')
    )
    print(f"\n[OK] From-scratch experiment on {target_env} complete.")
    return results


# ═══════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════

def generate_summary_plots(config, all_results):
    """Generate all summary plots from collected results."""
    plot_dir = str(config.results_dir / 'plots' / 'summary')
    ensure_dir(plot_dir)
    class_names = list(config.data.label_map.keys())

    baseline_acc = all_results['baseline']['accuracy']

    # --- Cross-environment comparison bar chart ---
    env_accs = {'A1 (baseline)': baseline_acc}
    for env, res in all_results.get('cross_env', {}).items():
        env_accs[f'{env} (zero-shot)'] = res['accuracy']
    plot_cross_environment_comparison(
        env_accs,
        save_path=os.path.join(plot_dir, 'cross_env_comparison')
    )

    # --- Fine-tuning curves (THE key figure) ---
    for env in config.data.target_envs:
        if env not in all_results.get('finetuning', {}):
            continue

        ft_results = all_results['finetuning'][env]
        scratch_results = all_results.get('scratch', {}).get(env, {})
        zero_shot_acc = all_results.get('cross_env', {}).get(env, {}).get('accuracy', 0.25)

        shots_list = sorted([int(k) for k in ft_results.keys()])

        # Build accuracy arrays: shape (n_shots, n_runs)
        ft_accs = []
        scratch_accs = []
        for k in shots_list:
            ft_accs.append(ft_results[k]['all_runs'])
            if k in scratch_results:
                scratch_accs.append(scratch_results[k]['all_runs'])

        # Pad to equal length per shot count
        max_runs = max(len(a) for a in ft_accs) if ft_accs else 1
        ft_array = np.array([
            a + [np.nan] * (max_runs - len(a)) for a in ft_accs
        ])

        accuracies_dict = {'Fine-tuned (pre-trained)': ft_array}

        if scratch_accs:
            max_runs_s = max(len(a) for a in scratch_accs) if scratch_accs else 1
            scratch_array = np.array([
                a + [np.nan] * (max_runs_s - len(a)) for a in scratch_accs
            ])
            accuracies_dict['Trained from scratch'] = scratch_array

        plot_finetuning_curve(
            shots_list, accuracies_dict,
            baseline_acc=baseline_acc,
            zero_shot_acc=zero_shot_acc,
            save_path=os.path.join(plot_dir, f'finetuning_curve_{env}'),
            title=f'Fine-tuning Curve — Target: {env}'
        )

        # --- Recovery curve ---
        ft_means = np.array([ft_results[k]['mean_acc'] for k in shots_list])
        gap = baseline_acc - zero_shot_acc
        if gap > 0:
            recovery = (ft_means - zero_shot_acc) / gap
        else:
            recovery = np.ones_like(ft_means)

        plot_accuracy_recovery_curve(
            shots_list, recovery,
            save_path=os.path.join(plot_dir, f'recovery_curve_{env}')
        )

    plt.close('all')


def generate_summary_table(config, all_results):
    """Generate and print summary tables, and save as CSV."""
    table_dir = str(config.results_dir / 'tables')
    ensure_dir(table_dir)

    baseline_acc = all_results['baseline']['accuracy']

    # --- Main results table ---
    rows = []

    # Baseline row
    rows.append({
        'Experiment': 'Baseline (A1)',
        'Environment': 'A1',
        'Shots/Class': 'Full',
        'Accuracy (mean)': f"{baseline_acc:.4f}",
        'Accuracy (std)': '-',
        '95% CI': '-',
    })

    # Cross-env rows
    for env, res in all_results.get('cross_env', {}).items():
        acc_drop = baseline_acc - res['accuracy']
        rows.append({
            'Experiment': f'Zero-shot ({env})',
            'Environment': env,
            'Shots/Class': '0',
            'Accuracy (mean)': f"{res['accuracy']:.4f}",
            'Accuracy (std)': '-',
            '95% CI': '-',
            'Accuracy Drop': f"{acc_drop:.4f}",
        })

    # Fine-tuning rows
    for env in config.data.target_envs:
        if env not in all_results.get('finetuning', {}):
            continue
        for k, res in sorted(all_results['finetuning'][env].items(), key=lambda x: int(x[0])):
            rows.append({
                'Experiment': f'Fine-tuned ({env})',
                'Environment': env,
                'Shots/Class': str(k),
                'Accuracy (mean)': f"{res['mean_acc']:.4f}",
                'Accuracy (std)': f"{res['std_acc']:.4f}",
                '95% CI': f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]",
            })

    # Scratch rows
    for env in config.data.target_envs:
        if env not in all_results.get('scratch', {}):
            continue
        for k, res in sorted(all_results['scratch'][env].items(), key=lambda x: int(x[0])):
            rows.append({
                'Experiment': f'From scratch ({env})',
                'Environment': env,
                'Shots/Class': str(k),
                'Accuracy (mean)': f"{res['mean_acc']:.4f}",
                'Accuracy (std)': f"{res['std_acc']:.4f}",
                '95% CI': f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]",
            })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("  SUMMARY RESULTS TABLE")
    print("=" * 90)
    print(df.to_string(index=False))

    csv_path = os.path.join(table_dir, 'summary_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Table saved to {csv_path}")

    return df


def run_full_pipeline(config=None):
    """
    Runs the complete experiment pipeline:

    a) Baseline experiment (train on A1)
    b) Cross-environment evaluation (test on A2, A3)
    c) Fine-tuning experiments (for each target env)
    d) From-scratch experiments (control, for each target env)
    e) Generate all plots
    f) Generate summary tables
    g) Save all results

    Returns:
        dict with all results
    """
    if config is None:
        config = ExperimentConfig()

    start_time = time.time()

    print("\n" + "=" * 70)
    print("  Wi-Fi CSI Activity Recognition — Full Experiment Pipeline")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {config.device}")
    print(f"  Source env: {config.data.source_env}")
    print(f"  Target envs: {config.data.target_envs}")
    print(f"  Activities: {config.data.activities}")
    print(f"  Seeds: {config.finetune.num_seeds}, Draws: {config.finetune.num_sample_draws}")
    print(f"  Shots per class: {config.finetune.shots_per_class}")
    print("=" * 70)

    all_results = {}

    # ─── (a) Baseline ─────────────────────────────────────────────
    baseline_result = run_baseline_experiment(config)
    all_results['baseline'] = baseline_result

    # ─── (b) Cross-environment ─────────────────────────────────────
    cross_env_result = run_cross_environment_experiment(
        config, baseline_result['model_path']
    )
    all_results['cross_env'] = cross_env_result

    # ─── (c) Fine-tuning ──────────────────────────────────────────
    all_results['finetuning'] = {}
    for env in config.data.target_envs:
        ft_result = run_finetuning_experiment(
            config, baseline_result['model_path'], env
        )
        all_results['finetuning'][env] = ft_result

    # ─── (d) From-scratch (control) ───────────────────────────────
    all_results['scratch'] = {}
    for env in config.data.target_envs:
        scratch_result = run_scratch_experiment(config, env)
        all_results['scratch'][env] = scratch_result

    # ─── (e) Generate plots ───────────────────────────────────────
    print("\n\nGenerating summary plots...")
    try:
        generate_summary_plots(config, all_results)
        print("  [OK] Plots saved to results/plots/summary/")
    except Exception as e:
        print(f"  Warning: Plot generation failed: {e}")
        import traceback
        traceback.print_exc()

    # ─── (f) Generate tables ──────────────────────────────────────
    try:
        generate_summary_table(config, all_results)
    except Exception as e:
        print(f"  Warning: Table generation failed: {e}")
        import traceback
        traceback.print_exc()

    # ─── (g) Save all results ─────────────────────────────────────
    save_results(
        all_results,
        str(config.results_dir / 'all_results.json')
    )

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {minutes}m {seconds}s")
    print(f"  Results saved to: {config.results_dir}")
    print("=" * 70)

    return all_results
