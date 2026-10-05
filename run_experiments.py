"""
Main entry point for the Wi-Fi CSI Activity Recognition Project.

This script runs the full experiment pipeline:
  1. Baseline training on source environment (A1)
  2. Cross-environment evaluation (zero-shot on A2, A3)
  3. Fine-tuning with limited target data
  4. Training from scratch (control)
  5. Generates all plots and summary tables

Usage:
    python run_experiments.py                # Run full pipeline
    python run_experiments.py --baseline     # Run only baseline
    python run_experiments.py --cross-env    # Run only cross-env (requires baseline)
    python run_experiments.py --quick        # Quick run (fewer seeds/draws)
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import ExperimentConfig
from src.experiments import (
    run_baseline_experiment,
    run_cross_environment_experiment,
    run_finetuning_experiment,
    run_scratch_experiment,
    run_full_pipeline,
)
from src.trainer import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Wi-Fi CSI Activity Recognition Experiments'
    )
    parser.add_argument(
        '--baseline', action='store_true',
        help='Run only the baseline experiment'
    )
    parser.add_argument(
        '--cross-env', action='store_true',
        help='Run only the cross-environment experiment'
    )
    parser.add_argument(
        '--finetune', type=str, default=None,
        choices=['A2', 'A3'],
        help='Run fine-tuning for a specific target environment'
    )
    parser.add_argument(
        '--scratch', type=str, default=None,
        choices=['A2', 'A3'],
        help='Run from-scratch for a specific target environment'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick run with fewer seeds (2) and draws (2) and fewer shots [5, 20, 50]'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override number of training epochs'
    )
    parser.add_argument(
        '--no-cuda', action='store_true',
        help='Force CPU even if CUDA is available'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build configuration
    config = ExperimentConfig()
    config.seed = args.seed

    if args.no_cuda:
        config.device = 'cpu'

    if args.epochs:
        config.train.num_epochs = args.epochs

    if args.quick:
        config.finetune.num_seeds = 2
        config.finetune.num_sample_draws = 2
        config.finetune.shots_per_class = [5, 20, 50]
        config.train.num_epochs = min(config.train.num_epochs, 100)
        config.finetune.ft_epochs = 50
        print("*** QUICK MODE: reduced seeds, draws, and shots ***\n")

    # Determine what to run
    if args.baseline:
        result = run_baseline_experiment(config)
        print(f"\nBaseline accuracy: {result['accuracy']:.4f}")

    elif args.cross_env:
        model_path = str(config.models_dir / 'baseline_A1.pt')
        result = run_cross_environment_experiment(config, model_path)
        for env, r in result.items():
            print(f"  {env}: {r['accuracy']:.4f}")

    elif args.finetune:
        model_path = str(config.models_dir / 'baseline_A1.pt')
        result = run_finetuning_experiment(config, model_path, args.finetune)

    elif args.scratch:
        result = run_scratch_experiment(config, args.scratch)

    else:
        # Run full pipeline
        result = run_full_pipeline(config)

    print("\nDone!")


if __name__ == '__main__':
    main()
