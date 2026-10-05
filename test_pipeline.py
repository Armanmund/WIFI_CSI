"""
Smoke test: verifies the entire pipeline works end-to-end with synthetic data.
Creates tiny fake .mat files mimicking the ReWiS dataset structure, then runs
a minimal version of all 4 experiments.
"""

import sys
import os
import numpy as np
import scipy.io as sio
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_synthetic_dataset():
    """Create small synthetic .mat files for testing."""
    base = PROJECT_ROOT / "few_shot_datasets" / "m1c4_PCA_80_300_extracted_3x4"
    
    activities = ['walk', 'empty', 'jump', 'stand']
    envs = {'train_A1': 40, 'test_A2': 30, 'test_A3': 30}  # samples per activity
    
    for env, n_samples in envs.items():
        for activity in activities:
            dir_path = base / env / activity
            dir_path.mkdir(parents=True, exist_ok=True)
            
            for i in range(n_samples):
                # Create synthetic CSI data with some class-specific patterns
                seed = hash(f"{env}_{activity}_{i}") % (2**32)
                rng = np.random.RandomState(seed)
                
                # Make different activities have different signal patterns
                act_offset = activities.index(activity) * 0.5
                data = rng.randn(80, 300) + act_offset
                
                filepath = dir_path / f"sample_{i:03d}.mat"
                sio.savemat(str(filepath), {'cfm_data': data})
    
    print(f"Created synthetic dataset at {base}")
    return str(base)


def run_smoke_test():
    """Run minimal pipeline to verify everything works."""
    from configs.config import ExperimentConfig
    from src.experiments import (
        run_baseline_experiment,
        run_cross_environment_experiment,
        run_finetuning_experiment,
        run_scratch_experiment,
    )
    
    # Create synthetic data
    create_synthetic_dataset()
    
    # Minimal config for fast testing
    config = ExperimentConfig()
    config.train.num_epochs = 5
    config.train.early_stopping_patience = 3
    config.finetune.shots_per_class = [5, 10]
    config.finetune.num_seeds = 1
    config.finetune.num_sample_draws = 1
    config.finetune.ft_epochs = 5
    config.finetune.ft_patience = 3
    
    print("\n" + "=" * 60)
    print("  SMOKE TEST: Full Pipeline with Synthetic Data")
    print("=" * 60)
    
    # Test 1: Baseline
    print("\n>>> Test 1: Baseline experiment...")
    baseline = run_baseline_experiment(config)
    print(f"  [OK] Baseline accuracy: {baseline['accuracy']:.4f}")
    assert 'accuracy' in baseline
    assert 'model_path' in baseline
    assert os.path.exists(baseline['model_path'])
    
    # Test 2: Cross-environment
    print("\n>>> Test 2: Cross-environment experiment...")
    cross_env = run_cross_environment_experiment(config, baseline['model_path'])
    for env, res in cross_env.items():
        print(f"  [OK] {env} accuracy: {res['accuracy']:.4f}")
        assert 'accuracy' in res
    
    # Test 3: Fine-tuning
    print("\n>>> Test 3: Fine-tuning experiment...")
    ft_result = run_finetuning_experiment(config, baseline['model_path'], 'A2')
    for k, res in ft_result.items():
        print(f"  [OK] k={k}: acc = {res['mean_acc']:.4f} ± {res['std_acc']:.4f}")
        assert 'mean_acc' in res
    
    # Test 4: From scratch
    print("\n>>> Test 4: From-scratch experiment...")
    scratch_result = run_scratch_experiment(config, 'A2')
    for k, res in scratch_result.items():
        print(f"  [OK] k={k}: acc = {res['mean_acc']:.4f} ± {res['std_acc']:.4f}")
        assert 'mean_acc' in res
    
    print("\n" + "=" * 60)
    print("  [OK] ALL SMOKE TESTS PASSED!")
    print("=" * 60)
    
    # Clean up synthetic data
    syn_path = PROJECT_ROOT / "few_shot_datasets" / "m1c4_PCA_80_300_extracted_3x4"
    # Don't delete — user might want to inspect
    print(f"\nSynthetic data at: {syn_path}")
    print("(You can delete this after downloading the real dataset)")


if __name__ == '__main__':
    run_smoke_test()
