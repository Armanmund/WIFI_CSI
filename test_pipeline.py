"""
Smoke test: verifies the entire pipeline works end-to-end with synthetic data.
Creates small synthetic .mat files in an isolated sandbox, then runs
a minimal version of all 4 experiments without polluting production assets.
"""

import sys
import os
import tempfile
import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure writable matplotlib cache in sandbox environments
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')


def create_synthetic_dataset(base_path: Path):
    """Create small synthetic .mat files inside an isolated directory."""
    base = base_path / "few_shot_datasets" / "m1c4_PCA_80_300_extracted_3x4"
    
    activities = ['walk', 'empty', 'jump', 'stand']
    envs = {'train_A1': 40, 'test_A2': 30, 'test_A3': 30}  # samples per activity
    
    for env, n_samples in envs.items():
        for activity in activities:
            dir_path = base / env / activity
            dir_path.mkdir(parents=True, exist_ok=True)
            
            for i in range(n_samples):
                seed = hash(f"{env}_{activity}_{i}") % (2**32)
                rng = np.random.RandomState(seed)
                
                # Make different activities have distinguishable signal offsets
                act_offset = activities.index(activity) * 0.5
                data = rng.randn(80, 300) + act_offset
                
                filepath = dir_path / f"sample_{i:03d}.mat"
                sio.savemat(str(filepath), {'cfm_data': data})
    
    return str(base)


def run_smoke_test(keep_sandbox: bool = False):
    """Run minimal pipeline in an isolated sandbox to verify everything works."""
    from configs.config import ExperimentConfig
    import src.experiments
    from src.experiments import (
        run_baseline_experiment,
        run_cross_environment_experiment,
        run_finetuning_experiment,
        run_scratch_experiment,
    )
    
    # Save original BASE_DIR
    orig_base_dir = src.experiments.BASE_DIR
    
    # Prepare isolated sandbox directory
    temp_dir_ctx = None
    if keep_sandbox:
        sandbox_path = PROJECT_ROOT / ".test_sandbox"
        sandbox_path.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir_ctx = tempfile.TemporaryDirectory()
        sandbox_path = Path(temp_dir_ctx.name)
        
    try:
        print("\n" + "=" * 60)
        print("  SMOKE TEST: Full Pipeline (Isolated Sandbox)")
        print(f"  Sandbox path: {sandbox_path}")
        print("=" * 60)
        
        # Route experiments to sandbox
        src.experiments.BASE_DIR = sandbox_path
        
        # 1. Create synthetic data in sandbox
        create_synthetic_dataset(sandbox_path)
        
        # 2. Configure minimal test parameters directed to sandbox
        config = ExperimentConfig()
        config.results_dir = sandbox_path / 'results'
        config.models_dir = sandbox_path / 'models'
        config.train.num_epochs = 5
        config.train.early_stopping_patience = 3
        config.finetune.shots_per_class = [5, 10]
        config.finetune.num_seeds = 1
        config.finetune.num_sample_draws = 1
        config.finetune.ft_epochs = 5
        config.finetune.ft_patience = 3
        
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
        print("  [OK] ALL SMOKE TESTS PASSED CLEANLY (Zero workspace side-effects)")
        print("=" * 60)
        
    finally:
        # Restore module state
        src.experiments.BASE_DIR = orig_base_dir
        if temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()
            print("  Cleaned up temporary sandbox directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Isolated smoke test for Wi-Fi CSI pipeline")
    parser.add_argument("--keep", action="store_true", help="Keep sandbox output in .test_sandbox for inspection")
    args = parser.parse_args()
    run_smoke_test(keep_sandbox=args.keep)
