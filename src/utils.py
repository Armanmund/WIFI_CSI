import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import timedelta
import time

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Configures Python logging with console and optionally file handlers."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        ensure_dir(os.path.dirname(os.path.abspath(log_file)))
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def ensure_dir(path: str) -> None:
    """Creates directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Saves results dict as JSON with numpy types converted."""
    ensure_dir(os.path.dirname(os.path.abspath(filepath)))
    with open(filepath, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=4)

def load_results(filepath: str) -> Dict[str, Any]:
    """Loads results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def format_time(seconds: float) -> str:
    """Returns human readable time string from seconds."""
    return str(timedelta(seconds=int(seconds)))

def print_experiment_header(config) -> None:
    """Prints formatted experiment configuration. Accepts str or dict."""
    if isinstance(config, str):
        print("=" * 50)
        print(f" {config} ".center(50, "="))
        print("=" * 50)
    elif isinstance(config, dict):
        print("=" * 50)
        print(" EXPERIMENT CONFIGURATION ".center(50, "="))
        print("=" * 50)
        for k, v in config.items():
            print(f"{k:<20}: {v}")
        print("=" * 50)
    else:
        print("=" * 50)

class ResultsAggregator:
    """Aggregates results across multiple runs and computes statistics."""
    def __init__(self):
        # Maps config_name -> list of metrics dictionaries
        self.results: Dict[str, List[Dict[str, float]]] = {}
        
    def add_run(self, config_name: str, seed: int, draw: int, metrics: Dict[str, float]) -> None:
        """Stores individual run metrics."""
        if config_name not in self.results:
            self.results[config_name] = []
        metrics_copy = metrics.copy()
        metrics_copy['seed'] = seed
        metrics_copy['draw'] = draw
        self.results[config_name].append(metrics_copy)
        
    def get_summary(self, config_name: str) -> Dict[str, Dict[str, float]]:
        """Returns mean, std, min, max, CI for all metrics for a given config."""
        runs = self.results.get(config_name, [])
        if not runs:
            return {}
            
        summary = {}
        # Get all keys except tracking keys
        metric_keys = [k for k in runs[0].keys() if k not in ['seed', 'draw']]
        
        for key in metric_keys:
            values = [run[key] for run in runs if key in run]
            if not values:
                continue
                
            n = len(values)
            mean = np.mean(values)
            std = np.std(values)
            ci = 1.96 * std / np.sqrt(n) if n > 0 else 0
            
            summary[key] = {
                'mean': float(mean),
                'std': float(std),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'ci_95': float(ci),
                'n_runs': n
            }
        return summary
        
    def get_all_summaries(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Returns summary for all configs."""
        return {config: self.get_summary(config) for config in self.results}
        
    def to_dataframe(self) -> pd.DataFrame:
        """Converts raw results to pandas DataFrame."""
        rows = []
        for config, runs in self.results.items():
            for run in runs:
                row = run.copy()
                row['config_name'] = config
                rows.append(row)
        return pd.DataFrame(rows)
        
    def save(self, filepath: str) -> None:
        """Saves aggregator state to JSON."""
        save_results(self.results, filepath)
        
    def load(self, filepath: str) -> None:
        """Loads aggregator state from JSON."""
        self.results = load_results(filepath)
