import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split

def read_mat(filepath):
    try:
        mat = sio.loadmat(filepath)
        if 'cfm_data' in mat:
            return mat['cfm_data']
        elif 'iq_data' in mat:
            return mat['iq_data']
        else:
            raise KeyError(f"Neither 'cfm_data' nor 'iq_data' found in {filepath}")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def load_environment_data(base_dir, env_name, data_folder, label_map=None):
    if label_map is None:
        label_map = {'walk': 0, 'empty': 1, 'jump': 2, 'stand': 3}
    
    X, y = [], []
    base_path = Path(base_dir) / "few_shot_datasets" / data_folder
    
    # Resolve directory: match direct name or fallback to train_/test_ prefix
    candidates = [env_name, f"train_{env_name}", f"test_{env_name}"]
    env_dir = None
    for cand in candidates:
        candidate_path = base_path / cand
        if candidate_path.exists():
            env_dir = candidate_path
            break
            
    if env_dir is None:
        print(f"Warning: Environment directory not found for '{env_name}' under {base_path}")
        return np.array([]), np.array([])
    
    for activity, label in label_map.items():
        activity_dir = env_dir / activity
        if not activity_dir.exists():
            print(f"Warning: Directory not found - {activity_dir}")
            continue
        for file in sorted(os.listdir(activity_dir)):
            if file.endswith('.mat'):
                filepath = activity_dir / file
                data = read_mat(filepath)
                if data is not None:
                    X.append(data)
                    y.append(label)
    
    if not X:
        return np.array([]), np.array([])
    
    X = np.stack(X)
    y = np.array(y)
    print(f"Loaded environment {env_name} - X shape: {X.shape}, y shape: {y.shape}")
    return X, y

class CSIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloaders(X, y, batch_size, val_split, seed):
    indices = list(range(len(y)))
    
    # Ensure we have enough samples for stratification
    try:
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X, y, indices, test_size=val_split, random_state=seed, stratify=y
        )
    except ValueError:
        # If stratification fails (too few samples), do non-stratified split
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X, y, indices, test_size=val_split, random_state=seed
        )
    
    train_dataset = CSIDataset(X_train, y_train)
    val_dataset = CSIDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(X_train)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(X_val)), shuffle=False)
    
    return train_loader, val_loader, idx_train, idx_val

def sample_k_shot_indices(y, k, num_classes, seed):
    """
    Deterministically sample k indices per class and return both sampled and remaining indices.
    Guarantees strict zero overlap between sampled and remaining sets.
    """
    rng = np.random.default_rng(seed)
    sampled_indices = []
    
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        n_samples = min(k, len(idx))
        chosen = rng.choice(idx, n_samples, replace=False)
        sampled_indices.extend(chosen.tolist())
        
    sampled_set = set(sampled_indices)
    remaining_indices = [i for i in range(len(y)) if i not in sampled_set]
    return np.array(sampled_indices), np.array(remaining_indices)

def sample_k_shot(X, y, k, num_classes, seed, return_remaining=False):
    """
    Sample k shots per class from dataset.
    If return_remaining=True, returns (X_sampled, y_sampled, X_remaining, y_remaining).
    Otherwise returns (X_sampled, y_sampled) for backward compatibility.
    """
    sampled_idx, rem_idx = sample_k_shot_indices(y, k, num_classes, seed)
    if len(sampled_idx) == 0:
        empty = np.array([])
        return (empty, empty, empty, empty) if return_remaining else (empty, empty)
        
    X_s, y_s = X[sampled_idx], y[sampled_idx]
    if return_remaining:
        X_r, y_r = X[rem_idx], y[rem_idx]
        return X_s, y_s, X_r, y_r
    return X_s, y_s
