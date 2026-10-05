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
    env_type = "train_" if "A1" in env_name else "test_"
    env_dir = Path(base_dir) / "few_shot_datasets" / data_folder / f"{env_type}{env_name}"
    
    for activity, label in label_map.items():
        activity_dir = env_dir / activity
        if not activity_dir.exists():
            print(f"Warning: Directory not found - {activity_dir}")
            continue
        for file in os.listdir(activity_dir):
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

def sample_k_shot(X, y, k, num_classes, seed):
    np.random.seed(seed)
    X_sampled = []
    y_sampled = []
    
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        n_samples = min(k, len(idx))
        selected_idx = np.random.choice(idx, n_samples, replace=False)
        X_sampled.append(X[selected_idx])
        y_sampled.append(y[selected_idx])
        
    if X_sampled:
        X_sampled = np.concatenate(X_sampled)
        y_sampled = np.concatenate(y_sampled)
    else:
        X_sampled, y_sampled = np.array([]), np.array([])
        
    return X_sampled, y_sampled

def get_full_test_data(base_dir, env_name, data_folder, batch_size=16, label_map=None):
    X, y = load_environment_data(base_dir, env_name, data_folder, label_map)
    dataset = CSIDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader
