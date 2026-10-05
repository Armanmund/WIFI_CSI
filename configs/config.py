from dataclasses import dataclass, field
from typing import List, Dict
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class DataConfig:
    data_folder: str = 'm1c4_PCA_80_300_extracted_3x4'
    environments: List[str] = field(default_factory=lambda: ['A1', 'A2', 'A3'])
    source_env: str = 'A1'
    target_envs: List[str] = field(default_factory=lambda: ['A2', 'A3'])
    activities: List[str] = field(default_factory=lambda: ['walk', 'empty', 'jump', 'stand'])
    label_map: Dict[str, int] = field(default_factory=lambda: {'walk': 0, 'empty': 1, 'jump': 2, 'stand': 3})
    data_format: str = 'PCA'
    num_classes: int = 4

@dataclass
class ModelConfig:
    model_type: str = 'cnn'
    conv_channels: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    fc_dims: List[int] = field(default_factory=lambda: [1000, 80, 40])
    num_classes: int = 4
    dropout_rate: float = 0.3

@dataclass
class TrainConfig:
    num_epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    step_size: int = 50
    gamma: float = 0.5
    early_stopping_patience: int = 20
    validation_split: float = 0.15

@dataclass
class FineTuneConfig:
    shots_per_class: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 15, 20, 30, 50])
    num_seeds: int = 5
    num_sample_draws: int = 5
    ft_epochs: int = 100
    ft_lr: float = 1e-4
    ft_weight_decay: float = 1e-4
    freeze_features: bool = False
    ft_patience: int = 15

@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir: Path = BASE_DIR / 'results'
    models_dir: Path = BASE_DIR / 'models'

    def __post_init__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
