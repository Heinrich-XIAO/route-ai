from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class DatasetConfig:
    name: str = "tscircuit/zero-obstacle-high-density-z01"
    split: str = "train"
    image_size: int = 256
    max_samples: int | None = 10000


@dataclass
class ModelConfig:
    input_channels: int = 3
    latent_dim: int = 64
    base_channels: int = 32


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    grad_clip: float = 1.0


@dataclass
class OutputConfig:
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100
    save_interval: int = 1000


@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            dataset=DatasetConfig(**data.get("dataset", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            output=OutputConfig(**data.get("output", {})),
        )
