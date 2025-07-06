import yaml
import torch
import numpy as np
import random
from pathlib import Path
from pydantic import BaseModel, Field


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DataConfig(BaseModel):
    eeg_dir: str = Field(..., description="The directory containing the EEG data.")
    image_dir: str = Field(..., description="The directory containing the image data.")
    train_path: str = Field(..., description="The path to the training data.")
    test_path: str = Field(..., description="The path to the test data.")
    extracted_embedding_dir: str = Field(..., description="The directory containing the extracted embeddings.")
    predicted_embedding_dir: str = Field(..., description="The directory containing the predicted embeddings.")


class RidgeRegressionConfig(BaseModel):
    alpha: float = Field(..., description="The regularization parameter for Ridge regression.")
    max_iter: int = Field(..., description="The maximum number of iterations for Ridge regression.")
    fit_intercept: bool = Field(..., description="Whether to fit an intercept for Ridge regression.")


class ModelConfig(BaseModel):
    checkpoint_dir: str = Field(..., description="The directory to save the models.")
    num_channels: int = Field(..., description="Number of EEG channels.")
    sequence_length: int = Field(..., description="Number of sequence time steps for each EEG channel.")
    clip_dim: int = Field(..., description="Clip latent dimension.")
    vae_dim: int = Field(..., description="VAE latent dimension.")
    ridge: RidgeRegressionConfig = Field(..., description="The configuration for Ridge regression.")


class EvaluationModelConfig(BaseModel):
    name: str = Field(..., description="Model name for evaluation")
    layers: list[str | int] = Field(..., description="Layers to extract features from")


class EvaluationConfig(BaseModel):
    models: list[EvaluationModelConfig] = Field(..., description="Models and layers for evaluation")
    batch_size: int = Field(..., description="Batch size for feature extraction")
    num_test_images: int = Field(..., description="Number of test images to evaluate")
    image_size: int = Field(..., description="Input image size for models")
    result_image_size: int = Field(..., description="Size for SSIM/pixel correlation evaluation")
    device_id: int = Field(..., description="CUDA device ID")


class AppConfig(BaseModel):
    seed: int
    data: DataConfig
    model: ModelConfig
    evaluation: EvaluationConfig


def find_project_root(marker: str = 'pyproject.toml') -> Path:
    """Find the project root by looking for a marker file."""
    path = Path.cwd()
    while path.parent != path:
        if (path / marker).exists():
            return path
        path = path.parent
    raise FileNotFoundError(f"Project root marker '{marker}' not found.")


def load_config() -> AppConfig:
    """
    Loads configuration from `config.yaml` and validates it.

    Returns:
        An instance of AppConfig with the loaded and validated settings.
    """
    project_root = find_project_root()
    config_path = project_root / "src" / "neural_decoder" / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    try:
        validated_config = AppConfig(**config_data)
        return validated_config
    except Exception as e:
        print(f"Error validating configuration: {e}")
        raise

config = load_config()

# Set the random seed globally for reproducibility
set_seed(config.seed)