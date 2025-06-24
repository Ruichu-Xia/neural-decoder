import yaml
from pathlib import Path
from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    eeg_dir: str = Field(..., description="The directory containing the EEG data.")
    train_path: str = Field(..., description="The path to the training data.")
    test_path: str = Field(..., description="The path to the test data.")
    embedding_dir: str = Field(..., description="The directory containing the extracted embeddings.")

class AppConfig(BaseModel):
    data: DataConfig

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