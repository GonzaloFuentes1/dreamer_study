"""
Utility functions for Dreamer (all versions).
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_config(version: str, env: str = "cheetah_run") -> Dict[str, Any]:
    """Load a predefined configuration.
    
    Args:
        version: Dreamer version ('v1', 'v2', 'v3' or 'dreamer_v1', 'dreamer_v2', 'dreamer_v3')
        env: Environment name (e.g., 'cheetah_run', 'walker_walk', 'small')
        
    Returns:
        Configuration dictionary
    """
    # Normalize version name
    if not version.startswith('dreamer_'):
        version = f'dreamer_{version}'
    
    config_dir = Path(__file__).parent.parent / version / "configs"
    config_path = config_dir / f"{env}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    return load_config(str(config_path))


def print_config(config: Dict[str, Any], indent: int = 0):
    """Pretty print configuration.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")
