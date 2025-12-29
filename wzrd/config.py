"""
Configuration management for WZRD pipeline.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


# Default configuration values (used if no config file is found)
DEFAULT_CONFIG = {
    'resolution': {
        'default_aspect': '16:9',
        'base_resolution': 1920,
        'aspect_tolerance': 0.02,
    },
    'darken': {
        'max_brightness': 0.15,
        'gamma': 1.5,
    },
    'subtract': {
        'threshold': 10,
        'boost': 1.1,
        'feather_radius': 4,
        'diff_mode': 'luminance',
        'min_alpha': 0.0,
        'output_mode': 'additive',
    },
}


def get_default_config_path() -> Path:
    """Get path to the default config file bundled with the package."""
    return Path(__file__).parent / 'default_config.yaml'


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to config file. If None, tries to load from:
                    1. ./config.yaml (current directory)
                    2. Package default config
                    3. Falls back to DEFAULT_CONFIG

    Returns:
        Configuration dictionary
    """
    # Try explicit path first
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    # Try current directory
    cwd_config = Path.cwd() / 'config.yaml'
    if cwd_config.exists():
        with open(cwd_config, 'r') as f:
            return yaml.safe_load(f)

    # Try package default
    default_path = get_default_config_path()
    if default_path.exists():
        with open(default_path, 'r') as f:
            return yaml.safe_load(f)

    # Fall back to hardcoded defaults
    return DEFAULT_CONFIG.copy()


def get_resolution_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Get resolution configuration section."""
    if config is None:
        config = load_config()
    return config.get('resolution', DEFAULT_CONFIG['resolution'])


def get_darken_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Get darken configuration section."""
    if config is None:
        config = load_config()
    return config.get('darken', DEFAULT_CONFIG['darken'])


def get_subtract_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Get subtract configuration section."""
    if config is None:
        config = load_config()
    return config.get('subtract', DEFAULT_CONFIG['subtract'])
