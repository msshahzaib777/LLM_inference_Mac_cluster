# config.py
import json, os, yaml
import mlx.core as mx

class Config:
    """Global configuration loader for model paths, tensor shapes, and backend options."""

    _instance = None

    def __new__(cls, config_file='config.yaml'):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load(config_file)
        return cls._instance

    def _load(self, config_file):
        """Load the main config YAML and supplement it with model-based shape info."""
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: config file '{config_file}' not found. Using empty config.")
            self.config = {}

        # Initialize distributed rank and size
        self._init_distributed()

    def _init_distributed(self):
        world = mx.distributed.init()
        self.rank = world.rank()
        self.size = world.size()

    def get(self, key, default=None):
        """Generic getter for config values."""
        return self.config.get(key, default)

    def get_tensor_template(self, name):
        """Generate a zero-filled MLX tensor based on name defined in config."""
        info = self.config.get('tensor_shapes', {}).get(name)
        if not info:
            raise ValueError(f"Tensor template '{name}' not defined in config.yaml or model config.")

        shape = tuple(info['shape'])
        dtype_str = info.get('dtype', 'float32')
        dtype = getattr(mx, dtype_str)
        return mx.zeros(shape, dtype=dtype)

# Create a global config instance
config = Config()
