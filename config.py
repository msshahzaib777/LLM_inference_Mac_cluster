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

        model_path = self.config.get('model_path')
        if model_path:
            print(f"model_path: {model_path}")
            self._load_model_shapes(model_path)
            print("model shapes loaded")

        # Initialize distributed rank and size
        self._init_distributed()

    def _init_distributed(self):
        world = mx.distributed.init()
        self.rank = world.rank()
        self.size = world.size()
        print(f"[Config] Rank: {self.rank} / Size: {self.size}")

    def _load_model_shapes(self, model_path):
        """Read model config.json and populate tensor shapes in memory."""
        config_json = os.path.join(model_path, 'config.json')
        if not os.path.exists(config_json):
            raise FileNotFoundError(f"Model config.json not found at {config_json}")

        with open(config_json, 'r') as f:
            model_cfg = json.load(f)

        hidden_size = model_cfg.get('hidden_size')
        vocab_size = model_cfg.get('vocab_size')
        max_seq_length = model_cfg.get('max_position_embeddings', 512)

        if not (hidden_size and vocab_size):
            raise ValueError("Model config must contain 'hidden_size' and 'vocab_size'.")

        # Read dtype from config.yaml → fallback to float32
        tensor_shapes_cfg = self.config.get('tensor_shapes', {})
        dtype_hidden = tensor_shapes_cfg.get('hidden_state', {}).get('dtype', 'float32')
        dtype_logits = tensor_shapes_cfg.get('logits', {}).get('dtype', 'float32')

        tensor_shapes = self.config.setdefault('tensor_shapes', {})

        tensor_shapes.setdefault('hidden_state', {
            'shape': [1, max_seq_length, hidden_size],
            'dtype': dtype_hidden
        })

        tensor_shapes.setdefault('logits', {
            'shape': [1, max_seq_length, vocab_size],
            'dtype': dtype_logits
        })
        print(json.dumps(tensor_shapes, indent=4))
        self.config.setdefault('tensor_shapes', tensor_shapes)
        print(json.dumps(self.config, indent=4))

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
