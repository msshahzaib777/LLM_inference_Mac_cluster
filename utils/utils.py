import os, datetime, glob, json
import mlx.core as mx
import mlx.nn as nn
from models.qwen2 import Model, ModelArgs
from config import config as cfg

def log_debug(message, print_msg=False):
    """Append a debug message to the debug log file with timestamp."""
    try:
        # Check if we're in a cleanup state where built-ins might not be available
        if 'open' not in dir(__builtins__) and not hasattr(__builtins__, 'open'):
            return
        
        DEBUG_LOG_FILE = os.path.abspath(
            f"./logs/debug_log_rank{cfg.rank}.txt"
        )

        with open(DEBUG_LOG_FILE, "a") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
            if print_msg: print(message)
    except (NameError, AttributeError, OSError):
        # Silently ignore errors during cleanup/shutdown
        pass



def load_model(path_or_hf_repo: str, start_layer: int = None, end_layer: int = None):
    log_debug(f"Starting to load model from '{path_or_hf_repo}' with layers {start_layer}-{end_layer}")

    path = path_or_hf_repo

    # Load config
    with open(path + "/config.json", "r") as f:
        config = json.load(f)
        log_debug("Loaded config.json")

        if start_layer is not None and end_layer is not None:
            config['start_layer'] = start_layer
            config['end_layer'] = end_layer
            log_debug(f"Updated config with start_layer={start_layer}, end_layer={end_layer}")

    # Find safetensor weight files
    weight_files = glob.glob(str(path + "/*.safetensors"))
    if not weight_files:
        error_msg = f"No safetensors found in {path}"
        log_debug(error_msg)
        raise FileNotFoundError(error_msg)
    log_debug(f"Found {len(weight_files)} weight files")

    # Load weights into a dict
    weights = {}
    for wf in weight_files:
        log_debug(f"Loading weights from {wf}")
        weights.update(mx.load(wf))
    log_debug(f"Total loaded weights: {len(weights)} tensors")

    # Instantiate model
    model_class, model_args_class = Model, ModelArgs
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
    log_debug("Initialized model class")

    # Sanitize weights if needed
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
        log_debug("Sanitized weights")

    # Apply quantization if specified in config
    if (quantization := config.get("quantization", None)) is not None:
        log_debug(f"Applying quantization: {quantization}")

        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )
        log_debug("Quantization complete")

    # Load weights into model
    model.load_weights(list(weights.items()))
    log_debug("Loaded weights into model")

    model.eval()
    log_debug("Set model to eval mode")

    return model

def trim_before_last_think(response):
    tag = "</think>"
    last_index = response.rfind(tag)
    if last_index != -1:
        # Keep everything after the last </think>
        return response[last_index + len(tag):].strip()
    else:
        # If no </think> tag found, return full response
        return response.strip()

# Define once at the top of the file
mlx_dtype_map = {
    'mlx.core.bfloat16': mx.bfloat16,
    'mlx.core.float16': mx.float16,
    'mlx.core.float32': mx.float32,
    'mlx.core.float64': mx.float64,
    'mlx.core.int32': mx.int32,
    'mlx.core.int64': mx.int64,
}
