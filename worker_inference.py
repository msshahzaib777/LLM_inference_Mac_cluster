import glob
import json
import mlx.core as mx
import mlx.nn as nn
from models.qwen2 import Model, ModelArgs
from network.mpi import send_tensor, wait_for_tensor
from master_inference import log_debug

def load_model(path_or_hf_repo: str, start_layer: int = None, end_layer: int = None):
    """
    Load a model from a specified path, optionally selecting layer range.
    Includes quantization, weight loading, and evaluation mode.
    """
    log_debug(f"Starting to load model from '{path_or_hf_repo}' with layers {start_layer}-{end_layer}")
    path = path_or_hf_repo

    # Load config.json
    with open(path + "/config.json", "r") as f:
        config = json.load(f)
        log_debug("Loaded config.json")

        if start_layer is not None and end_layer is not None:
            config['start_layer'] = start_layer
            config['end_layer'] = end_layer
            log_debug(f"Updated config with start_layer={start_layer}, end_layer={end_layer}")

    # Find weight files
    weight_files = glob.glob(str(path + "/*.safetensors"))
    if not weight_files:
        error_msg = f"No safetensors found in {path}"
        log_debug(error_msg)
        raise FileNotFoundError(error_msg)
    log_debug(f"Found {len(weight_files)} weight files")

    # Load all weight tensors
    weights = {}
    for wf in weight_files:
        log_debug(f"Loading weights from {wf}")
        weights.update(mx.load(wf))
    log_debug(f"Loaded total of {len(weights)} tensors")

    # Initialize model class
    model_class, model_args_class = Model, ModelArgs
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
    log_debug("Initialized model instance")

    # Optionally sanitize weights
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
        log_debug("Sanitized weights")

    # Optionally apply quantization
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

if __name__ == "__main__":
    log_debug("=== Worker script started ===")

    model_path = "./DeepSeek-R1-Distill-Qwen-32B"

    # Load second half of the model (layers 36-64)
    log_debug("Loading second half of the model (layers 36-64)")
    model = load_model(model_path, 36, 64)

    log_debug("Entering inference loop (waiting for incoming tensors)")

    while True:
        # Wait for incoming hidden state tensor
        log_debug("Waiting for hidden state tensor from rank 0")
        hidden = wait_for_tensor(0, 0)
        log_debug(f"Received hidden tensor: shape={hidden.shape}, dtype={hidden.dtype}")

        # Perform forward pass to get logits
        logits = model(hidden)
        log_debug(f"Computed logits: shape={logits.shape}, dtype={logits.dtype}")

        # Send logits back to rank 0
        log_debug("Sending logits tensor back to rank 0")
        send_tensor(logits, 0)

    log_debug("=== Worker script finished ===")  # (theoretically unreachable here)
