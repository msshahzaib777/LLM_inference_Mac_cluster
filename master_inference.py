from transformers import AutoTokenizer
import os
import glob
import json
import mlx.core as mx
import mlx.nn as nn
from models.qwen2 import Model, ModelArgs
from generate import generate
import datetime

DEBUG_LOG_FILE = os.path.abspath("./logs/debug_log_rank" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".txt")


def log_debug(message):
    """Append a debug message to the debug log file with timestamp."""
    with open(DEBUG_LOG_FILE, "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")


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


if __name__ == "__main__":
    log_debug("=== Script started ===")

    model_path = "./DeepSeek-R1-Distill-Qwen-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    log_debug("Loaded tokenizer")

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! How are you doing today?"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    log_debug("Applied chat template to messages")

    # For testing, override prompt with plain string
    prompt = "Hello! How are you doing today?"
    log_debug(f"Prompt to generate: '{prompt}'")

    # STEP 1: Load first half of the model (layers 0-35)
    log_debug("Loading first half of the model (layers 0-35)")
    model = load_model(model_path, 0, 35)

    # STEP 2: Generate response
    log_debug("Generating response...")
    response = generate(prompt, model, tokenizer, temperature=0.6, top_k=10, top_p=0.85, max_length=200)

    log_debug(f"Generated response: '{response}'")
    print(response)

    log_debug("=== Script finished ===")