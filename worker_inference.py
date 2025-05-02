from transformers import AutoTokenizer
import glob
import json
import mlx.core as mx
import mlx.nn as nn
from models.qwen2 import Model, ModelArgs
from generate import generate
from network.receiver import wait_for_tensor
from network.sender import send_tensor


def load_model(path_or_hf_repo: str, start_layer: int = None, end_layer: int = None):
    path = path_or_hf_repo
    with open(path + "/config.json", "r") as f:
        config = json.load(f)
        if start_layer is not None and end_layer is not None:
            config['start_layer'] = start_layer
            config['end_layer'] = end_layer
    weight_files = glob.glob(str(path +  "/*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {path}")
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
    model_class, model_args_class = Model, ModelArgs

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    if (quantization := config.get("quantization", None)) is not None:
        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()))
    model.eval()
    return model



if __name__ == "__main__":
    model_path = "./DeepSeek-R1-Distill-Qwen-32B"
    model = load_model(model_path, 36, 64)
    while True:
        hidden = wait_for_tensor()
        logits = model(hidden)
        send_tensor("192.168.2.1", logits)