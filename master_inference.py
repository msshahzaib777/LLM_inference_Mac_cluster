from transformers import AutoTokenizer
import glob
import json
import mlx.core as mx
import mlx.nn as nn
from models.qwen2 import Model, ModelArgs
from generate import generate

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
    tokenizer = AutoTokenizer.from_pretrained("./DeepSeek-R1-Distill-Qwen-32B")
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! How are you doing today?"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = "Hello! How are you doing today?"
    # STEP 1: LOAD FIRST HALF
    model = load_model(model_path, 0, 35)
    response = generate(prompt, model, tokenizer, temperature=0.6, top_k=10, top_p=0.85, max_length=200)
    print(response)