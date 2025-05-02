from mlx_lm import load
import torch

def load_worker1_model(path):
    print("🔧 [Worker1] Loading layers 0–15...")
    model, tokenizer = load(path)
    model.model.layers = model.model.layers[:16]
    return model, tokenizer

def load_worker2_model(path):
    print("🔧 [Worker2] Loading layers 16–31...")
    model, tokenizer = load(path)
    model.model.layers = model.model.layers[16:]
    return model

def encode(prompt, tokenizer):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return tokens

def forward_worker1(model, input_ids):
    input_ids = input_ids.to("cpu")
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        for layer in model.model.layers:
            hidden = layer(hidden)[0]
    return hidden.cpu(), input_ids.shape[1]

def forward_worker2(model, hidden, input_len):
    hidden = hidden.to("cpu")
    with torch.no_grad():
        for layer in model.model.layers:
            hidden = layer(hidden)[0]
        logits = model.lm_head(hidden)
    return logits[:, input_len - 1:].cpu()

# For controller to call remotely
def encode_and_forward(prompt):
    from worker1 import encode_and_forward as worker1_fn
    return worker1_fn(prompt)

def forward_hidden(hidden, input_len):
    from worker2 import forward_hidden as worker2_fn
    return worker2_fn(hidden, input_len)
