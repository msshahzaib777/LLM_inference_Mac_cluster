import torch
import json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model, load_checkpoint_and_dispatch

from generate_device_map import generate_device_map

checkpoint = "/Users/studentone/Documents/LLM_inference/DeepSeek-R1-Distill-Qwen-32B"

# Load configuration
config = AutoConfig.from_pretrained(checkpoint)

# Initialize model with empty weights
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# Define device map manually
device_map = generate_device_map(checkpoint + "/model.safetensors.index.json", 2)

model = load_checkpoint_and_dispatch(model, checkpoint, device_map=device_map)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenize input
input_text = "Explain the theory of relativity."
inputs = tokenizer(input_text, return_tensors="pt").to(0)  # Send inputs to device 0

# Generate output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=5)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))