import torch
import json

# from accelerate.inference import generate_device_map
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from generate_device_map import generate_device_map

checkpoint = "/Users/studentone/Documents/mlx_sharding/shard_0"

# Load configuration
config = AutoConfig.from_pretrained(checkpoint)

# Initialize model with empty weights
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

model = load_checkpoint_and_dispatch(model, checkpoint, device_map="auto", offload_folder="./offload_folder")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenize input
input_text = "Explain the theory of relativity."
inputs = tokenizer(input_text, return_tensors="pt").to(0)  # Send inputs to device 0

# Generate output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=5)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))