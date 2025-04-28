import os
from safetensors.torch import safe_open, save_file

# Replace with the path to your model directory
model_dir = "/Users/studentone/Documents/LLM_inference/DeepSeek-R1-Distill-Qwen-32B"

for filename in os.listdir(model_dir):
    if filename.endswith(".safetensors"):
        file_path = os.path.join(model_dir, filename)
        tensors = {}
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        save_file(tensors, file_path, metadata={"format": "pt"})
        print(f"Updated metadata for {filename}")
