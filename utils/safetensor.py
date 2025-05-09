import os
from config import config as cfg

from safetensors.torch import safe_open, save_file

model_path= cfg.get("model_path")
for filename in os.listdir(model_path):
    if filename.endswith(".safetensors"):
        file_path = os.path.join(model_path, filename)
        tensors = {}
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        save_file(tensors, file_path, metadata={"format": "pt"})
        print(f"Updated metadata for {filename}")
