import os
import torch.distributed.rpc as rpc
from model_split import load_worker2_model, forward_worker2

print("🔧 Loading model on Worker2")
model = load_worker2_model("mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit")

def forward_hidden(hidden, input_len):
    print("🔁 [Worker2] Running second half forward")
    return forward_worker2(model, hidden, input_len)

def main():
    rpc.init_rpc(
        name="worker2",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    print("✅ [Worker2] RPC initialized")
    import time
    while True:
        time.sleep
