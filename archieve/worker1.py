import os
import torch.distributed.rpc as rpc
from model_split import load_worker1_model, forward_worker1, encode

print("🔧 Loading model on Worker1")
model, tokenizer = load_worker1_model("mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit")

def encode_and_forward(prompt):
    print(f"📝 [Worker1] Prompt received: {prompt}")
    input_ids = encode(prompt, tokenizer)
    hidden, input_len = forward_worker1(model, input_ids)
    return hidden, input_len, input_ids

def main():
    rpc.init_rpc(
        name="worker1",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    print("✅ [Worker1] RPC initialized")
    import time
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
