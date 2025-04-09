import os
import torch
import torch.distributed.rpc as rpc

from model_split import encode_and_forward, forward_hidden

def main():
    rpc.init_rpc(
        name="controller",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    print("✅ [Controller] RPC initialized")

    prompt = "Tell me a fun fact about space."
    print(f"📨 Sending prompt: {prompt}")

    hidden, input_len, input_ids = rpc.rpc_sync("worker1", encode_and_forward, args=(prompt,))
    logits = rpc.rpc_sync("worker2", forward_hidden, args=(hidden, input_len))

    probs = torch.softmax(logits[0], dim=-1)
    topk = torch.topk(probs, k=1)
    predicted_id = topk.indices[0].item()
    from mlx_lm import load
    _, tokenizer =("mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit")
    token = tokenizer.decode([predicted_id])
    print(f"🤖 Predicted token: {token}")

    rpc.shutdown()

if __name__ == "__main__":
    main()
