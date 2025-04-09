import os
import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef

# Remote class handles from workers
from model_split import DeepSeekPart1, DeepSeekPart2
from model_split import encode_and_forward_rref, forward_rref

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"🧠 [Controller] Starting RPC init with rank={rank}, world_size={world_size}")

    rpc.init_rpc(
        name="controller",
        rank=rank,
        world_size=world_size,
    )

    print("✅ [Controller] RPC initialized and ready!\n")

    # --- Step 1: Launch remote workers ---
    print("📡 Creating remote worker handles...")
    w1 = rpc.remote("worker1", DeepSeekPart1)
    w2 = rpc.remote("worker2", DeepSeekPart2)

    # --- Step 2: Prepare the input ---
    prompt = "What is the capital of France?"
    print(f"📝 Prompt: {prompt}")

    # --- Step 3: Encode and run first part ---
    print("🔁 Forwarding to Worker 1...")
    hidden, input_len, input_ids = rpc.rpc_sync("worker1", encode_and_forward_rref, args=(w1, prompt))
    # --- Step 4: Forward through second part ---
    print("🔁 Forwarding to Worker 2...")
    logits = rpc.rpc_sync("worker2", forward_rref, args=(w2, hidden, input_len))

    # --- Step 5: Decode top prediction ---
    probs = torch.softmax(logits[0], dim=-1)
    topk = torch.topk(probs, k=1)
    predicted_id = topk.indices[0].item()

    # Need tokenizer locally (on controller)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/path/to/deepseek-llm-32b")  # adjust

    predicted_token = tokenizer.decode([predicted_id])
    print(f"🤖 Predicted next token: {predicted_token}")

    rpc.shutdown()
    print("🛑 [Controller] RPC shutdown complete.")


if __name__ == "__main__":
    main()
