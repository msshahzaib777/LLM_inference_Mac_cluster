import os
import torch.distributed.rpc as rpc
from mlx_lm import load, generate

# Load 14B model locally
print("🔧 [Worker] Loading model...")
model, tokenizer = load("/Users/student/Documents/LLM_inference/DeepSeek-R1-Distill-Qwen-14B")
print("✅ [Worker] Model ready.")

# Function exposed over RPC
def generate_response(prompt):
    print(f"🧠 [Worker] Generating response for: {prompt}")
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=100)

def main():
    rpc.init_rpc(
        name="worker",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    print("✅ [Worker] RPC initialized. Waiting for requests...")
    import time
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
