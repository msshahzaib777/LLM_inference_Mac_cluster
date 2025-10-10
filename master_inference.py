import os
from transformers import AutoTokenizer
from generate import generate
from utils.utils import load_model, log_debug, trim_before_last_think
from worker_inference import main as worker_inference
from worker_inference_grpc import main as worker_inference_grpc
from config import config as cfg

print(f"[INFO] Rank: {cfg.rank} / Size: {cfg.size} / Hostname: {os.uname().nodename}")


def main():
    log_debug("=== Master Script started ===")
    model_path = cfg.get("model_path")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    log_debug("Loaded tokenizer")

    messages = [{"role": "system", "content": "You are a AI assistant with alot of knowledge."}]
    # STEP 1: Load first half of the model (layers 0-35)
    log_debug("Loading first half of the model (layers 0-35)")
    model = load_model(model_path, 0, 35)
    log_debug("=== Chatbot started (type 'exit' to quit) ===", print_msg=True)

    # STEP 2: Take input and Generate response
    log_debug("Generating response (streaming)...")
    while True:
        try:
            log_debug("\nYou: ", print_msg=True)
            user_input = input()
            log_debug(user_input)
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            messages.append({"role": "user", "content": user_input})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            response = generate(prompt, model, tokenizer, max_length=500)
            response = trim_before_last_think(response)
            log_debug("Qwen2.5: " + response, print_msg=True)
            messages.append({"role": "assistant", "content": response})

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

    log_debug("Finished generating response.")

    log_debug("=== Script finished ===")

if __name__ == "__main__":
    if cfg.rank == 0:
        main()
    if cfg.rank == 1:
        backend = cfg.get('network_backend')
        if backend == 'grpc':
            worker_inference_grpc()
        else:
            worker_inference()