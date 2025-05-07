from termios import FLUSHO

from mpi4py import MPI
from transformers import AutoTokenizer
from generate import generate
from utils.utils import load_model, log_debug
from worker_inference import main as worker_inference
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    log_debug("=== Master Script started ===")
    model_path = "./DeepSeek-R1-Distill-Qwen-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    log_debug("Loaded tokenizer")

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

    # STEP 1: Load first half of the model (layers 0-35)
    log_debug("Loading first half of the model (layers 0-35)")
    model = load_model(model_path, 0, 35)
    log_debug("=== Chatbot started (type 'exit' to quit) ===", print_msg=True)

    # STEP 2: Take input and Generate response
    log_debug("Generating response (streaming)...")
    token_count = 0
    while True:
        try:
            log_debug("\nYou: ", print_msg=True)
            user_input = input()
            log_debug(user_input)
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            start_time = time.time()
            token_count += 1
            messages.append({"role": "user", "content": user_input})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            response = generate(prompt, model, tokenizer)
            # logic for token per second calculation
            end_time = time.time()
            elapsed_time = end_time - start_time
            log_debug("Qwen2.5: " + response, print_msg=True)
            if elapsed_time > 0:
                tps = token_count / elapsed_time
                log_debug(f"\n⚡ Tokens per second: {tps:.2f}")
            else:
                log_debug("\n⚡ Tokens per second: n/a (zero elapsed time)")

            messages.append({"role": "assistant", "content": response})

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

    log_debug("Finished generating response.")

    log_debug("=== Script finished ===")

if __name__ == "__main__":
    if rank == 0:
        main()
    if rank == 1:
        worker_inference()