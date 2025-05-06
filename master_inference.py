from mpi4py import MPI
from transformers import AutoTokenizer
from generate import generate
from network.mpi import send_tensor, wait_for_tensor
from utils.utils import load_model, log_debug
from worker_inference import main as worker_inference

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    log_debug("=== Master Script started ===")
    model_path = "./DeepSeek-R1-Distill-Qwen-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    log_debug("Loaded tokenizer")

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! How are you doing today?"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    log_debug("Applied chat template to messages")

    # For testing, override prompt with plain string
    prompt = "Hello! How are you doing today?"
    log_debug(f"Prompt to generate: '{prompt}'")

    # STEP 1: Load first half of the model (layers 0-35)
    log_debug("Loading first half of the model (layers 0-35)")
    model = load_model(model_path, 0, 35)

    # STEP 2: Generate response
    log_debug("Generating response...")
    response = generate(prompt, model, tokenizer, temperature=0.6, top_k=10, top_p=0.85, max_length=200)

    log_debug(f"Generated response: '{response}'")
    print(response)

    log_debug("=== Script finished ===")

if __name__ == "__main__":
    if rank == 0:
        main()
    if rank == 1:
        worker_inference()