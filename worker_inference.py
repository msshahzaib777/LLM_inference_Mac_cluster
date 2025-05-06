from network.mpi import send_tensor, wait_for_tensor
from utils.utils import log_debug, load_model

def main():
    log_debug("=== Worker script started ===")

    model_path = "./DeepSeek-R1-Distill-Qwen-32B"

    # Load second half of the model (layers 36-64)
    log_debug("Loading second half of the model (layers 36-64)")
    model = load_model(model_path, 36, 64)

    log_debug("Entering inference loop (waiting for incoming tensors)")

    while True:
        # Wait for incoming hidden state tensor
        log_debug("Waiting for hidden state tensor from rank 0")
        hidden = wait_for_tensor(0, 0)
        log_debug(f"Received hidden tensor: shape={hidden.shape}, dtype={hidden.dtype}")

        # Perform forward pass to get logits
        logits = model(hidden)
        log_debug(f"Computed logits: shape={logits.shape}, dtype={logits.dtype}")

        # Send logits back to rank 0
        log_debug("Sending logits tensor back to rank 0")
        send_tensor(logits, 0)

    log_debug("=== Worker script finished ===")  # (theoretically unreachable here)

if __name__ == "__main__":
    main()
