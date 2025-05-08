from config import config as cfg
from network import network
from utils.utils import log_debug, load_model

def main():

    log_debug("=== Worker script started ===")

    model_path = cfg.get('model_path')

    # Load second half of the model (layers 36-64)
    log_debug("Loading second half of the model (layers 36-64)")
    model = load_model(model_path, 36, 64)

    log_debug("Entering inference loop (waiting for incoming tensors)")
    loop = True
    while loop:
        # Wait for incoming hidden state tensor
        log_debug("Waiting for hidden state tensor from rank 0")
        try:
            cfg.world.barrier()
            hidden = network.wait_for_tensor(0, tensor_name='hidden_state')
            log_debug(f"Received hidden tensor: shape={hidden.shape}, dtype={hidden.dtype}")

            # Perform forward pass to get logits
            logits = model(hidden)
            log_debug(f"Computed logits: shape={logits.shape}, dtype={logits.dtype}")
            # Send logits back to rank 0
            log_debug("Sending logits tensor back to rank 0")

            network.send_tensor(logits, 0)
        except RuntimeError as e:
            loop = False
    log_debug("=== Worker script finished ===")  # (theoretically unreachable here)

if __name__ == "__main__":
    main()
