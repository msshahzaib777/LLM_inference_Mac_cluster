import time
import mlx.core as mx
import numpy as np
from network import network
from utils.utils import log_debug
from utils.token_sampler import sample_next_token as sample_next_token


def generate(prompt, model, tokenizer, max_length=200, temperature=1.0, top_k=50, top_p=0.80):
    """
    Generate a response by:
    1. Encoding prompt
    2. Iteratively sending hidden states, receiving logits
    3. Sampling next token from logits
    4. Appending sampled token
    5. Stopping at EOS or reaching max_length
    """
    log_debug(f"[Generate] Starting generation for prompt: '{prompt}'")

    input_ids = tokenizer.encode(prompt, return_tensors="mlx")
    len_of_input_ids = len(input_ids[0])
    log_debug(f"[Generate] Encoded input_ids: shape={input_ids.shape}")
    max_length = max_length + len_of_input_ids
    token_time_list = []
    half_pass_time_list = []
    for step in range(max_length):
        start_time = time.time()
        log_debug(f"[Generate] Step {step + 1}/{max_length}")

        # Forward pass to get hidden state for first partition
        hidden = model(input_ids)
        log_debug(f"[Generate] Computed hidden state: shape={hidden.shape}")

        # Send hidden state to worker (rank 1) and receive logits back
        half_pass_start_time = time.time()
        
        # For gRPC backend, send_tensor returns the result directly (more efficient)
        from config import config as cfg
        if cfg.get("network_backend") == "grpc":
            logits = network.send_tensor(hidden, 1, step=step)
            log_debug(f"[Generate] Sent hidden state and received logits from rank 1 via gRPC")
        else:
            # Legacy TCP/MPI behavior
            network.send_tensor(hidden, 1)
            log_debug(f"[Generate] Sent hidden state to rank 1")
            logits = network.wait_for_tensor(1)
        
        half_pass_time = time.time() - half_pass_start_time
        half_pass_time_list.append(half_pass_time)
        log_debug(f"[Generate] Received logits from rank 1: shape={logits.shape} in {half_pass_time:.4f} seconds")

        # Get logits for last token
        logits_last = logits[:, -1, :]
        log_debug(f"[Generate] Extracted logits for last token: shape={logits_last.shape}")

        # Sample next token from logits
        next_token = sample_next_token(logits_last[0], temperature, top_k, top_p)

        # Append sampled token to input_ids
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)
        log_debug(f"[Generate] Generated reply: {tokenizer.decode(np.array(input_ids)[0])}")

        # Stop if end-of-sequence token is generated
        if next_token == tokenizer.eos_token_id:
            log_debug("[Generate] Stopping generation (EOS token encountered)")
            break
        # logic for token per second calculation
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            log_debug(f"seconds per token: {elapsed_time:.2f}")
            token_time_list.append(elapsed_time)
        else:
            log_debug("Tokens per second: n/a (zero elapsed time)")

    # # Convert generated tokens to output string
    output_ids = np.array(input_ids)[0]
    decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True)
    log_debug(f"[Generate] Decoded output: '{decoded_output}' with {1/mx.mean(mx.array(token_time_list))} tps and {mx.mean(mx.array(half_pass_time_list))} seconds per network pass.")

    return decoded_output

# Example usage (commented out, for reference)
# tokenizer = AutoTokenizer.from_pretrained("mlx-community/DeepSeek-7B")
# model = load_your_mlx_model()  # load your MLX model here
# reply = generate("Tell me a joke", model, tokenizer, temperature=0.8, top_k=40, top_p=0.9)
# print(reply)
