import mlx.core as mx
import numpy as np
from network.mpi import wait_for_tensor, send_tensor
from utils.utils import log_debug

def sample_next_token(logits, temperature=1.0, top_k=50, top_p=0.95):
    """
    Apply temperature scaling, top-k and top-p filtering to logits and sample next token.
    """
    log_debug(f"[Sampler] Sampling next token with temperature={temperature}, top_k={top_k}, top_p={top_p}")

    logits = logits / temperature
    logits_np = np.array(logits)  # Convert to NumPy array for manipulation

    # Top-k filtering: keep top_k highest logits
    if top_k > 0:
        top_k_indices = np.argpartition(-logits_np, top_k)[:top_k]
        mask = np.full_like(logits_np, -np.inf)
        mask[top_k_indices] = logits_np[top_k_indices]
        logits_np = mask
        log_debug(f"[Sampler] Applied top-k filtering with top_k={top_k}")

    # Top-p filtering: keep smallest set of logits whose cumulative probability >= top_p
    if top_p < 1.0:
        sorted_indices = np.argsort(-logits_np)
        sorted_logits = logits_np[sorted_indices]

        exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
        probs = exp_logits / np.sum(exp_logits)
        cumulative_probs = np.cumsum(probs)

        cutoff = cumulative_probs > top_p
        if np.any(cutoff):
            first_cut = np.argmax(cutoff)
            sorted_logits[first_cut + 1:] = -np.inf
            logits_np[sorted_indices] = sorted_logits
            log_debug(f"[Sampler] Applied top-p filtering with top_p={top_p}")

    # Compute final probability distribution
    exp_logits = np.exp(logits_np - np.max(logits_np))
    probs = exp_logits / np.sum(exp_logits)

    next_token = np.random.choice(len(probs), p=probs)
    log_debug(f"[Sampler] Sampled next token id: {next_token}")
    return int(next_token)


def generate(prompt, model, tokenizer, max_length=200, temperature=1.0, top_k=50, top_p=0.95):
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
    log_debug(f"[Generate] Encoded input_ids: shape={input_ids.shape}")

    for step in range(max_length):
        log_debug(f"[Generate] Step {step + 1}/{max_length}")

        # Forward pass to get hidden state for first partition
        hidden = model(input_ids)
        log_debug(f"[Generate] Computed hidden state: shape={hidden.shape}")

        # Send hidden state to worker (rank 1)
        send_tensor(hidden, 1)
        log_debug(f"[Generate] Sent hidden state to rank 1")

        # Wait for logits back from worker
        logits = wait_for_tensor(1)
        log_debug(f"[Generate] Received logits from rank 1: shape={logits.shape}")

        # Get logits for last token
        logits_last = logits[:, -1, :]
        log_debug(f"[Generate] Extracted logits for last token: shape={logits_last.shape}")

        # Sample next token from logits
        next_token = sample_next_token(logits_last[0], temperature, top_k, top_p)

        # Append sampled token to input_ids
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)
        log_debug(f"[Generate] Appended next_token={next_token} to input_ids: new shape={input_ids.shape}")

        # Stop if end-of-sequence token is generated
        if next_token == tokenizer.eos_token_id:
            log_debug("[Generate] Stopping generation (EOS token encountered)")
            break

    # Convert generated tokens to output string
    output_ids = np.array(input_ids)[0]
    decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True)
    log_debug(f"[Generate] Decoded output: '{decoded_output}'")

    return decoded_output

# Example usage (commented out, for reference)
# tokenizer = AutoTokenizer.from_pretrained("mlx-community/DeepSeek-7B")
# model = load_your_mlx_model()  # load your MLX model here
# reply = generate("Tell me a joke", model, tokenizer, temperature=0.8, top_k=40, top_p=0.9)
# print(reply)
