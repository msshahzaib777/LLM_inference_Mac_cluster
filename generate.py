import time
import mlx.core as mx
import numpy as np
from network import network
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

    # Compute final probability distribution
    exp_logits = np.exp(logits_np - np.max(logits_np))
    probs = exp_logits / np.sum(exp_logits)

    next_token = np.random.choice(len(probs), p=probs)
    log_debug(f"[Sampler] Sampled next token id: {next_token}")
    return int(next_token)


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

        # Send hidden state to worker (rank 1)
        half_pass_start_time = time.time()
        network.send_tensor(hidden, 1)
        log_debug(f"[Generate] Sent hidden state to rank 1")

        # Receive logits back from rank 1 (use template to know shape/dtype)
        logits = network.wait_for_tensor(1)
        half_pass_time = time.time() - half_pass_start_time
        half_pass_time_list.append(half_pass_time)
        log_debug(f"[Generate] Received logits from rank 1: shape={logits.shape} in {half_pass_time} seconds")

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
