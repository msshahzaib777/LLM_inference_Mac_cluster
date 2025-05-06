import mlx.core as mx
import numpy as np
from network.mpi import wait_for_tensor, send_tensor


def sample_next_token(logits, temperature=1.0, top_k=50, top_p=0.95):
    logits = logits / temperature
    logits_np = np.array(logits)  # Convert to NumPy

    # Top-k filtering
    if top_k > 0:
        top_k_indices = np.argpartition(-logits_np, top_k)[:top_k]
        mask = np.full_like(logits_np, -np.inf)
        mask[top_k_indices] = logits_np[top_k_indices]
        logits_np = mask

    # Top-p filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(-logits_np)
        sorted_logits = logits_np[sorted_indices]

        # ✅ log-sum-exp trick
        exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
        probs = exp_logits / np.sum(exp_logits)
        cumulative_probs = np.cumsum(probs)

        cutoff = cumulative_probs > top_p
        if np.any(cutoff):
            first_cut = np.argmax(cutoff)
            sorted_logits[first_cut + 1:] = -np.inf
            logits_np[sorted_indices] = sorted_logits

    # Final probability distribution
    exp_logits = np.exp(logits_np - np.max(logits_np))  # ✅ also safe here
    probs = exp_logits / np.sum(exp_logits)

    next_token = np.random.choice(len(probs), p=probs)
    return int(next_token)

def generate(prompt, model, tokenizer, max_length=200, temperature=1.0, top_k=50, top_p=0.95):
    input_ids = tokenizer.encode(prompt, return_tensors="mlx")

    for _ in range(max_length):
        hidden = model(input_ids)
        send_tensor(hidden, 1)
        logits = wait_for_tensor(1)
        logits_last = logits[:, -1, :]  # get logits for last token

        next_token = sample_next_token(logits_last[0], temperature, top_k, top_p)

        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        if next_token == tokenizer.eos_token_id:
            break

    output_ids = np.array(input_ids)[0]

    return tokenizer.decode(output_ids, skip_special_tokens=True)

# Example usage:
# tokenizer = AutoTokenizer.from_pretrained("mlx-community/DeepSeek-7B")
# model = load_your_mlx_model()  # <-- load your MLX model here
# reply = generate("Tell me a joke", model, tokenizer, temperature=0.8, top_k=40, top_p=0.9)
# print(reply)
