import numpy as np
import mlx.core as mx

from utils.utils import log_debug


def sample_next_token_from_logits(
    logits,
    temperature=1.0,
    top_k=50,
    top_p=0.9
):
    if logits.ndim == 2:
        logits = logits[0]
    assert logits.ndim == 1, "Logits must be 1D"

    if temperature <= 0:
        raise ValueError("Temperature must be > 0")

    logits = logits / temperature
    probs = np.exp(logits - np.max(logits))
    probs /= np.sum(probs)

    if top_k > 0 and top_k < len(probs):
        top_k_indices = probs.argsort()[-top_k:]
        filtered = np.zeros_like(probs)
        filtered[top_k_indices] = probs[top_k_indices]
        probs = filtered / filtered.sum()

    if top_p < 1.0:
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumulative_probs, top_p) + 1
        if cutoff_idx == 0:
            cutoff_idx = 1
        keep_indices = sorted_indices[:cutoff_idx]
        mask = np.zeros_like(probs)
        mask[keep_indices] = probs[keep_indices]
        probs = mask / mask.sum()

    return int(np.random.choice(len(probs), p=probs))

def sample_next_token(logits, temperature=1.0, top_k=50, top_p=0.95):
    """
    Apply temperature scaling, top-k and top-p filtering to logits and sample next token.
    """
    log_debug(f"[Sampler] Sampling next token with temperature={temperature}, top_k={top_k}, top_p={top_p}")

    logits = logits / temperature
    # Properly convert MLX array to NumPy array
    if hasattr(logits, '__array__') or isinstance(logits, mx.array):
        logits_np = np.asarray(logits, dtype=np.float32)
    else:
        logits_np = np.array(logits, dtype=np.float32)

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