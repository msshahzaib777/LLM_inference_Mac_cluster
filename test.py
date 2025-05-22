from mpi4py import MPI
import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer
from config import config as cfg
from network import network
from utils.utils import load_model, log_debug

if cfg.rank == 0:
    log_debug("[Rank 0] Loading model and tokenizer")
    model_path = cfg.get("model_path")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_model(model_path, 0, 35)

    log_debug("[Rank 0] Preparing prompt and encoding")
    messages = [{"role": "system", "content": "You are a AI assistant with a lot of knowledge."}]
    messages.append({"role": "user", "content": "Hello! How are you?"})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, return_tensors="mlx")
    log_debug(f"[Rank 0] Encoded input_ids: shape={input_ids.shape}, dtype={input_ids.dtype}")

    hidden = model(input_ids)
    log_debug(f"[Rank 0] Computed hidden state: shape={hidden.shape}, dtype={hidden.dtype}")

    log_debug("[Rank 0] Sending hidden tensor to Rank 1")
    network.send_tensor(hidden, 1)

    log_debug("[Rank 0] Waiting for returned tensor from Rank 1")
    returned_tensor = network.wait_for_tensor(1)
    log_debug(f"[Rank 0] Received tensor: shape={returned_tensor.shape}, dtype={returned_tensor.dtype}")

    # Compare tensors
    log_debug("[Rank 0] Comparing tensors for equality")
    if np.allclose(np.array(hidden), np.array(returned_tensor), rtol=1e-3, atol=1e-5):
        log_debug("✅ Tensor roundtrip success: data preserved")
        print("✅ Tensor roundtrip success: data preserved")
    else:
        diff = np.max(np.abs(np.array(hidden) - np.array(returned_tensor)))
        log_debug(f"❌ Tensor mismatch. Max difference: {diff}")
        print(f"❌ Tensor mismatch. Max difference: {diff}")

elif cfg.rank == 1:
    log_debug("[Rank 1] Waiting to receive tensor from Rank 0")
    hidden = network.wait_for_tensor(0)
    log_debug(f"[Rank 1] Received tensor: shape={hidden.shape}, dtype={hidden.dtype}")

    log_debug("[Rank 1] Sending tensor back to Rank 0")
    network.send_tensor(hidden, 0)
