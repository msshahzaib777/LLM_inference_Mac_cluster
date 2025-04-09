from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "DeepSeek-R1-Distill-Qwen-32B"  # update this on each machine

class DeepSeekPart1:
    def __init__(self):
        print("🔧 [Part1] Loading layers 0–15...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            low_cpu_mem_usage=True,
            device_map={f"model.layers.{i}": "mps" for i in range(0, 16)},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model.eval()
        print("✅ [Part1] Model ready.")

    def encode_and_forward(self, text):
        print(f"[Part1] ✍️ Encoding input: {text}")
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to("mps")
        with torch.no_grad():
            hidden = self.model.model.embed_tokens(input_ids)
            for i in range(16):
                hidden = self.model.model.layers[i](hidden)[0]
        print("[Part1] ✅ Finished forward pass for layers 0–15")
        return hidden.cpu(), input_ids.shape[1], input_ids.cpu()


class DeepSeekPart2:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            low_cpu_mem_usage=True,
            device_map={f"model.layers.{i}": "mps" for i in range(16, 32)},
        )
        self.model.eval()

    def forward(self, hidden_states, input_len):
        hidden_states = hidden_states.to("mps")
        with torch.no_grad():
            for i in range(16, 32):
                hidden_states = self.model.model.layers[i](hidden_states)[0]
            logits = self.model.lm_head(hidden_states)
        return logits[:, input_len - 1:].cpu()

def encode_and_forward_rref(rref, text):
    return rref.local_value().encode_and_forward(text)

def forward_rref(rref, hidden, input_len):
    return rref.local_value().forward(hidden, input_len)
