from mlx_lm import load, generate
import mlx.core as mx

group = mx.distributed.init(backend="ring")


model, tokenizer = load("mlx-community/DeepSeek-R1-Distill-Qwen-14B")
prompt="hello"

if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
