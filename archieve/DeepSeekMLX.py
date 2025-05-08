from mlx_lm import load, generate
import mlx.core as mx

group = mx.distributed.init(backend="ring")


model, tokenizer = load("mlx-community/DeepSeek-R1-Distill-Qwen-14B")
messages = [{"role": "system", "content": "You are a confident assistant. Skip <think> steps and give a direct answer."}]
while True:
    prompt = input("Me: ")
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages.append({"role": "user", "content": prompt})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    print(prompt)
    response = generate(model, tokenizer, prompt=prompt, verbose=True)
    messages.append({"role": "assistant", "content": response})
