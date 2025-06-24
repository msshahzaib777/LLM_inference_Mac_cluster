from Exo_wrapper.exo_client import ExoClient
import time
# Local server example
# model = "mlx-community/DeepSeek-R1-0528-4bit"
model = "DeepSeek R1 Distill Llama 70B (8-bit)"
model = "Qwen2.5-VL-7B-Instruct"
client = ExoClient(api_url="http://127.0.0.1:52415", model=model)
# Initialize chat history
messages = [{"role": "system", "content": "You are a helpful assistant."}]

print("💬 Start chatting with the model (type 'exit' to quit):")

while True:
    user_input = input("👤 You: ")
    if user_input.lower() in ("exit", "quit"):
        break

    messages.append({"role": "user", "content": user_input})
    print("🤖 Assistant: ", end="", flush=True)

    full_response = ""
    token_count = 0
    start_time = time.time()

    for token in client.chat_stream(messages):
        print(token, end="", flush=True)
        full_response += token
        token_count += 1

    end_time = time.time()
    elapsed = end_time - start_time
    tps = token_count / elapsed if elapsed > 0 else 0

    messages.append({"role": "assistant", "content": full_response})
    print(f"\n⚡ Tokens: {token_count} | Time: {elapsed:.2f}s | Speed: {tps:.2f} tokens/sec\n")