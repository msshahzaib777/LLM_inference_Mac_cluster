import os
import torch.distributed.rpc as rpc

def main():
    rpc.init_rpc(
        name="controller",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    print("✅ [Controller] RPC started")

    prompt = "Explain how a rocket works in simple terms."
    print(f"🚀 Sending prompt to worker: {prompt}")

    # Call worker's generate function
    response = rpc.rpc_sync("worker", lambda p: generate_response(p), args=(prompt,))
    print(f"\n🤖 Worker Response:\n{response}\n")

    rpc.shutdown()

if __name__ == "__main__":
    from worker import generate_response  # For local resolution
    main()
