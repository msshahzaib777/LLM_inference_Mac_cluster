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

    # This calls generate_response() on the remote 'worker'
    response = rpc.rpc_sync("worker", "generate_response", args=(prompt,))
    print(f"\n🤖 Worker Response:\n{response}\n")

    rpc.shutdown()

if __name__ == "__main__":
    main()
