import os
import torch.distributed.rpc as rpc
from model_split import DeepSeekPart1

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"👷‍♂️ [Worker1] Starting RPC init with rank={rank}, world_size={world_size}")

    rpc.init_rpc(
        name="worker1",
        rank=rank,
        world_size=world_size,
    )

    print("✅ [Worker1] RPC initialized and ready!")

    import time
    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()
