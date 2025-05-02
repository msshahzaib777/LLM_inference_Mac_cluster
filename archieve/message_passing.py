import os
import argparse
import torch
import torch.distributed as dist

def run(rank, world_size, backend):
    print(f"[Rank {rank}] Initializing process on backend: {backend}")

    # Create a tensor to send or receive
    tensor = torch.tensor([rank], dtype=torch.float32)

    if rank == 0:
        for r in range(1, world_size):
            dist.send(tensor=tensor, dst=r)
            print(f"[Rank 0] Sent data to Rank {r}")
    else:
        dist.recv(tensor=tensor, src=0)
        print(f"[Rank {rank}] Received data from Rank 0: {tensor.item()}")

def init_processes(backend):
    # These are set automatically by torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    run(rank, world_size, backend)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    args = parser.parse_args()

    init_processes(args.backend)
