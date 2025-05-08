import mlx.core as mx
import os

world = mx.distributed.init(backend="mpi")
print(f"[INFO] Rank: {world.rank()} / Size: {world.size()} / Hostname: {os.uname().nodename}")
