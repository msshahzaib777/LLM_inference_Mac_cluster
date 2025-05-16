import os
import sys
import mlx.core as mx

group = mx.distributed.init(backend='mpi')
rank = group.rank()
size = group.size()

# Redirect stdout and stderr to a log file
log_file = open(f"./logs/debug_log_rank{rank}.txt", "w")
sys.stdout = log_file
sys.stderr = log_file

print(f"[Rank {rank}] Hello from rank {rank} of {size}", flush=True)

if rank == 0:
    tensor = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    print(f"[Rank {rank}] Sending tensor:\n{tensor}", flush=True)
    mx.distributed.send(tensor, dst=1, group=group)
    print(f"[Rank {rank}] Send complete.", flush=True)

elif rank == 1:
    shape = (2, 2)
    dtype = mx.float32
    print(f"[Rank {rank}] Waiting to receive tensor...", flush=True)
    received_tensor = mx.distributed.recv(shape, dtype, src=0, group=group)
    print(f"[Rank {rank}] Received tensor:\n{received_tensor}", flush=True)

print(f"[Rank {rank}] Done.", flush=True)

log_file.close()