import mlx.core as mx

group = mx.distributed.init(backend='mpi')
rank = group.rank()
size = group.size()

print(f"[Rank {rank}] Hello from rank {rank} of {size}", flush=True)

# Manual sync: send ready signal
if rank == 0:
    mx.distributed.send(mx.array([1], dtype=mx.int32), dst=1)
    print(f"[Rank {rank}] Sent ready signal", flush=True)
if rank == 1:
    _ = mx.distributed.recv((1,), mx.int32, src=0)
    print(f"[Rank {rank}] Received ready signal", flush=True)

if rank == 0:
    tensor = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    print(f"[Rank {rank}] Sending tensor:\n{tensor}", flush=True)
    mx.distributed.send(tensor, dst=1)
    print(f"[Rank {rank}] Send complete", flush=True)

if rank == 1:
    shape = (2, 2)
    dtype = mx.float32
    print(f"[Rank {rank}] Waiting to receive tensor...", flush=True)
    received_tensor = mx.distributed.recv(shape, dtype, src=0)
    print(f"[Rank {rank}] Received tensor:\n{received_tensor}", flush=True)

print(f"[Rank {rank}] Done.", flush=True)
