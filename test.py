import mlx.core as mx

group = mx.distributed.init()
rank = group.rank()
size = group.size()

print(f"[Rank {rank}] Hello from rank {rank} of {size}", flush=True)

# Send a dummy ready signal to synchronize
if rank == 0:
    ready_signal = mx.array([1], dtype=mx.int32)
    mx.distributed.send(ready_signal, dst=1)
elif rank == 1:
    _ = mx.distributed.recv((1,), mx.int32, src=0)
    print(f"[Rank {rank}] Received ready signal from rank 0", flush=True)

if rank == 1:
    shape = (2, 2)
    dtype = mx.float32
    print(f"[Rank {rank}] Waiting to receive tensor...", flush=True)
    received_tensor = mx.distributed.recv(shape, dtype, src=0)
    print(f"[Rank {rank}] Received tensor:\n{received_tensor}", flush=True)
elif rank == 0:
    tensor = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    print(f"[Rank {rank}] Sending tensor:\n{tensor}", flush=True)
    mx.distributed.send(tensor, dst=1)
    print(f"[Rank {rank}] Send complete", flush=True)

print(f"[Rank {rank}] Done.", flush=True)
