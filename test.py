import mlx.core as mx

group = mx.distributed.init(backend='mpi')
rank = group.rank()
size = group.size()

print(f"[Rank {rank}] Hello from rank {rank} of {size}", flush=True)
tensor = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
if rank == 0:
    print(f"[Rank {rank}] Sending tensor:\n{tensor}", flush=True)
    sent_tensor = mx.distributed.send(tensor, dst=1, group=group)
    print(f"[Rank {rank}] Send complete:\n{tensor}", flush=True)

if rank == 1:
    shape = tensor.shape
    dtype = tensor.dtype
    tensor = mx.zeros(shape, dtype=dtype)
    print(f"[Rank {rank}] Waiting to receive tensor...", flush=True)
    received_tensor = mx.distributed.recv_like(tensor, src=0, group=group)
    print(f"[Rank {rank}] Received tensor:\n{received_tensor}", flush=True)

print(f"[Rank {rank}] Done.", flush=True)
