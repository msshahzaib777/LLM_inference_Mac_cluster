import mlx.core as mx

# Initialize MLX distributed
group = mx.distributed.init()
rank = group.rank()
size = group.size()

print(f"[Rank {rank}] Hello from rank {rank} of {size}")

# Synchronize all ranks
mx.distributed.barrier()
print(f"[Rank {rank}] Passed barrier")

if rank == 0:
    # Sender: create and send tensor
    tensor = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    print(f"[Rank {rank}] Sending tensor:\n{tensor}")
    mx.distributed.send(tensor, dst=1)

elif rank == 1:
    # Receiver: create template tensor and receive into it
    template = mx.zeros((2, 2), dtype=mx.float32)
    received_tensor = mx.distributed.recv_like(template, src=0)
    print(f"[Rank {rank}] Received tensor using recv_like:\n{received_tensor}")

# Final barrier to cleanly finish
mx.distributed.barrier()
print(f"[Rank {rank}] Done.")
