import mlx.core as mx

# Initialize the MPI group
group = mx.distributed.init(backend='mpi')
rank = group.rank()
size = group.size()

print(f"[Rank {rank}] Hello from rank {rank} of {size}", flush=True)

if rank == 0:
    # Create a tensor to send
    tensor = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    print(f"[Rank {rank}] Sending tensor:\n{tensor}", flush=True)
    # Send the tensor to rank 1
    mx.distributed.send(tensor, dst=1, group=group)
    print(f"[Rank {rank}] Send complete.", flush=True)

elif rank == 1:
    # Define the shape and dtype of the expected tensor
    shape = (2, 2)
    dtype = mx.float32
    print(f"[Rank {rank}] Waiting to receive tensor...", flush=True)
    # Receive the tensor from rank 0
    received_tensor = mx.distributed.recv(shape, dtype, src=0, group=group)
    print(f"[Rank {rank}] Received tensor:\n{received_tensor}", flush=True)

print(f"[Rank {rank}] Done.", flush=True)