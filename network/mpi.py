from mpi4py import MPI
import numpy as np
import mlx.core as mx


def wait_for_tensor(source_rank=0, tag=0):
    comm = MPI.COMM_WORLD

    # Receive metadata: [shape tuple as list, dtype string]
    metadata = comm.recv(source=source_rank, tag=tag)
    shape, dtype_str = metadata
    dtype = np.dtype(dtype_str)

    print(f"[Receiver] Expecting tensor of shape {shape} and dtype {dtype}")

    num_elements = np.prod(shape)

    # Prepare empty NumPy array
    tensor_np = np.empty(num_elements, dtype=dtype)

    # Receive flattened data
    comm.Recv([tensor_np, MPI._typedict[dtype.char]], source=source_rank, tag=tag+1)

    # Reshape and convert to MLX tensor
    tensor_np = tensor_np.reshape(shape)
    tensor_mx = mx.array(tensor_np)

    print(f"[Receiver] Received tensor successfully: shape {tensor_mx.shape}, dtype {tensor_mx.dtype}")
    return tensor_mx

def send_tensor(tensor_mx, dest_rank=1, tag=0):
    comm = MPI.COMM_WORLD

    tensor_np = np.array(tensor_mx)  # MLX → NumPy
    shape = tensor_np.shape
    dtype_str = tensor_np.dtype.name
    num_elements = np.prod(shape)

    # Send metadata: [shape as list, dtype string]
    metadata = (shape, dtype_str)
    comm.send(metadata, dest=dest_rank, tag=tag)

    print(f"[Sender] Sent metadata: shape {shape}, dtype {dtype_str}")

    # Send flattened data
    comm.Send([tensor_np.flatten(), MPI._typedict[tensor_np.dtype.char]], dest=dest_rank, tag=tag+1)

    print(f"[Sender] Sent tensor data of {num_elements} elements to rank {dest_rank}")
