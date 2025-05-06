from mpi4py import MPI
import numpy as np
import mlx.core as mx
import datetime
import os

DEBUG_LOG_FILE = os.path.abspath(
    "./logs/debug_log_rank" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
)

def log_debug(message):
    """Append a debug message to the debug log file with timestamp."""
    with open(DEBUG_LOG_FILE, "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")

# Explicit mapping: NumPy dtype name → MPI dtype
numpy_to_mpi_dtype = {
    'float32': MPI.FLOAT,
    'float64': MPI.DOUBLE,
    'int32': MPI.INT,
    'int64': MPI.LONG,
    'uint8': MPI.UNSIGNED_CHAR,
    'int8': MPI.SIGNED_CHAR,
    'uint16': MPI.UNSIGNED_SHORT,
    'int16': MPI.SHORT,
    # Add more types if needed
}

def wait_for_tensor(source_rank=0, tag=0):
    """
    Receive a tensor (MLX array) from another MPI rank.
    1. Receive metadata (shape, dtype)
    2. Receive flattened tensor data
    3. Reshape and return as MLX array
    """
    comm = MPI.COMM_WORLD

    # Receive metadata (shape tuple and dtype string)
    metadata = comm.recv(source=source_rank, tag=tag)
    shape, dtype_str = metadata
    dtype = np.dtype(dtype_str)

    log_debug(f"[Receiver] Received metadata: shape={shape}, dtype={dtype_str}")

    num_elements = np.prod(shape)

    # Allocate empty NumPy array to receive data
    tensor_np = np.empty(num_elements, dtype=dtype)
    log_debug(f"[Receiver] Allocated empty array for {num_elements} elements")

    # Get MPI dtype from mapping
    mpi_dtype = numpy_to_mpi_dtype.get(dtype_str)
    if mpi_dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}' for MPI receive")

    # Receive the flattened tensor data
    comm.Recv([tensor_np, mpi_dtype], source=source_rank, tag=tag+1)
    log_debug(f"[Receiver] Received tensor data from rank {source_rank}")

    # Reshape NumPy array back to original shape
    tensor_np = tensor_np.reshape(shape)
    tensor_mx = mx.array(tensor_np)

    log_debug(f"[Receiver] Tensor reshaped to {tensor_mx.shape} and converted to MLX array")

    return tensor_mx

def send_tensor(tensor_mx, dest_rank=1, tag=0):
    """
    Send a tensor (MLX array) to another MPI rank.
    1. Send metadata (shape, dtype)
    2. Send flattened tensor data
    """
    comm = MPI.COMM_WORLD

    # Convert MLX array to NumPy for MPI sending
    tensor_np = np.array(tensor_mx)
    shape = tensor_np.shape
    dtype_str = tensor_np.dtype.name
    num_elements = np.prod(shape)

    log_debug(f"[Sender] Preparing to send tensor: shape={shape}, dtype={dtype_str}, num_elements={num_elements}")

    # Get MPI dtype from mapping
    mpi_dtype = numpy_to_mpi_dtype.get(dtype_str)
    if mpi_dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}' for MPI send")

    # Send metadata first (shape and dtype string)
    metadata = (shape, dtype_str)
    comm.send(metadata, dest=dest_rank, tag=tag)
    log_debug(f"[Sender] Sent metadata to rank {dest_rank}")

    # Send the flattened data buffer
    comm.Send([tensor_np.flatten(), mpi_dtype], dest=dest_rank, tag=tag+1)
    log_debug(f"[Sender] Sent tensor data to rank {dest_rank}")