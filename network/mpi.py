from utils.utils import mlx_dtype_map, log_debug
from mpi4py import MPI
import numpy as np
import mlx.core as mx

from .interface import NetworkInterface

class MPIBackend(NetworkInterface):
    def wait_for_tensor(self, source_rank=0, **kwargs):
        """
        Receive a tensor (MLX array) as raw bytes from another MPI rank,
        preserving MLX special types like bfloat16.
        """
        comm = MPI.COMM_WORLD
        tag = kwargs.get('tag', 0)
        # Step 1: Receive metadata (shape, numpy dtype, mlx dtype)
        metadata = comm.recv(source=source_rank, tag=tag)
        shape, numpy_dtype_str, mlx_dtype_str = metadata
        numpy_dtype = np.dtype(numpy_dtype_str)

        log_debug(f"[Receiver] Received metadata: shape={shape}, numpy_dtype={numpy_dtype_str}, mlx_dtype={mlx_dtype_str}")

        num_elements = np.prod(shape)
        num_bytes = num_elements * numpy_dtype.itemsize

        # Step 2: Prepare byte buffer
        recv_buffer = bytearray(num_bytes)

        # Step 3: Receive raw bytes
        req = comm.Irecv([recv_buffer, MPI.BYTE], source=source_rank, tag=tag + 1)
        req.Wait()
        log_debug(f"[Receiver] Received {num_bytes} bytes from rank {source_rank}")

        # Step 4: Reconstruct NumPy array
        tensor_np = np.frombuffer(recv_buffer, dtype=numpy_dtype).reshape(shape)

        # Step 5: Convert to MLX array with correct dtype
        tensor_mx = mx.array(tensor_np, dtype=mlx_dtype_map[mlx_dtype_str])

        log_debug(f"[Receiver] Converted to MLX array with shape={tensor_mx.shape}, dtype={tensor_mx.dtype}")

        return tensor_mx


    def send_tensor(self, tensor_mx, dest_rank=1, **kwargs):
        """
        Send a tensor (MLX array) as raw bytes to another MPI rank,
        including both NumPy dtype and MLX dtype in metadata.
        """
        comm = MPI.COMM_WORLD
        tag = kwargs.get('tag', 0)
        # Step 1: Convert MLX array to NumPy array
        tensor_np = np.array(tensor_mx)
        shape = tensor_np.shape
        numpy_dtype_str = tensor_np.dtype.str       # e.g., '<f4'
        mlx_dtype_str = str(tensor_mx.dtype)       # e.g., 'bfloat16'
        num_elements = np.prod(shape)
        num_bytes = num_elements * tensor_np.dtype.itemsize

        log_debug(f"[Sender] Preparing to send tensor: shape={shape}, numpy_dtype={numpy_dtype_str}, mlx_dtype={mlx_dtype_str}, num_elements={num_elements}, num_bytes={num_bytes}")

        # Step 2: Send metadata
        metadata = (shape, numpy_dtype_str, mlx_dtype_str)
        comm.send(metadata, dest=dest_rank, tag=tag)
        log_debug(f"[Sender] Sent metadata to rank {dest_rank}")

        # Step 3: Send raw bytes
        send_buffer = tensor_np.tobytes()
        req = comm.Isend([send_buffer, MPI.BYTE], dest=dest_rank, tag=tag + 1)
        req.Wait()
        log_debug(f"[Sender] Sent {num_bytes} bytes to rank {dest_rank}")