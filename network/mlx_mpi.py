from config import config
import mlx.core as mx
from utils.utils import log_debug
from .interface import NetworkInterface

class MLXBackend(NetworkInterface):
    def wait_for_tensor(self, source_rank=0, **kwargs):
        tensor_name = kwargs.get('tensor_name', 'unnamed')
        log_debug(f"[Receiver] Receiving tensor '{tensor_name}' from rank {source_rank}")

        # Step 1: Receive shape
        shape_array = mx.distributed.recv((3,), mx.int64, src=source_rank, tag=100)
        shape = tuple(int(x.item()) for x in shape_array)

        # Step 2: Receive dtype length and bytes
        dtype_len_array = mx.distributed.recv((1,), mx.int64, src=source_rank, tag=101)
        dtype_len = int(dtype_len_array[0].item())

        dtype_bytes_array = mx.distributed.recv((dtype_len,), mx.uint8, src=source_rank, tag=102)
        dtype_str = ''.join([chr(x.item()) for x in dtype_bytes_array])

        # Map string to MX dtype
        dtype_map = {
            'float16': mx.float16,
            'float32': mx.float32,
            'bfloat16': mx.bfloat16,
            'int32': mx.int32,
            'int64': mx.int64,
        }
        dtype = dtype_map.get(dtype_str)
        if dtype is None:
            raise ValueError(f"Unsupported dtype received: {dtype_str}")

        # Step 3: Receive actual tensor
        tensor = mx.distributed.recv(shape, dtype, src=source_rank, tag=103)

        log_debug(f"[Receiver] Received tensor '{tensor_name}' with shape={shape}, dtype={dtype_str}")
        return tensor

    def send_tensor(self, tensor, dest_rank=1, **kwargs):
        tensor_name = kwargs.get('tensor_name', 'unnamed')
        shape = tensor.shape
        dtype_str = str(tensor.dtype)

        log_debug(f"[Sender] Sending tensor '{tensor_name}' to rank {dest_rank}, shape={shape}, dtype={dtype_str}")

        # Step 1: Send shape as mlx tensor (int64)
        shape_array = mx.array(shape, dtype=mx.int64)
        mx.distributed.send(shape_array, dest_rank, tag=100)

        # Step 2: Send dtype string as mlx uint8 tensor (send length + bytes)
        dtype_bytes = dtype_str.encode('utf-8')
        dtype_len = len(dtype_bytes)
        mx.distributed.send(mx.array([dtype_len], dtype=mx.int64), dest_rank, tag=101)
        mx.distributed.send(mx.array(list(dtype_bytes), dtype=mx.uint8), dest_rank, tag=102)

        # Step 3: Send actual tensor
        mx.distributed.send(tensor, dest_rank, tag=103)


