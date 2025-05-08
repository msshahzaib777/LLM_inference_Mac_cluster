import time

from config import config
import mlx.core as mx
from utils.utils import log_debug
from .interface import NetworkInterface

class MLXBackend(NetworkInterface):
    def wait_for_tensor(self, source_rank=0, **kwargs):
        log_debug(f"[Receiver] Receiving tensor from rank {source_rank}")
        time.sleep(50)
        # Step 1: receive dtype string length and string
        dtype_len_array = mx.distributed.recv((1,), mx.int64, src=source_rank)
        dtype_len = int(dtype_len_array[0].item())

        dtype_bytes_array = mx.distributed.recv((dtype_len,), mx.uint8, src=source_rank)
        dtype_str = ''.join([chr(x.item()) for x in dtype_bytes_array])
        dtype = getattr(mx, dtype_str, None)
        if dtype is None:
            raise ValueError(f"Unsupported dtype received: {dtype_str}")

        # Step 2: receive shape length and shape array
        shape_len_array = mx.distributed.recv((1,), mx.int64, src=source_rank)
        shape_len = int(shape_len_array[0].item())

        shape_array = mx.distributed.recv((shape_len,), mx.int64, src=source_rank)
        shape = tuple(int(x.item()) for x in shape_array)

        # Step 3: calculate number of elements
        numel = int(mx.array(shape, dtype=mx.int64).prod().item())

        # Step 4: receive payload
        payload = mx.distributed.recv((numel,), dtype, src=source_rank)

        # Step 5: reshape to final tensor
        tensor = payload.reshape(shape)

        log_debug(f"[Receiver] Received tensor with shape={shape}, dtype={dtype_str}")
        return tensor

    def send_tensor(self, tensor, dest_rank=1, **kwargs):
        dtype_str = str(tensor.dtype)
        dtype_bytes = dtype_str.encode('utf-8')
        dtype_len = len(dtype_bytes)

        # Step 1: send dtype length and dtype bytes
        mx.distributed.send(mx.array([dtype_len], dtype=mx.int64), dest_rank)
        mx.distributed.send(mx.array(list(dtype_bytes), dtype=mx.uint8), dest_rank)

        # Step 2: send shape length and shape array
        shape = tensor.shape
        shape_len = len(shape)
        mx.distributed.send(mx.array([shape_len], dtype=mx.int64), dest_rank)
        mx.distributed.send(mx.array(shape, dtype=mx.int64), dest_rank)

        # Step 3: send flattened payload
        tensor_flat = tensor.flatten()
        mx.distributed.send(tensor_flat, dest_rank)

        log_debug(f"[Sender] Sent tensor to rank {dest_rank}, shape={shape}, dtype={dtype_str}")