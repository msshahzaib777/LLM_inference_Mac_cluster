from config import config
import mlx.core as mx
from utils.utils import log_debug
from .interface import NetworkInterface

class MLXBackend(NetworkInterface):
    def wait_for_tensor(self, source_rank=0, **kwargs):
        log_debug(f"[Receiver] Receiving tensor from rank {source_rank}")

        # Step 1: receive metadata
        meta = mx.distributed.recv((4,), mx.int64, src=source_rank)
        shape = tuple(int(x.item()) for x in meta[:3])
        dtype_code = int(meta[3].item())

        # Step 2: map dtype code
        dtype_reverse_map = {1: mx.float16, 2: mx.float32, 3: mx.bfloat16, 4: mx.int32, 5: mx.int64}
        dtype = dtype_reverse_map.get(dtype_code)
        if dtype is None:
            raise ValueError(f"Unsupported dtype code: {dtype_code}")

        # Step 3: receive payload
        numel = np.prod(shape)
        payload = mx.distributed.recv((numel,), dtype, src=source_rank)

        # Step 4: reshape
        tensor = payload.reshape(shape)

        log_debug(f"[Receiver] Received tensor with shape={shape}, dtype={dtype}")
        return tensor

    def send_tensor(self, tensor, dest_rank=1, **kwargs):
        shape = mx.array(tensor.shape, dtype=mx.int64)
        dtype_str = str(tensor.dtype)
        dtype_map = {'float16': 1, 'float32': 2, 'bfloat16': 3, 'int32': 4, 'int64': 5}
        dtype_code = mx.array([dtype_map[dtype_str]], dtype=mx.int64)

        log_debug(f"[Sender] Sending tensor to rank {dest_rank}, shape={tensor.shape}, dtype={dtype_str}")

        # Flatten the payload
        tensor_flat = tensor.flatten()

        # Step 1: send metadata
        metadata = mx.concatenate([shape, dtype_code], axis=0)
        mx.distributed.send(metadata, dest_rank)

        # Step 2: send payload
        mx.distributed.send(tensor_flat, dest_rank)