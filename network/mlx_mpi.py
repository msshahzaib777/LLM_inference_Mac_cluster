from config import config
import mlx.core as mx
from utils.utils import log_debug
from .interface import NetworkInterface

class MLXBackend(NetworkInterface):
    def wait_for_tensor(self, source_rank=0, **kwargs):
        tensor_name = kwargs.get('tensor_name')
        log_debug(f"[Receiver] Receiving tensor '{tensor_name}' from rank {source_rank}")
        template_tensor = config.get_tensor_template(tensor_name)
        return mx.distributed.recv(template_tensor.shape, template_tensor.dtype, src=source_rank)

    def send_tensor(self, tensor, dest_rank=1, tag=0, **kwargs):
        log_debug(f"[Sender] Sending tensor to rank {dest_rank}")
        mx.distributed.send(tensor, dst=dest_rank)
