"""
Updated worker inference script that supports both legacy and gRPC backends.
For gRPC, it starts a server instead of using the polling loop.
"""

from config import config as cfg
from network import network
from utils.utils import log_debug, load_model


def main():
    log_debug("=== Worker script started ===")
    
    model_path = cfg.get('model_path')
    backend = cfg.get('network_backend')
    
    if backend == 'grpc':
        # Start gRPC server for this worker
        from network.grpc_server import start_grpc_worker_server
        
        log_debug("Starting gRPC worker server...")
        start_grpc_worker_server(
            port=cfg.get('grpc', {}).get('worker_port', 50051),
            model_path=model_path,
            start_layer=36,
            end_layer=64
        )
    else:
        # Legacy TCP/MPI behavior
        log_debug("Loading second half of the model (layers 36-64)")
        model = load_model(model_path, 36, 64)

        log_debug("Entering inference loop (waiting for incoming tensors)")
        loop = True
        while loop:
            # Wait for incoming hidden state tensor
            log_debug("Waiting for hidden state tensor from rank 0")
            try:
                hidden = network.wait_for_tensor(0)
                log_debug(f"Received hidden tensor: shape={hidden.shape}, dtype={hidden.dtype}")

                # Perform forward pass to get logits
                logits = model(hidden)
                log_debug(f"Computed logits: shape={logits.shape}, dtype={logits.dtype}")
                # Send logits back to rank 0
                log_debug("Sending logits tensor back to rank 0")

                network.send_tensor(logits, 0)
            except RuntimeError as e:
                loop = False
                log_debug("Exception encountered: {}".format(e))
        log_debug("=== Worker script finished ===")


if __name__ == "__main__":
    main()