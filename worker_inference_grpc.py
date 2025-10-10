"""
gRPC-based worker inference script.
This script starts a gRPC server instead of using the polling loop like the TCP/MPI versions.
"""

from config import config as cfg
from utils.utils import log_debug
from network.grpc_server import start_grpc_worker_server


def main():
    """Main function for gRPC worker inference."""
    log_debug("=== gRPC Worker script started ===")
    
    model_path = cfg.get('model_path')
    
    # Get gRPC configuration
    grpc_config = cfg.get('grpc', {})
    worker_port = grpc_config.get('worker_port', 50051)
    
    log_debug(f"Starting gRPC worker server on port {worker_port}")
    log_debug(f"Model path: {model_path}")
    log_debug("Loading second half of the model (layers 36-64)")
    
    try:
        # Start gRPC server for this worker
        start_grpc_worker_server(
            port=worker_port,
            model_path=model_path,
            start_layer=36,
            end_layer=64
        )
    except KeyboardInterrupt:
        log_debug("gRPC worker server interrupted by user")
    except Exception as e:
        log_debug(f"gRPC worker server failed: {e}")
        raise
    finally:
        log_debug("=== gRPC Worker script finished ===")


if __name__ == "__main__":
    main()