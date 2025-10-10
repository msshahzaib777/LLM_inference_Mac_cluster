import os
from config import config
backend = config.get("network_backend")

if backend == 'mpi':
    from .mpi import MPIBackend as Backend
elif backend == 'mlx_mpi':
    from .mlx_mpi import MLXBackend as Backend
elif backend == 'tcp':
    from .tcp import TCPBackend as Backend
elif backend == 'grpc':
    from .grpc_client import GRPCBackend as Backend
else:
    raise ValueError(f"Unsupported backend: {backend}")

network = Backend()
