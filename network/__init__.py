import os
from config import config
backend = config.get("network_backend")

if backend == 'mpi':
    from .mpi import MPIBackend as Backend
elif backend == 'mlx_mpi':
    from .mlx_mpi import MLXBackend as Backend
elif backend == 'tcp':
    from .tcp import TCPBackend as Backend
else:
    raise ValueError(f"Unsupported backend: {backend}")

network = Backend()
