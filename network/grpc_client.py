"""
gRPC client implementation for distributed LLM inference master node.
This client sends hidden states to worker nodes and receives logits back.
"""

import grpc
import time
import uuid
import numpy as np
import mlx.core as mx
from typing import Optional, Dict, Any

from . import grpc_service_pb2
from . import grpc_service_pb2_grpc
from .interface import NetworkInterface
from utils.utils import log_debug
from config import config as cfg


class GRPCClient:
    """gRPC client for sending tensor requests to worker nodes."""
    
    def __init__(self, worker_ip: str, worker_port: int):
        self.worker_ip = worker_ip
        self.worker_port = worker_port
        self.channel = None
        self.stub = None
        self.connect()
    
    def connect(self):
        """Establish connection to the worker node."""
        try:
            # Configure gRPC channel options for high throughput
            options = [
                ('grpc.keepalive_time_ms', cfg.get('grpc', {}).get('keepalive_time_ms', 30000)),
                ('grpc.keepalive_timeout_ms', cfg.get('grpc', {}).get('keepalive_timeout_ms', 5000)),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.max_receive_message_length', cfg.get('grpc', {}).get('max_message_size', 104857600)),
                ('grpc.max_send_message_length', cfg.get('grpc', {}).get('max_message_size', 104857600)),
            ]
            
            target = f'{self.worker_ip}:{self.worker_port}'
            self.channel = grpc.insecure_channel(target, options=options)
            self.stub = grpc_service_pb2_grpc.LLMInferenceStub(self.channel)
            
            log_debug(f"[gRPC Client] Connected to worker at {target}")
            
        except Exception as e:
            log_debug(f"[gRPC Client] Failed to connect to worker: {e}")
            raise
    
    def close(self):
        """Close the gRPC connection."""
        if self.channel:
            self.channel.close()
            log_debug("[gRPC Client] Connection closed")
    
    def send_tensor_and_receive(self, tensor: mx.array, step: int, timeout: float = 30.0) -> mx.array:
        """
        Send a tensor to the worker and receive the processed result.
        
        Args:
            tensor: MLX tensor to send
            step: Generation step number
            timeout: Request timeout in seconds
            
        Returns:
            MLX tensor with the processed result (logits)
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Convert MLX tensor to numpy for serialization
            tensor_np = np.array(tensor)
            
            # Create gRPC request
            request = grpc_service_pb2.TensorRequest(
                request_id=request_id,
                step=step,
                shape=list(tensor_np.shape),
                dtype=tensor_np.dtype.name,
                data=tensor_np.tobytes(),
                timestamp=int(time.time() * 1000)  # milliseconds
            )
            
            log_debug(f"[gRPC Client] Sending tensor request {request_id} for step {step}")
            log_debug(f"[gRPC Client] Tensor shape: {tensor_np.shape}, dtype: {tensor_np.dtype}")
            
            # Send request and receive response
            response = self.stub.ProcessTensor(request, timeout=timeout)
            
            # Convert response back to MLX tensor
            response_shape = tuple(response.shape)
            response_dtype = getattr(np, response.dtype)
            response_np = np.frombuffer(response.data, dtype=response_dtype).reshape(response_shape)
            result_tensor = mx.array(response_np)
            
            processing_time = time.time() - start_time
            
            log_debug(f"[gRPC Client] Received response for {request_id}")
            log_debug(f"[gRPC Client] Response shape: {response_shape}, dtype: {response.dtype}")
            log_debug(f"[gRPC Client] Total round-trip time: {processing_time:.4f}s")
            log_debug(f"[gRPC Client] Worker processing time: {response.processing_time:.4f}ms")
            
            if not response.success:
                raise RuntimeError(f"Worker processing failed: {response.error_message}")
            
            return result_tensor
            
        except grpc.RpcError as e:
            log_debug(f"[gRPC Client] RPC error for request {request_id}: {e}")
            raise
        except Exception as e:
            log_debug(f"[gRPC Client] Error processing request {request_id}: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check worker node health."""
        try:
            request = grpc_service_pb2.HealthRequest(node_id="master")
            response = self.stub.HealthCheck(request, timeout=5.0)
            
            return {
                'status': response.status,
                'memory_usage': response.memory_usage,
                'cpu_usage': response.cpu_usage,
                'uptime': response.uptime,
                'model_loaded': response.model_loaded
            }
        except grpc.RpcError as e:
            log_debug(f"[gRPC Client] Health check failed: {e}")
            return {'status': 'unreachable', 'error': str(e)}


class GRPCBackend(NetworkInterface):
    """gRPC backend implementation for distributed inference."""
    
    def __init__(self):
        self.clients = {}  # dest_rank -> GRPCClient
        self.setup_clients()
    
    def setup_clients(self):
        """Initialize gRPC clients for worker nodes."""
        import json
        import os
        
        # Load nodes configuration
        nodes_file = os.path.join('network', 'nodes.json')
        with open(nodes_file, 'r') as f:
            nodes = json.load(f)
        
        # Create clients for worker nodes
        for i, node in enumerate(nodes):
            if i != 0:  # Skip master node (rank 0)
                try:
                    client = GRPCClient(node['ip'], node.get('grpc_port', 50051))
                    self.clients[i] = client
                    log_debug(f"[gRPC Backend] Initialized client for worker rank {i}")
                except Exception as e:
                    log_debug(f"[gRPC Backend] Failed to initialize client for rank {i}: {e}")
    
    def send_tensor(self, tensor: mx.array, dest_rank: int, step: int = 0, **kwargs) -> mx.array:
        """
        Send tensor to destination rank and receive response.
        
        This is the key method that your generate.py calls:
        logits = network.send_tensor(hidden, 1, step=step)
        """
        if dest_rank not in self.clients:
            raise ValueError(f"No gRPC client available for rank {dest_rank}")
        
        client = self.clients[dest_rank]
        return client.send_tensor_and_receive(tensor, step)
    
    def wait_for_tensor(self, source_rank: int, **kwargs) -> mx.array:
        """
        Wait for tensor from source rank.
        Note: For gRPC, this is handled by send_tensor (request-response pattern)
        """
        raise NotImplementedError("gRPC backend uses request-response pattern. Use send_tensor instead.")
    
    def health_check_all(self) -> Dict[int, Dict[str, Any]]:
        """Check health of all worker nodes."""
        health_status = {}
        for rank, client in self.clients.items():
            health_status[rank] = client.health_check()
        return health_status
    
    def close(self):
        """Close all gRPC connections."""
        for client in self.clients.values():
            client.close()
        self.clients.clear()
        log_debug("[gRPC Backend] All connections closed")
    
    def __del__(self):
        """Cleanup connections on destruction."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup