"""
gRPC client implementation for distributed LLM inference master node.
Provides high-performance communication with worker nodes.
"""

import grpc
import time
import uuid
import numpy as np
import mlx.core as mx
from typing import List, Optional, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor

from . import grpc_service_pb2
from . import grpc_service_pb2_grpc
from utils.utils import log_debug
from .interface import NetworkInterface


class GRPCClient:
    """High-performance gRPC client for communicating with worker nodes."""
    
    def __init__(self, host: str, port: int, max_retries: int = 3):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.channel = None
        self.stub = None
        self._connect()
    
    def _connect(self):
        """Establish connection to gRPC server."""
        options = [
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 300000),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_send_message_length', 100 * 1024 * 1024),     # 100MB
        ]
        
        self.channel = grpc.insecure_channel(f'{self.host}:{self.port}', options=options)
        self.stub = grpc_service_pb2_grpc.LLMInferenceStub(self.channel)
        
        # Test connection
        try:
            self._health_check()
            log_debug(f"Connected to gRPC server at {self.host}:{self.port}")
        except grpc.RpcError as e:
            log_debug(f"Failed to connect to gRPC server: {e}")
            raise
    
    def _health_check(self) -> bool:
        """Perform health check on the server."""
        request = grpc_service_pb2.HealthRequest(node_id=f"{self.host}:{self.port}")
        try:
            response = self.stub.HealthCheck(request, timeout=5.0)
            log_debug(f"Health check: {response.status}, Memory: {response.memory_usage:.1f}%, CPU: {response.cpu_usage:.1f}%")
            return response.status == "healthy"
        except grpc.RpcError:
            return False
    
    def _tensor_to_request(self, tensor: mx.array, step: int) -> grpc_service_pb2.TensorRequest:
        """Convert MLX tensor to gRPC request."""
        tensor_np = np.array(tensor)
        
        request = grpc_service_pb2.TensorRequest()
        request.shape.extend(tensor_np.shape)
        request.dtype = tensor_np.dtype.name
        request.data = tensor_np.tobytes()
        request.step = step
        request.request_id = str(uuid.uuid4())
        
        return request
    
    def _response_to_tensor(self, response: grpc_service_pb2.TensorResponse) -> mx.array:
        """Convert gRPC response to MLX tensor."""
        shape = tuple(response.shape)
        dtype = getattr(np, response.dtype)
        
        tensor_np = np.frombuffer(response.data, dtype=dtype).reshape(shape)
        tensor_mx = mx.array(tensor_np)
        
        return tensor_mx
    
    def send_tensor_and_receive(self, tensor: mx.array, step: int = 0, timeout: float = 30.0) -> mx.array:
        """Send tensor and receive processed result."""
        request = self._tensor_to_request(tensor, step)
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self.stub.ProcessTensor(request, timeout=timeout)
                network_time = time.time() - start_time
                
                log_debug(f"gRPC call completed in {network_time*1000:.2f}ms (processing: {response.processing_time:.2f}ms)")
                
                return self._response_to_tensor(response)
                
            except grpc.RpcError as e:
                log_debug(f"gRPC error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
    
    def close(self):
        """Close the gRPC connection."""
        if self.channel:
            self.channel.close()
            log_debug(f"Closed connection to {self.host}:{self.port}")


class GRPCBackend(NetworkInterface):
    """gRPC-based network backend for distributed LLM inference."""
    
    def __init__(self):
        self.clients: Dict[int, GRPCClient] = {}
        self.node_configs = self._load_node_configs()
        self._initialize_clients()
    
    def _load_node_configs(self) -> Dict[int, Dict]:
        """Load node configurations from config."""
        import json
        import os
        
        file_path = os.path.abspath('network/nodes.json')
        with open(file_path, 'r') as file:
            nodes = json.load(file)
        
        # Convert list to dict with node_id as key
        return {i: node for i, node in enumerate(nodes)}
    
    def _initialize_clients(self):
        """Initialize gRPC clients for all worker nodes."""
        for node_id, config in self.node_configs.items():
            if node_id != 0:  # Skip master node
                try:
                    # Use gRPC port instead of TCP port
                    grpc_port = config.get('grpc_port', config['port'] + 1000)  # Default offset
                    client = GRPCClient(config['ip'], grpc_port)
                    self.clients[node_id] = client
                    log_debug(f"Initialized gRPC client for node {node_id}")
                except Exception as e:
                    log_debug(f"Failed to initialize client for node {node_id}: {e}")
    
    def send_tensor(self, tensor: mx.array, dest_rank: int, tag: int = 0, **kwargs):
        """Send tensor to destination rank and receive result."""
        if dest_rank not in self.clients:
            raise ValueError(f"No client available for rank {dest_rank}")
        
        client = self.clients[dest_rank]
        step = kwargs.get('step', 0)
        
        log_debug(f"Sending tensor to rank {dest_rank}: shape={tensor.shape}")
        result = client.send_tensor_and_receive(tensor, step)
        log_debug(f"Received result from rank {dest_rank}: shape={result.shape}")
        
        # Store result for wait_for_tensor call
        self._last_result = result
        
        return result
    
    def wait_for_tensor(self, source_rank: int, **kwargs) -> mx.array:
        """Wait for tensor from source rank (returns cached result from send_tensor)."""
        # In gRPC backend, this returns the result from the last send_tensor call
        # since gRPC is synchronous request-response
        if hasattr(self, '_last_result'):
            result = self._last_result
            delattr(self, '_last_result')
            return result
        else:
            raise RuntimeError("No tensor result available. Call send_tensor first.")
    
    def close(self):
        """Close all gRPC connections."""
        for client in self.clients.values():
            client.close()
        self.clients.clear()
        log_debug("Closed all gRPC connections")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.close()