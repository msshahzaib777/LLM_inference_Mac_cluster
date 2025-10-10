"""
gRPC server implementation for distributed LLM inference worker nodes.
This server handles tensor processing requests from the master node.
"""

import grpc
import time
import asyncio
import numpy as np
import mlx.core as mx
from concurrent import futures
from typing import Dict, Any

from . import grpc_service_pb2
from . import grpc_service_pb2_grpc
from utils.utils import log_debug, load_model
from config import config as cfg


class LLMInferenceServicer(grpc_service_pb2_grpc.LLMInferenceServicer):
    """gRPC servicer for handling distributed LLM inference requests."""
    
    def __init__(self, model_path: str, start_layer: int, end_layer: int):
        """Initialize the servicer with model configuration."""
        self.model_path = model_path
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the model partition for this worker."""
        log_debug(f"Loading model layers {self.start_layer}-{self.end_layer}")
        self.model = load_model(self.model_path, self.start_layer, self.end_layer)
        log_debug("Model loaded successfully")
    
    def _tensor_from_request(self, request: grpc_service_pb2.TensorRequest) -> mx.array:
        """Convert gRPC tensor request to MLX tensor."""
        shape = tuple(request.shape)
        dtype = getattr(np, request.dtype)
        
        # Deserialize tensor data
        tensor_np = np.frombuffer(request.data, dtype=dtype).reshape(shape)
        tensor_mx = mx.array(tensor_np)
        
        return tensor_mx
    
    def _tensor_to_response(self, tensor: mx.array, request_id: str, step: int, processing_time: float) -> grpc_service_pb2.TensorResponse:
        """Convert MLX tensor to gRPC tensor response."""
        tensor_np = np.array(tensor)
        
        response = grpc_service_pb2.TensorResponse()
        response.shape.extend(tensor_np.shape)
        response.dtype = tensor_np.dtype.name
        response.data = tensor_np.tobytes()
        response.request_id = request_id
        response.step = step
        response.processing_time = processing_time
        
        return response
    
    def ProcessTensor(self, request: grpc_service_pb2.TensorRequest, context) -> grpc_service_pb2.TensorResponse:
        """Process a single tensor request."""
        start_time = time.time()
        
        try:
            log_debug(f"Processing tensor request {request.request_id} for step {request.step}")
            
            # Convert request to MLX tensor
            input_tensor = self._tensor_from_request(request)
            log_debug(f"Received tensor: shape={input_tensor.shape}, dtype={input_tensor.dtype}")
            
            # Forward pass through model partition
            output_tensor = self.model(input_tensor)
            log_debug(f"Computed output: shape={output_tensor.shape}, dtype={output_tensor.dtype}")
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Convert output to response
            response = self._tensor_to_response(output_tensor, request.request_id, request.step, processing_time)
            
            log_debug(f"Processed request {request.request_id} in {processing_time:.2f}ms")
            return response
            
        except Exception as e:
            log_debug(f"Error processing tensor: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Processing error: {str(e)}")
            return grpc_service_pb2.TensorResponse()
    
    def ProcessTensorStream(self, request_iterator, context):
        """Process streaming tensor requests for better performance."""
        for request in request_iterator:
            response = self.ProcessTensor(request, context)
            yield response
    
    def HealthCheck(self, request: grpc_service_pb2.HealthRequest, context) -> grpc_service_pb2.HealthResponse:
        """Handle health check requests."""
        import psutil
        
        response = grpc_service_pb2.HealthResponse()
        response.node_id = request.node_id
        response.status = "healthy"
        response.memory_usage = psutil.virtual_memory().percent
        response.cpu_usage = psutil.cpu_percent()
        
        return response


class GRPCServer:
    """gRPC server for distributed LLM inference."""
    
    def __init__(self, port: int, model_path: str, start_layer: int, end_layer: int, max_workers: int = 10):
        self.port = port
        self.server = None
        self.servicer = LLMInferenceServicer(model_path, start_layer, end_layer)
        self.max_workers = max_workers
    
    def start(self):
        """Start the gRPC server."""
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=[
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.http2.min_ping_interval_without_data_ms', 300000),
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
                ('grpc.max_send_message_length', 100 * 1024 * 1024),     # 100MB
            ]
        )
        
        grpc_service_pb2_grpc.add_LLMInferenceServicer_to_server(self.servicer, self.server)
        listen_addr = f'[::]:{self.port}'
        self.server.add_insecure_port(listen_addr)
        
        log_debug(f"Starting gRPC server on {listen_addr}")
        self.server.start()
        log_debug("gRPC server started successfully")
        
        return self.server
    
    def stop(self, grace_period: int = 10):
        """Stop the gRPC server."""
        if self.server:
            log_debug("Stopping gRPC server...")
            self.server.stop(grace_period)
            log_debug("gRPC server stopped")
    
    def wait_for_termination(self):
        """Wait for server termination."""
        if self.server:
            self.server.wait_for_termination()


def start_grpc_worker_server(port: int = None, model_path: str = None, start_layer: int = 36, end_layer: int = 64):
    """Start gRPC server for worker node."""
    if port is None:
        port = cfg.get('grpc', {}).get('worker_port', 50051)
    
    if model_path is None:
        model_path = cfg.get('model_path')
    
    server = GRPCServer(port, model_path, start_layer, end_layer)
    grpc_server = server.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        log_debug("Received interrupt signal")
        server.stop()


if __name__ == "__main__":
    start_grpc_worker_server()