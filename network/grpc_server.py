"""
gRPC server implementation for distributed LLM inference worker nodes.
This server receives hidden states from master node and returns logits.
"""

import grpc
import time
import os
import psutil
import numpy as np
import mlx.core as mx
from concurrent import futures
from typing import Optional

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
        self.worker_id = f"worker_{os.getpid()}"
        self.startup_time = time.time()
        self._load_model()
    
    def _load_model(self):
        """Load the model partition for this worker."""
        log_debug(f"[gRPC Server] Loading model layers {self.start_layer}-{self.end_layer}")
        try:
            self.model = load_model(self.model_path, self.start_layer, self.end_layer)
            log_debug(f"[gRPC Server] Model loaded successfully")
        except Exception as e:
            log_debug(f"[gRPC Server] Failed to load model: {e}")
            raise
    
    def _tensor_from_request(self, request: grpc_service_pb2.TensorRequest) -> mx.array:
        """Convert gRPC tensor request to MLX tensor."""
        try:
            shape = tuple(request.shape)
            dtype = getattr(np, request.dtype)
            
            # Deserialize tensor data
            tensor_np = np.frombuffer(request.data, dtype=dtype).reshape(shape)
            tensor_mx = mx.array(tensor_np)
            
            return tensor_mx
        except Exception as e:
            log_debug(f"[gRPC Server] Error converting request tensor: {e}")
            raise
    
    def _tensor_to_response(self, tensor: mx.array, request: grpc_service_pb2.TensorRequest, 
                          processing_time: float, success: bool = True, 
                          error_message: str = "") -> grpc_service_pb2.TensorResponse:
        """Convert MLX tensor to gRPC tensor response."""
        try:
            if success and tensor is not None:
                tensor_np = np.array(tensor)
                
                response = grpc_service_pb2.TensorResponse(
                    request_id=request.request_id,
                    step=request.step,
                    shape=list(tensor_np.shape),
                    dtype=tensor_np.dtype.name,
                    data=tensor_np.tobytes(),
                    processing_time=processing_time,
                    worker_id=self.worker_id,
                    success=True,
                    error_message=""
                )
            else:
                # Return error response
                response = grpc_service_pb2.TensorResponse(
                    request_id=request.request_id,
                    step=request.step,
                    shape=[],
                    dtype="",
                    data=b"",
                    processing_time=processing_time,
                    worker_id=self.worker_id,
                    success=False,
                    error_message=error_message
                )
            
            return response
        except Exception as e:
            log_debug(f"[gRPC Server] Error creating response: {e}")
            # Return error response
            return grpc_service_pb2.TensorResponse(
                request_id=request.request_id,
                step=request.step,
                shape=[],
                dtype="",
                data=b"",
                processing_time=processing_time,
                worker_id=self.worker_id,
                success=False,
                error_message=str(e)
            )
    
    def ProcessTensor(self, request: grpc_service_pb2.TensorRequest, context) -> grpc_service_pb2.TensorResponse:
        """Process a single tensor request."""
        start_time = time.time()
        
        try:
            log_debug(f"[gRPC Server] Processing tensor request {request.request_id} for step {request.step}")
            
            # Convert request to MLX tensor
            input_tensor = self._tensor_from_request(request)
            log_debug(f"[gRPC Server] Received tensor: shape={input_tensor.shape}, dtype={input_tensor.dtype}")
            
            # Forward pass through model partition
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            output_tensor = self.model(input_tensor)
            log_debug(f"[gRPC Server] Computed output: shape={output_tensor.shape}, dtype={output_tensor.dtype}")
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Convert output to response
            response = self._tensor_to_response(output_tensor, request, processing_time)
            
            log_debug(f"[gRPC Server] Processed request {request.request_id} in {processing_time:.2f}ms")
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            log_debug(f"[gRPC Server] Error processing tensor {request.request_id}: {e}")
            
            # Set gRPC error status
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Processing error: {str(e)}")
            
            # Return error response
            return self._tensor_to_response(None, request, processing_time, False, str(e))
    
    def ProcessTensorStream(self, request_iterator, context):
        """Process streaming tensor requests for better performance."""
        try:
            for request in request_iterator:
                response = self.ProcessTensor(request, context)
                yield response
        except Exception as e:
            log_debug(f"[gRPC Server] Error in tensor stream: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Stream processing error: {str(e)}")
    
    def HealthCheck(self, request: grpc_service_pb2.HealthRequest, context) -> grpc_service_pb2.HealthResponse:
        """Handle health check requests."""
        try:
            # Get system statistics
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=0.1)
            uptime = int(time.time() - self.startup_time)
            
            # Determine status
            status = "healthy"
            if memory_percent > 90 or cpu_percent > 95:
                status = "degraded"
            if self.model is None:
                status = "unhealthy"
            
            response = grpc_service_pb2.HealthResponse(
                node_id=self.worker_id,
                status=status,
                memory_usage=memory_percent,
                cpu_usage=cpu_percent,
                uptime=uptime,
                model_loaded=f"layers_{self.start_layer}-{self.end_layer}" if self.model else "not_loaded"
            )
            
            log_debug(f"[gRPC Server] Health check: {status}, Memory: {memory_percent:.1f}%, CPU: {cpu_percent:.1f}%")
            return response
            
        except Exception as e:
            log_debug(f"[gRPC Server] Error in health check: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Health check error: {str(e)}")
            
            return grpc_service_pb2.HealthResponse(
                node_id=self.worker_id,
                status="unhealthy",
                memory_usage=0,
                cpu_usage=0,
                uptime=0,
                model_loaded="error"
            )


class GRPCServer:
    """gRPC server for distributed LLM inference."""
    
    def __init__(self, port: int, model_path: str, start_layer: int, end_layer: int, max_workers: int = 10):
        self.port = port
        self.server = None
        self.servicer = LLMInferenceServicer(model_path, start_layer, end_layer)
        self.max_workers = max_workers
    
    def start(self):
        """Start the gRPC server."""
        try:
            # Configure server options for high throughput
            options = [
                ('grpc.keepalive_time_ms', cfg.get('grpc', {}).get('keepalive_time_ms', 30000)),
                ('grpc.keepalive_timeout_ms', cfg.get('grpc', {}).get('keepalive_timeout_ms', 5000)),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.http2.min_ping_interval_without_data_ms', 300000),
                ('grpc.max_receive_message_length', cfg.get('grpc', {}).get('max_message_size', 104857600)),
                ('grpc.max_send_message_length', cfg.get('grpc', {}).get('max_message_size', 104857600)),
            ]
            
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=options
            )
            
            grpc_service_pb2_grpc.add_LLMInferenceServicer_to_server(self.servicer, self.server)
            listen_addr = f'[::]:{self.port}'
            self.server.add_insecure_port(listen_addr)
            
            log_debug(f"[gRPC Server] Starting server on {listen_addr}")
            self.server.start()
            log_debug(f"[gRPC Server] Server started successfully")
            
            return self.server
            
        except Exception as e:
            log_debug(f"[gRPC Server] Failed to start server: {e}")
            raise
    
    def stop(self, grace_period: int = 10):
        """Stop the gRPC server."""
        if self.server:
            log_debug(f"[gRPC Server] Stopping server...")
            self.server.stop(grace_period)
            log_debug(f"[gRPC Server] Server stopped")
    
    def wait_for_termination(self):
        """Wait for server termination."""
        if self.server:
            try:
                self.server.wait_for_termination()
            except KeyboardInterrupt:
                log_debug(f"[gRPC Server] Received interrupt signal")
                self.stop()


def start_grpc_worker_server(port: Optional[int] = None, model_path: Optional[str] = None, 
                           start_layer: int = 36, end_layer: int = 64):
    """Start gRPC server for worker node."""
    if port is None:
        port = cfg.get('grpc', {}).get('worker_port', 50051)
    
    if model_path is None:
        model_path = cfg.get('model_path')
    
    log_debug(f"[gRPC Server] Starting worker server on port {port}")
    log_debug(f"[gRPC Server] Model path: {model_path}")
    log_debug(f"[gRPC Server] Layers: {start_layer}-{end_layer}")
    
    server = GRPCServer(port, model_path, start_layer, end_layer)
    grpc_server = server.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        log_debug(f"[gRPC Server] Received interrupt signal")
        server.stop()


if __name__ == "__main__":
    start_grpc_worker_server()