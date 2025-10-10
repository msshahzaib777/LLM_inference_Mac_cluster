#!/usr/bin/env python3
"""
Test script for gRPC distributed LLM inference.
This script tests the gRPC communication between master and worker nodes.
"""

import os
import sys
import time
import numpy as np
import mlx.core as mx
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import config as cfg
from utils.utils import log_debug


def test_grpc_connection():
    """Test basic gRPC connection and communication."""
    print("🧪 Testing gRPC Connection...")
    
    # Check if we're running as master (rank 0) or worker (rank 1)
    rank_env = os.environ.get('RANK', '0')
    if rank_env == '':
        rank_env = '0'
    rank = int(rank_env)
    
    if rank == 0:
        test_master_node()
    elif rank == 1:
        test_worker_node()
    else:
        print(f"❌ Invalid rank: {rank}. Use RANK=0 for master, RANK=1 for worker")
        sys.exit(1)


def test_master_node():
    """Test master node functionality."""
    print("🎯 Running as MASTER node (rank 0)")
    
    try:
        # Import network backend
        from network import network
        print("✅ Network backend imported successfully")
        
        # Create a test tensor
        test_tensor = mx.random.normal((1, 10, 4096))  # Typical hidden state shape
        print(f"✅ Created test tensor: shape={test_tensor.shape}")
        
        # Test sending tensor to worker
        print("📤 Sending test tensor to worker node...")
        start_time = time.time()
        
        result = network.send_tensor(test_tensor, dest_rank=1, step=0)
        
        end_time = time.time()
        print(f"✅ Received result from worker: shape={result.shape}")
        print(f"⏱️  Round-trip time: {(end_time - start_time)*1000:.2f}ms")
        
        # Verify the result makes sense
        if result.shape[-1] > test_tensor.shape[-1]:  # Should be logits (vocab_size > hidden_size)
            print("✅ Result shape looks correct (logits)")
        else:
            print("⚠️  Result shape might be incorrect")
            
        print("🎉 Master node test completed successfully!")
        
    except Exception as e:
        print(f"❌ Master node test failed: {e}")
        import traceback
        traceback.print_exc()


def test_worker_node():
    """Test worker node functionality."""
    print("🔧 Running as WORKER node (rank 1)")
    
    try:
        # Start the gRPC server
        from network.grpc_server import start_grpc_worker_server
        
        print("🚀 Starting gRPC worker server...")
        
        # Use test configuration
        port = cfg.get('grpc', {}).get('worker_port', 50051)
        model_path = cfg.get('model_path')
        
        if not model_path or not os.path.exists(model_path):
            print(f"⚠️  Model path not found: {model_path}")
            print("📝 Using dummy model for testing...")
            
            # Create a simple test server without loading actual model
            test_worker_server(port)
        else:
            print(f"📂 Using model: {model_path}")
            start_grpc_worker_server(port, model_path, 36, 64)
            
    except Exception as e:
        print(f"❌ Worker node test failed: {e}")
        import traceback
        traceback.print_exc()


def test_worker_server(port):
    """Start a test worker server with dummy model."""
    import grpc
    from concurrent import futures
    from network.grpc_server import LLMInferenceServicer
    from network import grpc_service_pb2_grpc
    
    class TestServicer(grpc_service_pb2_grpc.LLMInferenceServicer):
        def ProcessTensor(self, request, context):
            print(f"📨 Received tensor request: {len(request.data)} bytes")
            
            # Create dummy response (simulate logits)
            import numpy as np
            from network import grpc_service_pb2
            
            # Simulate processing
            time.sleep(0.01)  # 10ms processing time
            
            # Create fake logits (vocab_size = 151936 for Qwen2.5)
            fake_logits = np.random.normal(0, 1, (1, 1, 151936)).astype(np.float32)
            
            response = grpc_service_pb2.TensorResponse()
            response.shape.extend(fake_logits.shape)
            response.dtype = fake_logits.dtype.name
            response.data = fake_logits.tobytes()
            response.request_id = request.request_id
            response.step = request.step
            response.processing_time = 10.0  # ms
            
            print(f"📤 Sending response: shape={fake_logits.shape}")
            return response
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grpc_service_pb2_grpc.add_LLMInferenceServicer_to_server(TestServicer(), server)
    
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    
    print(f"🎧 Test server listening on {listen_addr}")
    server.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("🛑 Stopping test server...")
        server.stop(10)


if __name__ == "__main__":
    test_grpc_connection()