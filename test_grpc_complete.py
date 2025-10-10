#!/usr/bin/env python3
"""
Test script for gRPC distributed LLM inference.
This script validates the gRPC communication between master and worker nodes.
"""

import sys
import time
import argparse
import numpy as np
import mlx.core as mx
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config as cfg
from utils.utils import log_debug


def test_grpc_worker():
    """Test gRPC worker by starting a test server."""
    print("🔧 Testing gRPC Worker Node...")
    
    try:
        from network.grpc_server import start_grpc_worker_server
        
        print("🚀 Starting gRPC worker server...")
        print(f"📂 Model path: {cfg.get('model_path')}")
        print("📡 Server will start on port 50051")
        print("⏳ Loading model (this may take a while)...")
        
        # Start the gRPC server
        start_grpc_worker_server(
            port=50051,
            model_path=cfg.get('model_path'),
            start_layer=36,
            end_layer=64
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Worker server stopped by user")
    except Exception as e:
        print(f"❌ Worker test failed: {e}")
        import traceback
        traceback.print_exc()


def test_grpc_master():
    """Test gRPC master by connecting to worker and sending test tensor."""
    print("🎯 Testing gRPC Master Node...")
    
    try:
        from network.grpc_client import GRPCClient
        
        # Connect to worker
        print("🔗 Connecting to worker at 192.168.2.11:50051...")
        client = GRPCClient("192.168.2.11", 50051)
        
        # Test health check
        print("🏥 Testing health check...")
        health = client.health_check()
        print(f"📊 Worker status: {health}")
        
        if health.get('status') == 'healthy':
            # Create test tensor (typical hidden state shape)
            print("🧪 Creating test tensor...")
            test_tensor = mx.random.normal((1, 10, 4096))  # batch=1, seq_len=10, hidden_size=4096
            print(f"📤 Test tensor shape: {test_tensor.shape}")
            
            # Send tensor and measure performance
            print("🚀 Sending test tensor to worker...")
            start_time = time.time()
            
            result = client.send_tensor_and_receive(test_tensor, step=0)
            
            end_time = time.time()
            
            print(f"📥 Received result tensor shape: {result.shape}")
            print(f"⏱️  Round-trip time: {(end_time - start_time)*1000:.2f}ms")
            
            # Verify result makes sense (should be logits)
            if len(result.shape) == 3 and result.shape[-1] > test_tensor.shape[-1]:
                print("✅ Result looks correct (logits with vocab_size)")
            else:
                print("⚠️  Result shape might be unexpected")
            
            print("🎉 gRPC master test completed successfully!")
            
        else:
            print("⚠️  Worker not healthy, skipping tensor test")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Master test failed: {e}")
        import traceback
        traceback.print_exc()


def test_grpc_full_system():
    """Test the full gRPC system using the actual inference scripts."""
    print("🧪 Testing Full gRPC System...")
    
    try:
        print("ℹ️  This test requires:")
        print("  1. Worker node running on second Mac Studio (192.168.2.11)")
        print("  2. Model files available on both machines")
        print("  3. Thunderbolt network connectivity")
        print("")
        
        # Test network backend import
        print("📦 Testing network backend import...")
        from network import network
        print("✅ gRPC network backend imported successfully")
        
        # Test configuration
        print("⚙️  Current configuration:")
        print(f"  - Network backend: {cfg.get('network_backend')}")
        print(f"  - Model path: {cfg.get('model_path')}")
        print(f"  - gRPC worker port: {cfg.get('grpc', {}).get('worker_port', 50051)}")
        
        # Test network connectivity
        print("🔗 Testing network connectivity...")
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            result = s.connect_ex(("192.168.2.11", 50051))
            if result == 0:
                print("✅ Network connectivity to worker successful")
            else:
                print("❌ Cannot connect to worker - make sure it's running")
                return
        
        print("🎉 Full system test setup completed!")
        print("")
        print("📋 Next steps:")
        print("1. On Mac Studio 2 (192.168.2.11): python worker_inference_grpc.py")
        print("2. On Mac Studio 1 (192.168.2.10): python master_inference.py")
        
    except Exception as e:
        print(f"❌ Full system test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test gRPC distributed inference")
    parser.add_argument("--mode", choices=["worker", "master", "full"], default="full",
                      help="Test mode: worker, master, or full system")
    
    args = parser.parse_args()
    
    print("🧪 gRPC Distributed LLM Inference Test")
    print("=" * 40)
    
    if args.mode == "worker":
        test_grpc_worker()
    elif args.mode == "master":
        test_grpc_master()
    else:
        test_grpc_full_system()


if __name__ == "__main__":
    main()