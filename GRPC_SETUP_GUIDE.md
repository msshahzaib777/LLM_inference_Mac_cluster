# gRPC Distributed LLM Inference Setup Guide

This guide explains how to run distributed LLM inference using gRPC across two Mac computers for improved performance.

## 🏗️ Architecture Overview

```
Mac 1 (Master Node - Rank 0)          Mac 2 (Worker Node - Rank 1)
┌─────────────────────────────┐       ┌─────────────────────────────┐
│  Model Layers 0-35          │       │  Model Layers 36-64         │
│  ┌─────────────────────┐    │       │  ┌─────────────────────┐    │
│  │ User Interface      │    │       │  │ gRPC Server         │    │
│  │ Token Generation    │    │ gRPC  │  │ (Port 50051)        │    │
│  │ gRPC Client         │◄──►│       │  │ Tensor Processing   │    │
│  └─────────────────────┘    │       │  └─────────────────────┘    │
└─────────────────────────────┘       └─────────────────────────────┘
         192.168.2.1                           192.168.2.2
```

## 📋 Prerequisites

### Both Machines:
1. **Python Environment**: Ensure you have the same Python environment setup
2. **Model Files**: Both machines should have access to the model files
3. **Network Connection**: Machines should be on the same network
4. **Required Packages**: grpcio, grpcio-tools, mlx, transformers, etc.

### Install Dependencies:
```bash
# On both machines
pip install grpcio grpcio-tools psutil
```

## 🔧 Configuration

### 1. Update Network Configuration

Edit `network/nodes.json` with your actual IP addresses:
```json
[
    {
        "ip": "192.168.1.100",    # Mac 1 IP address
        "port": 5001,
        "grpc_port": 50050
    },
    {
        "ip": "192.168.1.101",    # Mac 2 IP address  
        "port": 5001,
        "grpc_port": 50051
    }
]
```

### 2. Update config.yaml
```yaml
model_path: /path/to/your/model
network_backend: grpc
grpc:
  master_port: 50050
  worker_port: 50051
  max_message_size: 104857600  # 100MB
  keepalive_time_ms: 30000
  keepalive_timeout_ms: 5000
```

## 🚀 Running the Distributed Inference

### Method 1: Using the Launch Script (Recommended)

#### On Mac 1 (Master Node):
```bash
cd /path/to/LLM_inference
./launch_grpc.sh start
```

#### On Mac 2 (Worker Node):
```bash
cd /path/to/LLM_inference
RANK=1 ./DeepSeek/bin/python worker_inference_grpc.py
```

### Method 2: Manual Startup

#### Step 1: Start Worker Node (Mac 2)
```bash
cd /path/to/LLM_inference
export RANK=1
./DeepSeek/bin/python worker_inference_grpc.py
```

You should see:
```
=== Worker script started ===
Starting gRPC worker server...
Loading model layers 36-64
Starting gRPC server on [::]:50051
gRPC server started successfully
```

#### Step 2: Start Master Node (Mac 1)
```bash
cd /path/to/LLM_inference  
export RANK=0
./DeepSeek/bin/python master_inference.py
```

You should see:
```
=== Master Script started ===
Loaded tokenizer
Loading first half of the model (layers 0-35)
=== Chatbot started (type 'exit' to quit) ===
```

## 🧪 Testing the Setup

### Quick Connection Test:
```bash
# On Mac 2 (Worker - start first)
cd /path/to/LLM_inference
RANK=1 ./DeepSeek/bin/python test_grpc.py

# On Mac 1 (Master - start second)  
cd /path/to/LLM_inference
RANK=0 ./DeepSeek/bin/python test_grpc.py
```

### Network Connectivity Test:
```bash
# From Mac 1, test if Mac 2 is reachable
telnet 192.168.1.101 50051

# If successful, you should see:
# Trying 192.168.1.101...
# Connected to 192.168.1.101.
```

## 🔍 Monitoring and Troubleshooting

### Check Process Status:
```bash
./launch_grpc.sh status
```

### View Logs:
```bash
./launch_grpc.sh logs
```

### Manual Log Check:
```bash
# Master logs
tail -f logs/debug_log_rank0.txt

# Worker logs  
tail -f logs/debug_log_rank1.txt
```

### Common Issues:

#### 1. **Connection Refused**
```
grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with: status = StatusCode.UNAVAILABLE, details = "failed to connect to all addresses"
```
**Solution**: 
- Check if worker node is running
- Verify IP addresses in `nodes.json`
- Check firewall settings
- Ensure ports are not blocked

#### 2. **Model Not Found**
```
FileNotFoundError: Model path not found
```
**Solution**:
- Verify `model_path` in `config.yaml`
- Ensure both machines have access to model files
- Use absolute paths

#### 3. **Import Errors**
```
ImportError: No module named 'grpc_service_pb2'
```
**Solution**:
```bash
cd network
python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. grpc_service.proto
```

#### 4. **Memory Issues**
```
grpc._channel._InactiveRpcError: Resource exhausted
```
**Solution**:
- Increase `max_message_size` in config.yaml
- Reduce batch size or sequence length

## 📊 Performance Optimization

### Network Optimization:
1. **Use Ethernet**: Prefer wired connection over WiFi
2. **Jumbo Frames**: Enable if both machines support it
3. **Network Interface**: Use fastest available (Thunderbolt > Ethernet > WiFi)

### gRPC Settings:
```yaml
grpc:
  max_message_size: 134217728    # 128MB for larger tensors
  keepalive_time_ms: 15000       # More frequent keepalives
  keepalive_timeout_ms: 3000     # Faster timeout detection
```

### Model Loading:
- Pre-load models before starting inference
- Use model sharding for very large models
- Consider model quantization to reduce transfer size

## 🎯 Expected Performance

With gRPC optimization, you should see:
- **Latency**: 5-20ms per inference step (network dependent)
- **Throughput**: ~10-50 tokens/second (model dependent)
- **Memory**: More efficient than TCP due to connection pooling

## 🛑 Stopping the System

### Graceful Shutdown:
```bash
./launch_grpc.sh stop
```

### Manual Cleanup:
```bash
# Kill processes by PID
kill $(cat master.pid)
kill $(cat worker_*.pid)

# Or force kill
pkill -f "master_inference.py"
pkill -f "worker_inference_grpc.py"
```

## 🎮 Usage Example

Once running, interact with the system:
```
You: Hello, can you explain quantum computing?
Qwen2.5: Quantum computing is a revolutionary computing paradigm...

You: What's the weather like?  
Qwen2.5: I don't have access to real-time weather data...

You: exit
Goodbye!
```

## 📈 Monitoring Performance

The system logs performance metrics:
- **Tokens per second (TPS)**
- **Network round-trip time**  
- **Processing time per step**
- **Memory usage on both nodes**

Look for these in the debug logs to monitor system performance.