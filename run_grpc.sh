#!/bin/bash

# Simple gRPC Distributed LLM Inference Runner
# This script helps you start the distributed inference system easily

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH="${SCRIPT_DIR}/DeepSeek/bin/python"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 gRPC Distributed LLM Inference Runner${NC}"
echo "=========================================="
echo ""

# Check Python environment
if [ ! -f "$PYTHON_PATH" ]; then
    echo -e "${RED}❌ Python environment not found at $PYTHON_PATH${NC}"
    exit 1
fi

# Get the current IP address
CURRENT_IP=$(ifconfig | grep -A 1 "en0\|en1\|bridge0" | grep "inet " | head -1 | awk '{print $2}')
echo -e "${BLUE}📍 Current IP Address: $CURRENT_IP${NC}"

echo ""
echo "Select your role:"
echo "1) Master Node (Mac 1) - Handles user interaction and first half of model"
echo "2) Worker Node (Mac 2) - Handles second half of model processing"
echo "3) Test gRPC Connection"
echo "4) Show Setup Guide"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo -e "${GREEN}🎯 Starting as Master Node (Rank 0)${NC}"
        echo "Make sure the worker node is running first!"
        echo ""
        read -p "Press Enter when worker node is ready..."
        
        cd "$SCRIPT_DIR"
        export RANK=0
        exec $PYTHON_PATH master_inference.py
        ;;
        
    2)
        echo -e "${GREEN}🔧 Starting as Worker Node (Rank 1)${NC}"
        echo "This will start the gRPC server and wait for connections"
        echo ""
        
        cd "$SCRIPT_DIR"
        export RANK=1
        exec $PYTHON_PATH worker_inference_grpc.py
        ;;
        
    3)
        echo -e "${YELLOW}🧪 Testing gRPC Connection${NC}"
        echo ""
        echo "Choose test mode:"
        echo "  1) Test as Master Node (requires worker running)"  
        echo "  2) Test as Worker Node (dummy server)"
        echo ""
        
        read -p "Enter choice (1-2): " test_choice
        cd "$SCRIPT_DIR"
        
        case $test_choice in
            1)
                echo "Testing as Master Node..."
                export RANK=0
                exec $PYTHON_PATH test_grpc.py
                ;;
            2)
                echo "Testing as Worker Node..."
                export RANK=1
                exec $PYTHON_PATH test_grpc.py
                ;;
            *)
                echo "Invalid choice. Defaulting to Master test..."
                export RANK=0
                exec $PYTHON_PATH test_grpc.py
                ;;
        esac
        ;;
        
    4)
        echo -e "${BLUE}📖 Setup Guide${NC}"
        echo ""
        echo "=== Quick Setup ==="
        echo "1. Update network/nodes.json with your Mac IP addresses"
        echo "2. Make sure both Macs have the same model files"
        echo "3. Start worker node first: $0 -> Choice 2"
        echo "4. Start master node second: $0 -> Choice 1" 
        echo ""
        echo "=== Network Configuration ==="
        echo "Edit network/nodes.json:"
        echo "["
        echo "    {"
        echo "        \"ip\": \"$CURRENT_IP\",    # Mac 1 IP"
        echo "        \"port\": 5001,"
        echo "        \"grpc_port\": 50050"
        echo "    },"
        echo "    {"
        echo "        \"ip\": \"<OTHER_MAC_IP>\",    # Mac 2 IP"
        echo "        \"port\": 5001,"
        echo "        \"grpc_port\": 50051"
        echo "    }"
        echo "]"
        echo ""
        echo "For detailed setup, see GRPC_SETUP_GUIDE.md"
        ;;
        
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac