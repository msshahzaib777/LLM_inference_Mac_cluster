#!/bin/bash

# Complete gRPC Distributed Inference Setup and Runner
# This script helps you set up and run gRPC-based distributed inference

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH="${SCRIPT_DIR}/DeepSeek/bin/python"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check Python environment
    if [ ! -f "$PYTHON_PATH" ]; then
        error "Python environment not found at $PYTHON_PATH"
        exit 1
    fi
    
    # Check model path
    model_path=$(grep "model_path:" config.yaml | cut -d: -f2 | xargs)
    if [ ! -d "$model_path" ]; then
        error "Model not found at $model_path"
        exit 1
    fi
    
    # Check network backend
    backend=$(grep "network_backend:" config.yaml | cut -d: -f2 | xargs)
    if [ "$backend" != "grpc" ]; then
        warning "Network backend is '$backend', should be 'grpc'"
        info "Updating config to use gRPC backend..."
        sed -i '' 's/network_backend: .*/network_backend: grpc/' config.yaml
    fi
    
    log "Prerequisites check completed"
}

# Get current IP from bridge0 interface
get_current_ip() {
    local ip=$(ifconfig bridge0 2>/dev/null | grep 'inet ' | awk '{print $2}' || echo "")
    if [ -z "$ip" ]; then
        # Fallback to main interface
        ip=$(ifconfig en0 2>/dev/null | grep 'inet ' | awk '{print $2}' || echo "127.0.0.1")
    fi
    echo "$ip"
}

# Test network connectivity
test_connectivity() {
    local target_ip="$1"
    local port="$2"
    
    info "Testing connectivity to $target_ip:$port..."
    
    if command -v nc >/dev/null 2>&1; then
        if nc -z -w3 "$target_ip" "$port" 2>/dev/null; then
            log "✅ Connection to $target_ip:$port successful"
            return 0
        else
            warning "❌ Cannot connect to $target_ip:$port"
            return 1
        fi
    else
        warning "netcat not available, skipping connectivity test"
        return 0
    fi
}

# Start gRPC worker
start_worker() {
    log "Starting gRPC Worker Node..."
    info "This will load the second half of the model (layers 36-64)"
    info "Model loading may take several minutes..."
    
    cd "$SCRIPT_DIR"
    exec "$PYTHON_PATH" worker_inference_grpc.py
}

# Start gRPC master
start_master() {
    log "Starting gRPC Master Node..."
    info "This will load the first half of the model (layers 0-35)"
    info "Make sure the worker node is running on the second Mac Studio"
    
    cd "$SCRIPT_DIR"
    exec "$PYTHON_PATH" master_inference.py
}

# Test gRPC system
test_system() {
    log "Testing gRPC system..."
    cd "$SCRIPT_DIR"
    "$PYTHON_PATH" test_grpc_complete.py --mode full
}

# Show system status
show_status() {
    echo ""
    echo -e "${BLUE}🚀 gRPC Distributed LLM Inference Status${NC}"
    echo "============================================="
    
    local current_ip=$(get_current_ip)
    echo "📍 Current IP (bridge0): $current_ip"
    
    echo ""
    echo "⚙️  Configuration:"
    echo "   Network Backend: $(grep 'network_backend:' config.yaml | cut -d: -f2 | xargs)"
    echo "   Model Path: $(grep 'model_path:' config.yaml | cut -d: -f2 | xargs)"
    echo "   gRPC Worker Port: $(grep 'worker_port:' config.yaml | cut -d: -f2 | xargs)"
    
    echo ""
    echo "🔗 Network Configuration:"
    if [ -f "network/nodes.json" ]; then
        cat network/nodes.json | head -20
    else
        echo "   nodes.json not found"
    fi
    
    echo ""
    echo "🔍 Interface Status:"
    ifconfig bridge0 2>/dev/null | grep -E "(flags|inet)" || echo "   bridge0 not found"
    
    echo ""
    echo "📊 Process Status:"
    pgrep -f "worker_inference_grpc" >/dev/null && echo "   ✅ Worker process running" || echo "   ❌ Worker process not running"
    pgrep -f "master_inference" >/dev/null && echo "   ✅ Master process running" || echo "   ❌ Master process not running"
}

# Configure network for two Mac Studios
configure_network() {
    echo ""
    info "Network Configuration for Two Mac Studios"
    echo ""
    
    local current_ip=$(get_current_ip)
    echo "Current IP detected: $current_ip"
    echo ""
    
    echo "Please enter the IP addresses for your Mac Studios:"
    read -p "Mac Studio 1 (Master) IP [$current_ip]: " master_ip
    master_ip=${master_ip:-$current_ip}
    
    read -p "Mac Studio 2 (Worker) IP: " worker_ip
    if [ -z "$worker_ip" ]; then
        error "Worker IP is required"
        return 1
    fi
    
    # Update nodes.json
    cat > network/nodes.json << EOF
[
    {
        "ip": "$master_ip",
        "port": 5001,
        "grpc_port": 50050,
        "role": "master",
        "interface": "bridge0"
    },
    {
        "ip": "$worker_ip",
        "port": 5002,
        "grpc_port": 50051,
        "role": "worker",
        "interface": "bridge0"
    }
]
EOF
    
    log "Updated network/nodes.json with IPs: $master_ip (master), $worker_ip (worker)"
    
    # Test connectivity
    if test_connectivity "$worker_ip" 22; then
        log "✅ SSH connectivity to worker Mac looks good"
    else
        warning "⚠️  Cannot reach worker Mac. Make sure:"
        echo "   - Thunderbolt connection is active"
        echo "   - Worker Mac is on and accessible"
        echo "   - IP address is correct"
    fi
}

# Show help
show_help() {
    echo ""
    echo -e "${BLUE}=== gRPC Distributed LLM Inference Help ===${NC}"
    echo ""
    echo "This system runs distributed inference across two Mac Studios connected via Thunderbolt."
    echo ""
    echo "🏗️  Architecture:"
    echo "   • Mac Studio 1 (Master): Loads model layers 0-35, handles user interaction"
    echo "   • Mac Studio 2 (Worker): Loads model layers 36-64, processes hidden states"
    echo "   • Communication: High-speed gRPC over Thunderbolt network"
    echo ""
    echo "📋 Setup Steps:"
    echo "   1. Configure network (option 6) with correct IP addresses"
    echo "   2. Copy this project to both Mac Studios"
    echo "   3. On Mac Studio 2: Run as Worker (option 2)"
    echo "   4. On Mac Studio 1: Run as Master (option 1)"
    echo ""
    echo "⚡ Performance Benefits:"
    echo "   • ~2x faster inference using both machines"
    echo "   • High-bandwidth Thunderbolt communication (40+ Gbps)"
    echo "   • Optimized gRPC reduces connection overhead"
    echo "   • Memory distributed across both machines"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   • Use 'Test System' (option 3) to verify setup"
    echo "   • Check 'System Status' (option 4) for configuration"
    echo "   • Ensure model files exist on both machines"
    echo "   • Verify Thunderbolt network connectivity"
    echo ""
}

# Main menu
show_menu() {
    echo ""
    echo -e "${BLUE}🚀 gRPC Distributed LLM Inference${NC}"
    echo "================================="
    echo ""
    echo "1) 🎯 Start Master Node (Mac Studio 1)"
    echo "2) 🔧 Start Worker Node (Mac Studio 2)"
    echo "3) 🧪 Test System"
    echo "4) 📊 Show System Status"
    echo "5) 🔍 Test Network Connectivity"
    echo "6) ⚙️  Configure Network"
    echo "7) 📖 Help & Documentation"
    echo "8) 🚪 Exit"
    echo ""
}

# Main loop
main() {
    cd "$SCRIPT_DIR"
    check_prerequisites
    
    while true; do
        show_menu
        read -p "Enter your choice (1-8): " choice
        
        case $choice in
            1)
                start_master
                ;;
            2)
                start_worker
                ;;
            3)
                test_system
                ;;
            4)
                show_status
                ;;
            5)
                echo ""
                if [ -f "network/nodes.json" ]; then
                    worker_ip=$("$PYTHON_PATH" -c "import json; nodes=json.load(open('network/nodes.json')); print([n for n in nodes if n.get('role')=='worker'][0]['ip'])" 2>/dev/null || echo "")
                    if [ -n "$worker_ip" ]; then
                        test_connectivity "$worker_ip" 50051
                        test_connectivity "$worker_ip" 22
                    else
                        error "Could not determine worker IP from configuration"
                    fi
                else
                    error "Network configuration not found. Please configure network first."
                fi
                ;;
            6)
                configure_network
                ;;
            7)
                show_help
                ;;
            8)
                echo "Goodbye!"
                exit 0
                ;;
            *)
                error "Invalid choice. Please try again."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@"