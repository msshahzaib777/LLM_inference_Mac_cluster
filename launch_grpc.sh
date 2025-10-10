#!/bin/bash

# gRPC-based Distributed LLM Inference Launcher
# This script starts both master and worker nodes for distributed inference

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH="${SCRIPT_DIR}/DeepSeek/bin/python"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check if Python environment exists
if [ ! -f "$PYTHON_PATH" ]; then
    error "Python environment not found at $PYTHON_PATH"
    error "Please ensure the DeepSeek virtual environment is set up"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    error "Config file not found at $CONFIG_FILE"
    exit 1
fi

# Function to start worker node
start_worker() {
    local node_rank=$1
    local node_ip=$2
    local grpc_port=$3
    
    log "Starting worker node $node_rank on $node_ip:$grpc_port"
    
    if [ "$node_ip" = "127.0.0.1" ] || [ "$node_ip" = "localhost" ] || [ "$node_ip" = "192.168.2.1" ]; then
        # Local worker
        RANK=$node_rank $PYTHON_PATH worker_inference_grpc.py &
        local worker_pid=$!
        echo $worker_pid > "worker_${node_rank}.pid"
        log "Started local worker $node_rank (PID: $worker_pid)"
    else
        # Remote worker
        warning "Remote worker startup not implemented. Please start worker manually on $node_ip"
        info "Command to run on $node_ip:"
        info "cd $SCRIPT_DIR && RANK=$node_rank $PYTHON_PATH worker_inference_grpc.py"
    fi
}

# Function to start master node
start_master() {
    log "Starting master node..."
    RANK=0 $PYTHON_PATH master_inference.py &
    local master_pid=$!
    echo $master_pid > "master.pid"
    log "Started master node (PID: $master_pid)"
    return $master_pid
}

# Function to cleanup processes
cleanup() {
    log "Cleaning up processes..."
    
    if [ -f "master.pid" ]; then
        local master_pid=$(cat master.pid)
        if kill -0 $master_pid 2>/dev/null; then
            kill $master_pid
            log "Stopped master process (PID: $master_pid)"
        fi
        rm -f master.pid
    fi
    
    for worker_pid_file in worker_*.pid; do
        if [ -f "$worker_pid_file" ]; then
            local worker_pid=$(cat "$worker_pid_file")
            if kill -0 $worker_pid 2>/dev/null; then
                kill $worker_pid
                log "Stopped worker process (PID: $worker_pid)"
            fi
            rm -f "$worker_pid_file"
        fi
    done
}

# Trap cleanup on exit
trap cleanup EXIT INT TERM

# Parse command line arguments
case "${1:-start}" in
    "start")
        log "Starting gRPC-based distributed LLM inference"
        
        # Read node configuration
        info "Reading node configuration from nodes.json"
        
        # Start worker nodes first
        if command -v jq >/dev/null 2>&1; then
            # Use jq if available
            worker_count=$(jq length network/nodes.json)
            for ((i=1; i<$worker_count; i++)); do
                node_ip=$(jq -r ".[$i].ip" network/nodes.json)
                grpc_port=$(jq -r ".[$i].grpc_port" network/nodes.json)
                start_worker $i $node_ip $grpc_port
            done
        else
            # Fallback for common configurations
            warning "jq not found, using fallback configuration"
            start_worker 1 "192.168.2.2" 50051
        fi
        
        # Wait a moment for workers to start
        sleep 3
        
        # Start master node
        master_pid=$(start_master)
        
        log "All nodes started. Master PID: $master_pid"
        log "Press Ctrl+C to stop all processes"
        
        # Wait for master to finish
        wait $master_pid
        ;;
        
    "stop")
        log "Stopping all processes"
        cleanup
        ;;
        
    "status")
        info "Checking process status"
        
        if [ -f "master.pid" ]; then
            master_pid=$(cat master.pid)
            if kill -0 $master_pid 2>/dev/null; then
                log "Master process running (PID: $master_pid)"
            else
                warning "Master process not running"
            fi
        else
            warning "No master PID file found"
        fi
        
        for worker_pid_file in worker_*.pid; do
            if [ -f "$worker_pid_file" ]; then
                worker_pid=$(cat "$worker_pid_file")
                worker_rank=$(echo "$worker_pid_file" | sed 's/worker_\(.*\)\.pid/\1/')
                if kill -0 $worker_pid 2>/dev/null; then
                    log "Worker $worker_rank running (PID: $worker_pid)"
                else
                    warning "Worker $worker_rank not running"
                fi
            fi
        done
        ;;
        
    "logs")
        info "Showing recent logs"
        if [ -f "logs/debug_log_rank0.txt" ]; then
            echo -e "\n${BLUE}=== Master Logs ===${NC}"
            tail -20 logs/debug_log_rank0.txt
        fi
        
        if [ -f "logs/debug_log_rank1.txt" ]; then
            echo -e "\n${BLUE}=== Worker Logs ===${NC}"
            tail -20 logs/debug_log_rank1.txt
        fi
        ;;
        
    "help"|*)
        echo "gRPC Distributed LLM Inference Launcher"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start   - Start distributed inference (default)"
        echo "  stop    - Stop all processes"
        echo "  status  - Check process status"
        echo "  logs    - Show recent logs"
        echo "  help    - Show this help message"
        echo ""
        echo "Configuration:"
        echo "  - Edit config.yaml to configure model path and network settings"
        echo "  - Edit network/nodes.json to configure node IPs and ports"
        echo ""
        ;;
esac