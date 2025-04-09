#!/bin/bash

# Usage:
# On Mac 1: ./run_distributed.sh 0
# On Mac 2: ./run_distributed.sh 1

# === Configuration ===
MASTER_ADDR=192.168.2.1   # IP of Mac 1 (acts as master)
MASTER_PORT=29500         # Use any free port
NNODES=2                  # Total number of Macs
NPROC_PER_NODE=1          # One process per Mac
BACKEND=gloo              # Must use gloo for macOS (no CUDA)

# === Read NODE_RANK from CLI or ENV ===
if [ -z "$1" ]; then
  echo "Usage: $0 <node_rank> (e.g. 0 for Mac1, 1 for Mac2)"
  exit 1
fi

NODE_RANK=$1

# === Prepare Dataset ===
mkdir -p data
cd data
if [ ! -d "cifar-10-batches-py" ]; then
  echo "Downloading CIFAR-10 dataset..."
  wget -c --quiet https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  tar -xvzf cifar-10-python.tar.gz
fi
cd ..
mkdir -p saved_models

# === Run Distributed Training ===
echo "Starting training on node_rank=$NODE_RANK ..."
torchrun \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  resnet_ddp.py --backend $BACKEND
