#!/bin/bash
# Usage: ./launch.sh worker1 | worker2 | controller

ROLE=$1

if [[ -z "$ROLE" ]]; then
  echo "Usage: ./launch.sh [worker1|worker2|controller]"
  exit 1
fi

export MASTER_ADDR=192.168.2.1
export MASTER_PORT=29500
export WORLD_SIZE=3

if [[ "$ROLE" == "worker1" ]]; then
  export RANK=0
elif [[ "$ROLE" == "worker2" ]]; then
  export RANK=1
elif [[ "$ROLE" == "controller" ]]; then
  export RANK=2
else
  echo "Invalid role. Choose worker1, worker2, or controller."
  exit 1
fi

echo "🚀 Launching $ROLE (rank $RANK)..."
python3 $ROLE.py
