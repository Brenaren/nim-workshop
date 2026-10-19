#!/bin/bash
# Simple NeMo Container Start Script (No SSH Required)

export SCRIPTS_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
echo "SCRIPTS_DIR: $SCRIPTS_DIR"

# Set default values for environment variables
export PROJECT_DIR="${PROJECT_DIR:-$SCRIPTS_DIR/..}"
export CONTAINER_NAME="${CONTAINER_NAME:-docker_nemo}"

echo "PROJECT_DIR: $PROJECT_DIR"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "Starting NeMo container (no SSH required)..."

# Kill any existing container if running
docker kill $CONTAINER_NAME 2>/dev/null || true

# Ensure Jupyter runtime dir exists, if not, create it
HOST_RUNTIME="$HOME/.local/share/jupyter/runtime"
if [[ ! -d "$HOST_RUNTIME" ]]; then
  mkdir -p $HOST_RUNTIME
fi

# Get HF token from ../docker/.env file.  Source all the variables in the file
source ${PROJECT_DIR}/docker/.env


# Start the container - just keep it running with a command
# Use host networking so kernel can bind to host's 127.0.0.1
# Add python path to be PYTHONPATH=PYTHONPATH:./scripts
docker run -d \
  --name $CONTAINER_NAME \
  --rm \
  --user ubuntu \
  --gpus all \
  --network host \
  --shm-size=16GB \
  -v /etc/resolv.conf:/etc/resolv.conf:ro \
  -v "${HOST_RUNTIME}:${HOST_RUNTIME}" \
  -v "${PROJECT_DIR}:/workspace/" \
  -e PYTHONPATH=/opt/NeMo/:/workspace/scripts:/opt/NeMo-Framework-Launcher/launcher_scripts:/opt/megatron-lm \
  -e HF_TOKEN=$HF_TOKEN \
  nemo-nim-workshop:latest \
  sleep infinity

echo "[start_nemo] Container ${CONTAINER_NAME} is running."
echo "[start_nemo] No SSH required - using direct docker exec."
echo "[start_nemo] Select 'Python (NeMo Container)' kernel in Jupyter."

# Test if container is working
if docker exec $CONTAINER_NAME python --version; then
    echo "✓ Container is ready!"
else
    echo "✗ Container test failed"
    exit 1
fi 
