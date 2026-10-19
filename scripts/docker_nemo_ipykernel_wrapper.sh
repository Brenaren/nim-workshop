#!/bin/bash
# This script is used to launch the ipykernel on a running NeMo container.
# 1.  run build_nemo.sh to build nemo container
# 2.  run start_nemo.sh to launch nemo container
# 3.  run start_jupyter.sh
# 4.  select this ipykernel 
#     configured -> ~/.local/share/jupyter/kernels/docker-nemo-exec/kernel.json
# Docker Exec Kernel Wrapper

CONTAINER_NAME="docker_nemo"

# Parse arguments
CONNECTION_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--connection-file)
            CONNECTION_FILE="$2"
            shift 2
            ;;
        *)
            # Pass through other arguments
            OTHER_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate connection file exists
if [[ ! -f "$CONNECTION_FILE" ]]; then
    echo "Error: Connection file not found: $CONNECTION_FILE" >&2
    exit 1
fi

# Check if container is running
if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container $CONTAINER_NAME is not running" >&2
    echo "Start it with: ./scripts/start_nemo.sh" >&2
    exit 1
fi

# Run the kernel inside the container using docker exec
# The connection file should be accessible via the mounted runtime directory
exec docker exec -i $CONTAINER_NAME bash -c "cd /workspace && python -m ipykernel_launcher -f '$CONNECTION_FILE'" 
