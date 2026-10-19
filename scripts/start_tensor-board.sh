#!/bin/bash

# If ../nim-workshop-env/bin/activate exists use it, otherwise just use the system python
export SCRIPTS_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
echo "SCRIPTS_DIR: $SCRIPTS_DIR"
source $SCRIPTS_DIR/../docker/.env
echo PROJECT_DIR: $PROJECT_DIR
cd $PROJECT_DIR

# Kill any existing jupyter-lab processes
pkill -f "tensorboard"
if [ -f ./nim-workshop-env/bin/activate ]; then
    # Local environment
    # Make sure  ~/.local/share/jupyter/kernels/docker-nemo-exec/kernel.json exists 
    echo "Starting Tensorboard in Local environment"
    if [ ! -f ~/.local/share/jupyter/kernels/docker-nemo-exec/kernel.json ]; then
        echo "WARNING ~/.local/share/jupyter/kernels/docker-nemo-exec/kernel.json does not exist"
        echo "You need to manually create it"
    fi 
    source ./nim-workshop-env/bin/activate && tensorboard --logdir=./output --port=6006 --host=0.0.0.0 --path_prefix=/tensorboard &
else
    # Launchpad environment and redirect output to tensorboard.log
    echo "Starting Tensorboard in Launchpad environment"
    tensorboard --logdir=./output --port=6006 --host=0.0.0.0 --path_prefix=/tensorboard  > tensorboard.log 2>&1 &
fi




