#!/bin/bash

source /home/nvidia/.bashrc
# start jupyter
cd $HOME/nim-workshop

# kill existing jupyter notebook and ipykernels
pkill -f "jupyter notebook"
pkill -f "ipykernel"

# Stop nim containers and start up Nemo Container
scripts/stop_containers.sh
scripts/start_nemo.sh
scripts/start_tensor-board.sh
scripts/start_jupyter.sh

# scripts/copy_data.sh




