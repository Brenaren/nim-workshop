#!/bin/bash
cd $HOME/nim-workshop

# Bootstrap pip into the venv the Jupyter kernel uses
/home/ubuntu/.venv/bin/python -m ensurepip --upgrade
/home/ubuntu/.venv/bin/python -m pip install -r docker/requirements.txt

bash scripts/start_jupyter.sh
