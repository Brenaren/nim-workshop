#!/bin/bash
cd $HOME/nim-workshop

# Bootstrap pip into the venv the Jupyter kernel uses
curl -sS https://bootstrap.pypa.io/get-pip.py | /home/ubuntu/.venv/bin/python
/home/ubuntu/.venv/bin/pip install -r /home/ubuntu/nim-workshop/docker/requirements.txt

bash scripts/start_jupyter.sh
