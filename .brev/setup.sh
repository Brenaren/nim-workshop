#!/bin/bash
cd $HOME/nim-workshop
python -m pip install -r docker/requirements.txt
bash scripts/start_jupyter.sh
