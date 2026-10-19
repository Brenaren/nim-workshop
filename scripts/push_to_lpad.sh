#!/bin/bash
rsync -avz \
  --exclude='nim-workshop-env' \
  --exclude='temp' \
  --exclude='nemo_experiments' \
  --exclude='results' \
  --exclude='pubmedqa' \
  --exclude='models' \
  --exclude='lib' \
  --exclude='.git' \
  --exclude='utils' \
  --exclude='data/curated-data/*jsonl' \
  --exclude='output' \
  --exclude='tmp' \
  ~/data/projects/2025-07-nim-workshop/ nvidia@launchpad:/home/nvidia/nim-workshop/

rsync -avz \
  --exclude='.git' \
  --exclude='files/*tgz' \
  --exclude='files/nim-workshop' \
  ~/data/projects/2025-07-nim-workshop-lp/ nvidia@launchpad:/home/nvidia/lp/
