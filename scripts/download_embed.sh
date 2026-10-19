#!/usr/bin/bash
source ../docker/.env
echo LOCAL_NIM_CACHE=$LOCAL_NIM_CACHE
docker run --rm --runtime=nvidia --gpus=all \
  -e NGC_API_KEY=${NGC_CLI_API_KEY} \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:latest \
  download-to-cache
