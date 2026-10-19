#!/bin/bash

###################################################################################
# This script is just a helper script to run the models as shown in build.nvidia.com
# We add some boilerplate checking code to make sure the local nim cache is setup properly
# Note that we run the docker container in daemon mode so that we can run the server in the background
######################################


export SCRIPTS_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
echo "SCRIPTS_DIR: $SCRIPTS_DIR"
source $SCRIPTS_DIR/../docker/.env
docker login -u \$oauthtoken -p $NGC_API_KEY nvcr.io

echo Pulling $NIM_IMG
docker pull $NIM_IMG
# make sure $Local_NIM_CACHE is set to the path of the local cache
if [ -z "$LOCAL_NIM_CACHE" ]; then
  echo "error LOCAL_NIM_CACHE is not set.  Make sure environment .env file is sourced"
  exit 1
fi

echo "saving model data to $LOCAL_NIM_CACHE"
MYUID=$(id -u)
mkdir -p $LOCAL_NIM_CACHE
sudo chown -R $MYUID:$MYUID $LOCAL_NIM_CACHE

docker run -d --rm --shm-size=24g --gpus '"device=0"' \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u $(id -u) \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -e NIM_SERVED_MODEL_NAME \
  -e NIM_MODEL_PROFILE=vllm-bf16-tp1 \
  -e NIM_MAX_MODEL_LEN=16384 \ 
  --name nim_llama_3.1 \
   $NIM_IMG

# Nim Override
#  -v "$PROJECT_DIR/scripts/start-server.sh:/opt/nim/start-server.sh" \

echo "Nim Llama  is running in daemon mode.  Use "
echo "  docker logs nim_llama_3.1"
echo "to check status"
