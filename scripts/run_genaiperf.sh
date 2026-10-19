#!/bin/bash
# Get environment variables
# Make this run for less time .. took way too long in notebook
export SCRIPTS_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
echo "SCRIPTS_DIR: $SCRIPTS_DIR"
source $SCRIPTS_DIR/../docker/.env

echo "Running with model ${NIM_SERVED_MODEL_NAME}"

# Define synthetic data parameters - 
SYN_DATA="--num-prompts 100 \
          --synthetic-input-tokens-mean 200 \
          --synthetic-input-tokens-stddev 5 \
          --output-tokens-mean 20 \
          --random-seed 1"


# Create genai-perf command as a string
GENAI_PERF_CMD="genai-perf profile -m ${NIM_SERVED_MODEL_NAME} ${SYN_DATA} \
                --artifact-dir /nim-workshop/logs/ \
                --concurrency 20 \
                --service-kind openai \
                --measurement-interval 4000 \
                --endpoint-type chat \
                --endpoint v1/chat/completions \
                --streaming \
                --u 0.0.0.0:8000 \
                -- --max-threads 256"

# Print the command
echo "${GENAI_PERF_CMD}"
# Run the command
${GENAI_PERF_CMD}
