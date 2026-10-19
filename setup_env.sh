#!/bin/bash

ENV_FILE="/home/ubuntu/nim-workshop/docker/.env"

echo "============================================"
echo "  NIM Workshop - Setup"
echo "============================================"
echo ""

# --- Step 1: Install dependencies ---
echo "[1/3] Installing pip..."
curl -sS https://bootstrap.pypa.io/get-pip.py | /home/ubuntu/.venv/bin/python

echo "[2/3] Installing dependencies..."
/home/ubuntu/.venv/bin/pip install -r /home/ubuntu/nim-workshop/docker/requirements.txt

# --- Step 2: API Keys ---
echo ""
echo "[3/3] API Key Setup"
echo "Paste each key when prompted (input is hidden)."
echo "Press Enter to skip a key you don't have yet."
echo ""

read -rsp "NGC API Key: " NGC_API_KEY
echo ""
read -rsp "NVIDIA API Key: " NVIDIA_API_KEY
echo ""
read -rsp "NGC CLI API Key (leave blank to use NGC API Key): " NGC_CLI_API_KEY
echo ""
read -rsp "Hugging Face Token (only needed for notebooks 03a/03b): " HF_TOKEN
echo ""

# If NGC CLI key left blank, default to NGC key
if [ -z "$NGC_CLI_API_KEY" ]; then
    NGC_CLI_API_KEY="$NGC_API_KEY"
fi

# Write keys into the .env file
cat > "$ENV_FILE" << EOF
PROJECT_DIR=/home/ubuntu/nim-workshop

NGC_API_KEY=${NGC_API_KEY}
NVIDIA_API_KEY=${NVIDIA_API_KEY}
NGC_CLI_API_KEY=${NGC_CLI_API_KEY}
HF_TOKEN=${HF_TOKEN}

NIM_IMG="nvcr.io/nim/meta/llama-3.1-8b-instruct:1.3.3"
EMBED_IMG="nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:latest"
RERANKER_IMG="nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2:latest"

LOCAL_NIM_CACHE="\${PROJECT_DIR}/.cache"
NIM_SERVED_MODEL_NAME="meta/llama-3.1-8b-instruct"
EOF

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Click Kernel → Restart Kernel in your"
echo "  notebook before running any cells."
echo "============================================"
