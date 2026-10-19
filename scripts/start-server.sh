#!/bin/bash
set -u
export VLLM_CONFIGURE_LOGGING=1

NIM_LOG_LEVEL="${NIM_LOG_LEVEL:-DEFAULT}"

case "${NIM_LOG_LEVEL}" in
  "TRACE")
    export TLLM_LOG_LEVEL="TRACE"
    export VLLM_NVEXT_LOG_LEVEL="DEBUG"
    export UVICORN_LOG_LEVEL="trace"
    ;;
  "DEBUG")
    export TLLM_LOG_LEVEL="DEBUG"
    export VLLM_NVEXT_LOG_LEVEL="DEBUG"
    export UVICORN_LOG_LEVEL="debug"
    ;;
  "INFO")
    export TLLM_LOG_LEVEL="INFO"
    export VLLM_NVEXT_LOG_LEVEL="INFO"
    export UVICORN_LOG_LEVEL="info"
    ;;
  "DEFAULT")
    export TLLM_LOG_LEVEL="ERROR"
    export VLLM_NVEXT_LOG_LEVEL="INFO"
    export UVICORN_LOG_LEVEL="info"
    ;;
  "WARNING")
    export TLLM_LOG_LEVEL="WARNING"
    export VLLM_NVEXT_LOG_LEVEL="WARNING"
    export UVICORN_LOG_LEVEL="warning"
    ;;
  "ERROR")
    export TLLM_LOG_LEVEL="ERROR"
    export VLLM_NVEXT_LOG_LEVEL="ERROR"
    export UVICORN_LOG_LEVEL="error"
    ;;
  "CRITICAL")
    export TLLM_LOG_LEVEL="ERROR"
    export VLLM_NVEXT_LOG_LEVEL="CRITICAL"
    export UVICORN_LOG_LEVEL="critical"
    ;;
  *)
    echo "Unsupported value ('${NIM_LOG_LEVEL}') of NIM_LOG_LEVEL. Supported values are: TRACE, DEBUG, INFO, DEFAULT, WARNING, ERROR, CRITICAL." >&2
    exit 1
    ;;
esac

NIM_JSONL_LOGGING="${NIM_JSONL_LOGGING:-0}"
if [ "${NIM_JSONL_LOGGING}" = 1 ]; then
  export VLLM_LOGGING_CONFIG_PATH="/etc/nim/config/python_jsonl_logging_config.json"
  export VLLM_NVEXT_LOGGING_CONFIG_PATH="/etc/nim/config/python_jsonl_logging_config.json"
elif [ "${NIM_JSONL_LOGGING}" = 0 ]; then
  export VLLM_LOGGING_CONFIG_PATH="/etc/nim/config/python_readable_logging_config.json"
  export VLLM_NVEXT_LOGGING_CONFIG_PATH="/etc/nim/config/python_readable_logging_config.json"
else
  echo "ERROR: Unsupported value ('${NIM_JSONL_LOGGING}') of NIM_JSONL_LOGGING env variable. Supported values are 0 and 1" >&2
  exit 1
fi
# Required for launchpad A30!
python3 -m vllm_nvext.entrypoints.openai.api_server --gpu-memory-utilization 0.9 --max-num-seqs 1 --max-model-len 10000
