#!/usr/bin/env bash
set -euo pipefail

APP_MODULE="${TRIPOSR_SERVER_APP:-${ROSE_WSGI_APP:-triposr_server:app}}"
HOST="${TRIPOSR_SERVER_HOST:-0.0.0.0}"
if [[ -n "${SAGEMAKER_BIND_TO_HOST:-}" ]]; then
  HOST="${SAGEMAKER_BIND_TO_HOST}"
fi

PORT="${TRIPOSR_SERVER_PORT:-8080}"
if [[ -n "${SAGEMAKER_BIND_TO_PORT:-}" ]]; then
  PORT="${SAGEMAKER_BIND_TO_PORT}"
fi

WORKERS="${TRIPOSR_SERVER_WORKERS:-${ROSE_GUNICORN_WORKERS:-1}}"
THREADS="${TRIPOSR_SERVER_THREADS:-${ROSE_GUNICORN_THREADS:-1}}"
TIMEOUT="${TRIPOSR_SERVER_TIMEOUT:-${ROSE_GUNICORN_TIMEOUT:-}}"
LOG_LEVEL="${TRIPOSR_SERVER_LOG_LEVEL:-${ROSE_GUNICORN_LOG_LEVEL:-}}"
CERT_PATH="${TRIPOSR_SERVER_CERT_PATH:-}"
KEY_PATH="${TRIPOSR_SERVER_KEY_PATH:-}"

CMD=("gunicorn" "${APP_MODULE}" "--bind" "${HOST}:${PORT}" "--workers" "${WORKERS}" "--threads" "${THREADS}")

if [[ -n "$TIMEOUT" ]]; then
  CMD+=("--timeout" "$TIMEOUT")
fi
if [[ -n "$LOG_LEVEL" ]]; then
  CMD+=("--log-level" "$LOG_LEVEL")
fi

if [[ -n "$CERT_PATH" || -n "$KEY_PATH" ]]; then
  if [[ -z "$CERT_PATH" || -z "$KEY_PATH" ]]; then
    echo "Both TRIPOSR_SERVER_CERT_PATH and TRIPOSR_SERVER_KEY_PATH must be set to enable TLS." >&2
    exit 1
  fi
  CMD+=("--certfile" "$CERT_PATH" "--keyfile" "$KEY_PATH")
fi

exec "${CMD[@]}"
