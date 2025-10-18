#!/usr/bin/env bash
set -euo pipefail

APP_MODULE="${DEPTH_SERVER_APP:-${ROSE_WSGI_APP:-depth_estimator_server:app}}"
HOST="${DEPTH_SERVER_HOST:-0.0.0.0}"
if [[ -n "${SAGEMAKER_BIND_TO_HOST:-}" ]]; then
  HOST="${SAGEMAKER_BIND_TO_HOST}"
fi

PORT="${DEPTH_SERVER_PORT:-8080}"
if [[ -n "${SAGEMAKER_BIND_TO_PORT:-}" ]]; then
  PORT="${SAGEMAKER_BIND_TO_PORT}"
fi

WORKERS="${DEPTH_SERVER_WORKERS:-${ROSE_GUNICORN_WORKERS:-2}}"
THREADS="${DEPTH_SERVER_THREADS:-${ROSE_GUNICORN_THREADS:-1}}"
TIMEOUT="${DEPTH_SERVER_TIMEOUT:-${ROSE_GUNICORN_TIMEOUT:-}}"
LOG_LEVEL="${DEPTH_SERVER_LOG_LEVEL:-${ROSE_GUNICORN_LOG_LEVEL:-}}"
CERT_PATH="${DEPTH_SERVER_CERT_PATH:-}"
KEY_PATH="${DEPTH_SERVER_KEY_PATH:-}"

CMD=("gunicorn" "${APP_MODULE}" "--bind" "${HOST}:${PORT}" "--workers" "${WORKERS}" "--threads" "${THREADS}")

if [[ -n "$TIMEOUT" ]]; then
  CMD+=("--timeout" "$TIMEOUT")
fi
if [[ -n "$LOG_LEVEL" ]]; then
  CMD+=("--log-level" "$LOG_LEVEL")
fi

if [[ -n "$CERT_PATH" || -n "$KEY_PATH" ]]; then
  if [[ -z "$CERT_PATH" || -z "$KEY_PATH" ]]; then
    echo "Both DEPTH_SERVER_CERT_PATH and DEPTH_SERVER_KEY_PATH must be set to enable TLS." >&2
    exit 1
  fi
  CMD+=("--certfile" "$CERT_PATH" "--keyfile" "$KEY_PATH")
fi

exec "${CMD[@]}"
