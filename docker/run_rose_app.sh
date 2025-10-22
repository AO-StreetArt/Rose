#!/usr/bin/env bash
set -euo pipefail

APP_MODULE="${ROSE_WSGI_APP:-rose.exec.texture_server:app}"
HOST="${IMAGE_TO_3D_SERVER_HOST:-${TEXTURE_SERVER_HOST:-0.0.0.0}}"
if [[ -n "${SAGEMAKER_BIND_TO_HOST:-}" ]]; then
  HOST="${SAGEMAKER_BIND_TO_HOST}"
fi

PORT_DEFAULT="${IMAGE_TO_3D_SERVER_PORT:-${TEXTURE_SERVER_PORT:-8080}}"
if [[ -n "${SAGEMAKER_BIND_TO_PORT:-}" ]]; then
  PORT="${SAGEMAKER_BIND_TO_PORT}"
else
  PORT="${PORT_DEFAULT}"
fi
WORKERS="${ROSE_GUNICORN_WORKERS:-2}"
THREADS="${ROSE_GUNICORN_THREADS:-1}"
CERT_PATH="${TEXTURE_SERVER_CERT_PATH:-}"
KEY_PATH="${TEXTURE_SERVER_KEY_PATH:-}"
EXTRA_OPTS=()

if [[ -n "${ROSE_GUNICORN_TIMEOUT:-}" ]]; then
  EXTRA_OPTS+=("--timeout" "${ROSE_GUNICORN_TIMEOUT}")
fi
if [[ -n "${ROSE_GUNICORN_LOG_LEVEL:-}" ]]; then
  EXTRA_OPTS+=("--log-level" "${ROSE_GUNICORN_LOG_LEVEL}")
fi

CMD=("gunicorn" "${APP_MODULE}" "--bind" "${HOST}:${PORT}" "--workers" "${WORKERS}" "--threads" "${THREADS}")
CMD+=("${EXTRA_OPTS[@]}")

if [[ -n "$CERT_PATH" || -n "$KEY_PATH" ]]; then
  if [[ -z "$CERT_PATH" || -z "$KEY_PATH" ]]; then
    echo "Both TEXTURE_SERVER_CERT_PATH and TEXTURE_SERVER_KEY_PATH must be set to enable TLS" >&2
    exit 1
  fi
  CMD+=("--certfile" "$CERT_PATH" "--keyfile" "$KEY_PATH")
else
  echo "WARNING: TLS cert/key not provided. Gunicorn will listen over HTTP" >&2
fi

exec "${CMD[@]}"
