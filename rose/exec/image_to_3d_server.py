"""HTTPS Flask server exposing image-to-3D conversion backends.

The server mirrors the texture server's structure: it enforces optional
basic-auth, exposes a health endpoint, and lazily constructs converter
instances for each supported backend. Requests are JSON payloads containing a
base64 encoded source image and optional parameters forwarded to the
underlying `ImageTo3DConverter` methods whose names start with `image_to_3d`.

Example request body for the TripoSR backend::

    {
      "image_base64": "...",
      "params": {"seed": 7},
      "model_name": "stabilityai/TripoSR",
      "device": "cuda"
    }

Responses contain a JSON-serialisable representation of the model output as
well as the backend metadata. Non-serialisable objects are surfaced via their
string representation so callers can handle them downstream (for example,
persisting meshes to disk).
"""

from __future__ import annotations

import base64
import hmac
import io
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import Flask, abort, jsonify, make_response, request

try:
    from PIL import Image
except ImportError:  # pragma: no cover - handled during server start
    Image = None  # type: ignore

try:
    from rose.processing.image_to_3d import ImageTo3DConverter
except ImportError:  # pragma: no cover - script mode without package path
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from rose.processing.image_to_3d import ImageTo3DConverter


app = Flask(__name__)

_logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    level_name = os.getenv(
        "IMAGE_TO_3D_SERVER_LOG_LEVEL",
        os.getenv("TEXTURE_SERVER_LOG_LEVEL", "INFO"),
    )
    try:
        level_value = getattr(logging, str(level_name).upper())
    except AttributeError:
        level_value = logging.INFO

    logging.basicConfig(level=level_value)
    logging.getLogger().setLevel(level_value)
    _logger.setLevel(level_value)


_configure_logging()


def _parse_bool_env(env_value: Optional[str]) -> Optional[bool]:
    if env_value is None:
        return None
    value = env_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


_AUTH_USERNAME = os.getenv("IMAGE_TO_3D_SERVER_USERNAME", os.getenv("TEXTURE_SERVER_USERNAME", "admin"))
_AUTH_PASSWORD = os.getenv("IMAGE_TO_3D_SERVER_PASSWORD", os.getenv("TEXTURE_SERVER_PASSWORD", "change-me"))

_AUTH_FORCE_ENABLE = _parse_bool_env(os.getenv("IMAGE_TO_3D_ENABLE_AUTH"))
_AUTH_FORCE_DISABLE = _parse_bool_env(os.getenv("IMAGE_TO_3D_DISABLE_AUTH"))
_RUNNING_IN_SAGEMAKER = bool(os.getenv("SAGEMAKER_ENDPOINT_NAME"))

if _AUTH_FORCE_ENABLE is not None:
    _AUTH_ENABLED = _AUTH_FORCE_ENABLE
elif _AUTH_FORCE_DISABLE is not None:
    _AUTH_ENABLED = not _AUTH_FORCE_DISABLE
elif _RUNNING_IN_SAGEMAKER:
    _AUTH_ENABLED = False
else:
    _AUTH_ENABLED = True

_DEFAULT_BACKEND = os.getenv("IMAGE_TO_3D_DEFAULT_BACKEND", "3dtopia")


_DEFAULT_DEVICE = os.getenv("IMAGE_TO_3D_DEVICE", "cuda")


_BACKENDS: Dict[str, Dict[str, Any]] = {
    "trellis": {
        "method": "image_to_3d",
        "default_model": "microsoft/TRELLIS-image-large",
        "input_field": "image",
        "description": "Microsoft TRELLIS image-to-3D pipeline",
    },
    "vggt": {
        "method": "image_to_3d_vggt",
        "default_model": "facebook/VGGT-1B",
        "input_field": "image_path",
        "description": "Meta VGGT monocular geometry estimator",
    },
    "hunyuan": {
        "method": "image_to_3d_hunyuan",
        "default_model": "tencent/Hunyuan3D",
        "input_field": "image_path",
        "description": "Tencent Hunyuan3D Gaussian reconstruction",
    },
    "triposr": {
        "method": "image_to_3d_triposr",
        "default_model": "stabilityai/TripoSR",
        "input_field": "image_path",
        "description": "Stability AI TripoSR mesh extractor",
    },
    "3dtopia": {
        "method": "image_to_3d_3dtopia",
        "default_model": "3DTopia/3DTopia-XL",
        "input_field": "prompt",
        "description": "3DTopia-XL text-to-3D generator",
    },
}


_converter_lock = threading.Lock()
_converter_cache: Dict[Tuple[str, str, str], ImageTo3DConverter] = {}


def _get_converter(backend: str, model_name: Optional[str], device: Optional[str]) -> ImageTo3DConverter:
    resolved_model = model_name or _BACKENDS[backend]["default_model"]
    resolved_device = device or _DEFAULT_DEVICE
    cache_key = (backend, resolved_model, resolved_device)
    with _converter_lock:
        converter = _converter_cache.get(cache_key)
        if converter is None:
            _logger.info(
                "Initialising ImageTo3DConverter for backend=%s model=%s device=%s",
                backend,
                resolved_model,
                resolved_device,
            )
            converter = ImageTo3DConverter(model_name=resolved_model, device=resolved_device)
            _converter_cache[cache_key] = converter
    return converter


def _unauthorized_response():
    response = make_response("", 401)
    response.headers["WWW-Authenticate"] = 'Basic realm="ImageTo3D Server", charset="UTF-8"'
    return response


def _verify_basic_auth() -> bool:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Basic "):
        return False

    token = auth_header.split(" ", 1)[1].strip()
    try:
        decoded = base64.b64decode(token).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        _logger.warning("Failed to decode basic auth token: %s", exc)
        return False

    username, _, password = decoded.partition(":")
    return hmac.compare_digest(username, _AUTH_USERNAME) and hmac.compare_digest(password, _AUTH_PASSWORD)


def _requires_auth(view_func):
    from functools import wraps

    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not _AUTH_ENABLED:
            return view_func(*args, **kwargs)
        if not _verify_basic_auth():
            return _unauthorized_response()
        return view_func(*args, **kwargs)

    return wrapper


def _decode_base64_image(encoded: str) -> Tuple[bytes, Image.Image]:
    try:
        payload = base64.b64decode(encoded)
    except (ValueError, TypeError) as exc:
        abort(400, description=f"Invalid base64 image payload: {exc}")

    if Image is None:
        abort(500, description="Pillow is required but not installed")

    try:
        with Image.open(io.BytesIO(payload)) as img:
            pil_image = img.convert("RGB")
    except Exception as exc:
        abort(400, description=f"Failed to decode image: {exc}")

    return payload, pil_image


def _serialize(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _serialize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _parse_request_payload() -> Dict[str, Any]:
    if not request.data:
        abort(400, description="Missing request body")
    try:
        data = json.loads(request.data)
    except json.JSONDecodeError as exc:
        abort(400, description=f"Request body is not valid JSON: {exc}")
    if not isinstance(data, dict):
        abort(400, description="JSON body must be an object")
    return data


def _invoke_backend(backend: str, data: Dict[str, Any]) -> Dict[str, Any]:
    params = data.get("params") or {}
    if not isinstance(params, dict):
        abort(400, description="params must be a JSON object if provided")

    model_name = data.get("model_name")
    device_override = data.get("device")

    converter = _get_converter(backend, model_name, device_override)
    method = getattr(converter, _BACKENDS[backend]["method"])

    input_field = _BACKENDS[backend]["input_field"]

    call_kwargs = dict(params)

    image_bytes: Optional[bytes] = None
    pil_image: Optional[Image.Image] = None
    temp_path: Optional[str] = None

    if input_field in {"image", "image_path"}:
        image_b64 = data.get("image_base64") or call_kwargs.pop("image_base64", None)
        if not isinstance(image_b64, str) or not image_b64.strip():
            abort(400, description="Missing required field: image_base64")
        image_bytes, pil_image = _decode_base64_image(image_b64)
        if input_field == "image_path":
            suffix = call_kwargs.pop("temp_suffix", ".png")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(image_bytes)
                temp_path = tmp.name
            call_kwargs["image_path"] = temp_path
        else:
            call_kwargs[input_field] = pil_image
    elif input_field == "prompt":
        prompt = data.get("prompt")
        if prompt is None:
            prompt = call_kwargs.pop("prompt", None)
        if not isinstance(prompt, str) or not prompt.strip():
            abort(400, description="Missing required field: prompt")
        call_kwargs["prompt"] = prompt
    else:
        abort(500, description=f"Unsupported input field mapping: {input_field}")

    try:
        result = method(**call_kwargs)
    except ImportError as exc:
        _logger.exception("Backend dependency missing for %s", backend)
        abort(503, description=str(exc))
    except Exception as exc:  # pragma: no cover - surface runtime errors
        _logger.exception("Image-to-3D backend failed")
        abort(500, description=str(exc))
    finally:
        if temp_path and Path(temp_path).exists():
            try:
                os.remove(temp_path)
            except OSError:
                _logger.warning("Failed to delete temporary file: %s", temp_path)

    return {
        "backend": backend,
        "model_name": model_name or _BACKENDS[backend]["default_model"],
        "device": device_override or _DEFAULT_DEVICE,
        "output": _serialize(result),
    }


def _healthy_payload() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/health")
def healthcheck():
    return jsonify(_healthy_payload())


@app.get("/ping")
def ping():
    return jsonify(_healthy_payload())


def _create_backend_route(backend: str):
    endpoint_path = f"/{backend}"
    view_name = f"handle_{backend}"

    @_requires_auth
    def handler(backend_name: str = backend):
        data = _parse_request_payload()
        payload = _invoke_backend(backend_name, data)
        return jsonify(payload)

    handler.__name__ = view_name
    handler.__doc__ = f"Invoke the {backend} backend ({_BACKENDS[backend]['description']})."
    app.add_url_rule(endpoint_path, view_name, handler, methods=["POST"])


@app.post("/invocations")
@_requires_auth
def invoke_default():
    if request.content_type and "json" not in request.content_type.lower():
        abort(415, description="Only application/json requests are supported")
    data = _parse_request_payload()
    backend = data.get("backend") or _DEFAULT_BACKEND
    if backend not in _BACKENDS:
        abort(400, description=f"Unsupported backend: {backend}")
    payload = _invoke_backend(backend, data)
    return jsonify(payload)


for _backend_name in _BACKENDS:
    _create_backend_route(_backend_name)


def _resolve_host_port() -> Tuple[str, int]:
    host = os.getenv("IMAGE_TO_3D_SERVER_HOST") or os.getenv("TEXTURE_SERVER_HOST") or os.getenv("SAGEMAKER_BIND_TO_HOST") or "0.0.0.0"
    port_env = (
        os.getenv("SAGEMAKER_BIND_TO_PORT")
        or os.getenv("IMAGE_TO_3D_SERVER_PORT")
        or os.getenv("TEXTURE_SERVER_PORT")
        or "8080"
    )
    try:
        port = int(port_env)
    except ValueError:
        _logger.warning("Invalid port value '%s'; defaulting to 8444", port_env)
        port = 8444
    return host, port


def main() -> None:
    host, port = _resolve_host_port()
    level_name = os.getenv("IMAGE_TO_3D_SERVER_LOG_LEVEL", os.getenv("TEXTURE_SERVER_LOG_LEVEL", "INFO"))
    level_value = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(level=level_value)
    app.run(host=host, port=port, ssl_context="adhoc")


if __name__ == "__main__":
    main()
