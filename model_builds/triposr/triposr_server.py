"""Flask server exposing the ImageTo3DConverter (TripoSR) for HTTP inference.

The service mirrors the contract used by SageMaker containers:
    * ``GET /ping`` acts as the health check.
    * ``POST /invocations`` accepts inference requests.

Request payload::

    {
        "image_base64": "...",               # Required: base64-encoded RGB image.
        "model_name": "stabilityai/TripoSR", # Optional override for the model id.
        "response_format": "mesh_base64",    # Optional: 'mesh_base64' (default) or 'mesh_repr'.
        "include_raw_result": false          # Optional: include repr(result) for debugging.
    }

The response provides metadata and, when possible, a base64-encoded GLB payload
representing the generated mesh.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple, Union

from flask import Flask, jsonify, request

try:
    from PIL import Image
except ImportError:  # pragma: no cover - handled during container startup
    Image = None  # type: ignore

from image_to_3d import ImageTo3DConverter


app = Flask(__name__)

_logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    level_name = os.getenv("TRIPOSR_SERVER_LOG_LEVEL", "INFO")
    try:
        level_value = getattr(logging, str(level_name).upper())
    except AttributeError:
        level_value = logging.INFO

    logging.basicConfig(level=level_value)
    logging.getLogger().setLevel(level_value)
    _logger.setLevel(level_value)


_configure_logging()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_HF_CACHE_DIR = os.getenv("TRIPOSR_SERVER_HF_CACHE_DIR")
if _HF_CACHE_DIR:
    os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _HF_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", _HF_CACHE_DIR)

_DEFAULT_MODEL_NAME = os.getenv("TRIPOSR_SERVER_MODEL_ID", "stabilityai/TripoSR")
_DEFAULT_DEVICE = os.getenv("TRIPOSR_SERVER_DEVICE", "cuda")
_DEFAULT_RESPONSE_FORMAT = os.getenv("TRIPOSR_SERVER_DEFAULT_RESPONSE", "mesh_base64").strip().lower()
_INCLUDE_RAW_RESULT = _env_bool("TRIPOSR_SERVER_INCLUDE_RAW_RESULT", default=False)

_SUPPORTED_RESPONSE_FORMATS = {"mesh_base64", "mesh_repr"}

_converter_lock = threading.Lock()
_converter_instance: Optional[ImageTo3DConverter] = None


def _get_converter() -> ImageTo3DConverter:
    global _converter_instance
    with _converter_lock:
        if _converter_instance is None:
            _logger.info(
                "Initialising ImageTo3DConverter (model=%s, device=%s)",
                _DEFAULT_MODEL_NAME,
                _DEFAULT_DEVICE,
            )
            _converter_instance = ImageTo3DConverter(
                model_name=_DEFAULT_MODEL_NAME,
                device=_DEFAULT_DEVICE,
            )
    return _converter_instance


try:
    # Warm up the converter during startup so health checks reflect readiness.
    _get_converter()
except Exception as exc:  # pragma: no cover - fail fast if initialisation is broken
    _logger.exception("Failed to initialise TripoSR converter during startup: %s", exc)
    raise


def _decode_base64_image(image_b64: str) -> Image.Image:
    if Image is None:
        raise RuntimeError("Pillow is not available; cannot decode images")

    if "," in image_b64:
        # Support data URLs: "data:image/png;base64,<payload>"
        image_b64 = image_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except (ValueError, binascii.Error) as exc:  # type: ignore[name-defined]
        _logger.warning("Failed to decode base64 payload: %s", exc)
        raise ValueError("Invalid base64-encoded image payload") from exc

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except (OSError, ValueError) as exc:
        _logger.warning("Failed to open image: %s", exc)
        raise ValueError("Unable to decode image; unsupported or corrupt file") from exc

    return image.convert("RGB")


def _summarise_value(value: Any) -> str:
    try:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return repr(value)
        if isinstance(value, (list, tuple)):
            sample = value[0] if value else None
            return f"<{type(value).__name__} len={len(value)} sample_type={type(sample).__name__ if sample is not None else 'None'}>"
        if isinstance(value, dict):
            keys = list(value.keys())
            return f"<dict keys={keys[:3]}{'...' if len(keys) > 3 else ''}>"
        return f"<{type(value).__name__}>"
    except Exception:
        return "<unserialisable>"


def _export_mesh_to_bytes(mesh: Any) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """Attempt to export mesh-like objects into GLB/GLTF/OBJ bytes."""
    if mesh is None:
        return None, None, None

    if isinstance(mesh, (list, tuple)) and mesh:
        mesh = mesh[0]

    # Direct bytes payloads can be returned as-is.
    if isinstance(mesh, (bytes, bytearray)):
        return bytes(mesh), "application/octet-stream", "bin"

    # Strings may already represent serialized geometry.
    if isinstance(mesh, str):
        return mesh.encode("utf-8"), "text/plain", "txt"

    candidates = (
        ("glb", "model/gltf-binary"),
        ("gltf", "model/gltf+json"),
        ("obj", "text/plain"),
    )

    for ext, mime in candidates:
        if hasattr(mesh, "export"):
            try:
                exported = mesh.export(file_type=ext)
                if isinstance(exported, (bytes, bytearray)):
                    return bytes(exported), mime, ext
                if isinstance(exported, str):
                    return exported.encode("utf-8"), "text/plain", ext
                if exported is None:
                    buffer = io.BytesIO()
                    mesh.export(buffer, file_type=ext)  # type: ignore[call-arg]
                    buffer.seek(0)
                    data = buffer.getvalue()
                    if data:
                        return data, mime, ext
            except TypeError:
                buffer = io.BytesIO()
                try:
                    mesh.export(buffer, file_type=ext)  # type: ignore[call-arg]
                    buffer.seek(0)
                    data = buffer.getvalue()
                    if data:
                        return data, mime, ext
                except Exception:
                    continue
            except Exception:
                continue

    if hasattr(mesh, "to_dict"):
        try:
            data = json.dumps(mesh.to_dict()).encode("utf-8")
            return data, "application/json", "json"
        except Exception:
            pass

    try:
        return repr(mesh).encode("utf-8"), "text/plain", "txt"
    except Exception:
        return None, None, None


def _serialise_mesh(mesh: Any, response_format: str) -> Dict[str, Any]:
    if mesh is None:
        return {"mesh_format": "none", "mesh_base64": None}

    if isinstance(mesh, dict):
        candidate = mesh.get("mesh") or mesh.get("meshes")
        mesh_payload = candidate if candidate is not None else mesh
    else:
        mesh_payload = mesh

    if response_format == "mesh_repr":
        return {
            "mesh_format": "repr",
            "mesh_repr": repr(mesh_payload),
        }

    data, mime, ext = _export_mesh_to_bytes(mesh_payload)
    if data is None:
        return {
            "mesh_format": "repr",
            "mesh_repr": repr(mesh_payload),
        }

    return {
        "mesh_format": ext,
        "content_type": mime,
        "mesh_base64": base64.b64encode(data).decode("utf-8"),
    }


def _serialise_response(result: Any, response_format: str, include_raw: bool) -> Dict[str, Any]:
    mesh_info = _serialise_mesh(result, response_format)
    response: Dict[str, Any] = {
        "mesh": mesh_info,
        "response_format": response_format,
    }

    if isinstance(result, dict):
        filtered = {k: _summarise_value(v) for k, v in result.items() if k not in {"mesh", "meshes"}}
        if filtered:
            response["additional_outputs"] = filtered

    if include_raw:
        response["raw_result_repr"] = _summarise_value(result)

    return response


def _parse_request_payload() -> Tuple[Image.Image, Optional[str], str, bool]:
    if request.content_type and request.content_type.startswith("application/json"):
        payload = request.get_json(silent=True)
    else:
        payload = None

    if payload is None:
        try:
            payload = json.loads(request.data.decode("utf-8"))
        except Exception as exc:  # pragma: no cover - defensive path
            _logger.warning("Invalid JSON payload: %s", exc)
            raise ValueError("Request body must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Request JSON must be an object")

    image_b64 = payload.get("image_base64")
    if not isinstance(image_b64, str) or not image_b64.strip():
        raise ValueError("'image_base64' must be a non-empty string")

    model_name = payload.get("model_name")
    if model_name is not None and not isinstance(model_name, str):
        raise ValueError("'model_name' must be a string when provided")

    response_format = payload.get("response_format", _DEFAULT_RESPONSE_FORMAT)
    if not isinstance(response_format, str):
        raise ValueError("'response_format' must be a string")
    response_format = response_format.strip().lower() or _DEFAULT_RESPONSE_FORMAT
    if response_format not in _SUPPORTED_RESPONSE_FORMATS:
        raise ValueError(f"Unsupported response_format '{response_format}'. Expected one of {sorted(_SUPPORTED_RESPONSE_FORMATS)}")

    include_raw = payload.get("include_raw_result", _INCLUDE_RAW_RESULT)
    if not isinstance(include_raw, bool):
        raise ValueError("'include_raw_result' must be a boolean when provided")

    image = _decode_base64_image(image_b64)
    return image, model_name, response_format, include_raw


@app.route("/ping", methods=["GET"])
def ping():
    try:
        _get_converter()
    except Exception as exc:
        _logger.exception("Health check failed: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500
    return jsonify({"status": "ok"}), 200


@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        image, model_name, response_format, include_raw = _parse_request_payload()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive logging path
        _logger.exception("Failed to parse request: %s", exc)
        return jsonify({"error": "Invalid request payload"}), 400

    converter = _get_converter()

    try:
        result = converter.image_to_3d(image, model_name=model_name)
    except Exception as exc:
        _logger.exception("Inference failed: %s", exc)
        return jsonify({"error": "Image-to-3D conversion failed", "details": str(exc)}), 500

    response = _serialise_response(result, response_format, include_raw)
    return jsonify(response), 200


__all__ = ["app"]
