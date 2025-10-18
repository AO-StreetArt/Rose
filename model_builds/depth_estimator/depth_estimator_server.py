"""Flask server exposing the DepthEstimator for AWS SageMaker endpoints.

The server follows the SageMaker container contract:
    * ``GET /ping`` acts as the health check.
    * ``POST /invocations`` performs inference.

Requests must provide a JSON body containing a base64 encoded image. Optional
fields control which depth backend to use and how to serialise the response.

Example request payload::

    {
        "image_base64": "...",
        "estimator": "dpt",
        "output_format": "png"
    }

The response includes metadata about the inference and either the raw depth map
as a nested list (``output_format=array``) or a base64-encoded 16-bit PNG
(``output_format=png``).
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, request

try:
    from PIL import Image
except ImportError:  # pragma: no cover - handled during container startup
    Image = None  # type: ignore

from depth_estimator import DepthEstimator


app = Flask(__name__)

_logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    level_name = os.getenv("DEPTH_SERVER_LOG_LEVEL", "INFO")
    try:
        level_value = getattr(logging, str(level_name).upper())
    except AttributeError:
        level_value = logging.INFO

    logging.basicConfig(level=level_value)
    logging.getLogger().setLevel(level_value)
    _logger.setLevel(level_value)


_configure_logging()


_DEFAULT_OUTPUT_FORMAT = os.getenv("DEPTH_SERVER_DEFAULT_OUTPUT", "array").lower()

_SUPPORTED_ESTIMATORS = {"dpt", "zoedepth"}
_DEFAULT_ESTIMATOR = os.getenv("DEPTH_SERVER_DEFAULT_ESTIMATOR", "dpt").strip().lower()
if _DEFAULT_ESTIMATOR not in _SUPPORTED_ESTIMATORS:
    _logger.warning(
        "Unsupported default estimator '%s'; falling back to 'dpt'. Supported options: %s",
        _DEFAULT_ESTIMATOR,
        ", ".join(sorted(_SUPPORTED_ESTIMATORS)),
    )
    _DEFAULT_ESTIMATOR = "dpt"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_HF_CACHE_DIR = os.getenv("DEPTH_SERVER_HF_CACHE_DIR")
_LOCAL_FILES_ONLY = _env_bool("DEPTH_SERVER_LOCAL_FILES_ONLY", default=False)
_DPT_MODEL_ID = os.getenv("DEPTH_SERVER_DPT_MODEL_ID", "Intel/dpt-large")
_ZOE_MODEL_ID = os.getenv("DEPTH_SERVER_ZOE_MODEL_ID", "Intel/zoedepth-nyu-kitti")
_DPT_REVISION = os.getenv("DEPTH_SERVER_DPT_REVISION")
_ZOE_REVISION = os.getenv("DEPTH_SERVER_ZOE_REVISION")

_estimator_lock = threading.Lock()
_estimator_instance: Optional[DepthEstimator] = None


def _get_estimator() -> DepthEstimator:
    global _estimator_instance
    with _estimator_lock:
        if _estimator_instance is None:
            _logger.info("Initialising DepthEstimator with backend '%s'", _DEFAULT_ESTIMATOR)
            _estimator_instance = DepthEstimator(
                default_backend=_DEFAULT_ESTIMATOR,
                dpt_model_id=_DPT_MODEL_ID,
                zoedepth_model_id=_ZOE_MODEL_ID,
                local_files_only=_LOCAL_FILES_ONLY,
                cache_dir=_HF_CACHE_DIR,
                dpt_revision=_DPT_REVISION,
                zoedepth_revision=_ZOE_REVISION,
            )
    return _estimator_instance


try:
    # Eagerly load the model backend during application startup.
    _get_estimator()
except Exception as exc:  # pragma: no cover - startup failure should halt container build
    _logger.exception("Failed to initialise DepthEstimator during startup: %s", exc)
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


def _serialise_depth_map(depth_map, output_format: str) -> Tuple[Dict[str, Any], str]:
    import numpy as np

    response: Dict[str, Any] = {
        "shape": list(depth_map.shape),
        "dtype": str(depth_map.dtype),
    }

    if output_format == "png":
        if Image is None:
            raise RuntimeError("Pillow is required to serialise depth maps as PNG")
        normalised = depth_map - depth_map.min()
        max_val = float(normalised.max())
        if max_val > 0:
            normalised /= max_val
        as_uint16 = (normalised * 65535.0).astype("uint16")
        buffer = io.BytesIO()
        Image.fromarray(as_uint16).save(buffer, format="PNG")
        buffer.seek(0)
        response["format"] = "png"
        response["depth_map_base64"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
        response["normalisation"] = {"min": float(depth_map.min()), "max": float(depth_map.max())}
    else:
        response["format"] = "array"
        response["depth_map"] = depth_map.astype("float32").tolist()

    return response, response["format"]


def _parse_request_payload() -> Tuple[str, str, Dict[str, Any]]:
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

    estimator_value = payload.get("estimator", _DEFAULT_ESTIMATOR)
    if not isinstance(estimator_value, str):
        raise ValueError("'estimator' must be a string if provided")
    estimator = estimator_value.strip().lower() or _DEFAULT_ESTIMATOR
    if estimator not in _SUPPORTED_ESTIMATORS:
        raise ValueError("Unsupported estimator; choose 'dpt' or 'zoedepth'")

    output_format_value = payload.get("output_format", _DEFAULT_OUTPUT_FORMAT)
    if not isinstance(output_format_value, str):
        raise ValueError("'output_format' must be a string if provided")
    output_format = output_format_value.strip().lower()
    if output_format not in {"array", "png"}:
        raise ValueError("Unsupported output_format; choose 'array' or 'png'")

    return image_b64, estimator, {"output_format": output_format, "raw_payload": payload}


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"}), 200


@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        image_b64, estimator_name, context = _parse_request_payload()
        image = _decode_base64_image(image_b64)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    depth_estimator = _get_estimator()

    try:
        if estimator_name == "zoedepth":
            depth_map = depth_estimator.estimate_depth_zoedepth(image)
            if depth_map is None:
                raise RuntimeError("ZoeDepth inference returned no result")
        else:
            depth_map = depth_estimator.estimate_depth(image)
    except Exception as exc:
        _logger.exception("Depth estimation failed")
        return jsonify({"error": f"Depth estimation failed: {exc}"}), 500

    import numpy as np

    depth_array = np.asarray(depth_map)
    response, response_format = _serialise_depth_map(depth_array, context["output_format"])

    response.update(
        {
            "estimator": estimator_name,
            "format": response_format,
            "min_depth": float(depth_array.min()),
            "max_depth": float(depth_array.max()),
        }
    )

    return jsonify(response), 200


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "depth-estimator",
            "health": "/ping",
            "invocations": "/invocations",
        }
    ), 200


__all__ = ["app"]
