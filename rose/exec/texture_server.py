"""Simple HTTPS Flask server exposing texture generation and health endpoints.

The server expects basic authentication credentials and POST requests that
include a `control_image` upload alongside generation parameters. It uses the
`TextureGenerator` class to produce texture maps and returns the results as
base64-encoded PNG data in a JSON payload.
"""
import base64
import hmac
import io
import logging
import os
import threading
from typing import Optional, Tuple

from flask import Flask, abort, jsonify, make_response, request
from PIL import Image, UnidentifiedImageError

try:
    from rose.processing.texture_generator import TextureGenerator
except ImportError:  # pragma: no cover - script mode without package path
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from rose.processing.texture_generator import TextureGenerator

app = Flask(__name__)

_logger = logging.getLogger(__name__)

_AUTH_USERNAME = os.getenv("TEXTURE_SERVER_USERNAME", "admin")
_AUTH_PASSWORD = os.getenv("TEXTURE_SERVER_PASSWORD", "change-me")

_generator_lock = threading.Lock()
_generator: Optional[TextureGenerator] = None


def _get_texture_generator() -> TextureGenerator:
    """Lazily instantiate the texture generator with thread-safety."""
    global _generator
    with _generator_lock:
        if _generator is None:
            _logger.info("Initializing TextureGenerator instance")
            _generator = TextureGenerator()
    return _generator


def _unauthorized_response():
    response = make_response("", 401)
    response.headers["WWW-Authenticate"] = 'Basic realm="Texture Server", charset="UTF-8"'
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
    """Decorator enforcing basic authentication."""
    from functools import wraps

    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not _verify_basic_auth():
            return _unauthorized_response()
        return view_func(*args, **kwargs)

    return wrapper


def _parse_opt_int(param: str) -> Optional[int]:
    value = request.values.get(param)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        abort(400, description=f"Invalid integer for '{param}': {value}")


def _parse_opt_float(param: str) -> Optional[float]:
    value = request.values.get(param)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        abort(400, description=f"Invalid float for '{param}': {value}")


def _image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.get("/health")
def healthcheck():
    """Return a lightweight status response for container orchestrators."""
    return jsonify({"status": "ok"})


@app.post("/textures")
@_requires_auth
def generate_textures_endpoint():
    prompt = request.values.get("prompt")
    if not prompt:
        abort(400, description="Missing required parameter: prompt")

    control_file = request.files.get("control_image")
    if control_file is None or control_file.filename == "":
        abort(400, description="Missing required file upload: control_image")

    control_file.stream.seek(0)
    try:
        with Image.open(control_file.stream) as img:
            control_image = img.convert("RGB")
    except UnidentifiedImageError:
        abort(400, description="control_image is not a valid image file")
    except Exception as exc:  # pragma: no cover - unexpected PIL errors
        abort(400, description=f"Failed to load control image: {exc}")

    negative_prompt = request.values.get("negative_prompt")
    seed = _parse_opt_int("seed")
    steps = _parse_opt_int("steps")
    guidance = _parse_opt_float("guidance")
    conditioning_scale = _parse_opt_float("conditioning_scale")

    width = _parse_opt_int("width")
    height = _parse_opt_int("height")
    output_size: Optional[Tuple[int, int]] = None
    if width is not None or height is not None:
        if width is None or height is None:
            abort(400, description="Both width and height must be provided for output sizing")
        output_size = (width, height)

    generator = _get_texture_generator()

    kwargs = {
        "control_image": control_image,
        "prompt": prompt,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if seed is not None:
        kwargs["seed"] = seed
    if steps is not None:
        kwargs["steps"] = steps
    if guidance is not None:
        kwargs["guidance"] = guidance
    if conditioning_scale is not None:
        kwargs["conditioning_scale"] = conditioning_scale
    if output_size is not None:
        kwargs["output_size"] = output_size

    try:
        textures = generator.generate_textures(**kwargs)
    except Exception as exc:  # pragma: no cover - surface runtime issues cleanly
        _logger.exception("Texture generation failed")
        abort(500, description=str(exc))

    payload = {name: _image_to_base64_png(image) for name, image in textures.items()}
    return jsonify({"textures": payload})


def _resolve_ssl_context():
    cert_path = os.getenv("TEXTURE_SERVER_CERT_PATH")
    key_path = os.getenv("TEXTURE_SERVER_KEY_PATH")
    if cert_path and key_path:
        return cert_path, key_path
    _logger.warning("TEXTURE_SERVER_CERT_PATH/TEXTURE_SERVER_KEY_PATH not set; falling back to adhoc certificate")
    return "adhoc"


def main() -> None:
    host = os.getenv("TEXTURE_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("TEXTURE_SERVER_PORT", "8443"))
    logging.basicConfig(level=os.getenv("TEXTURE_SERVER_LOG_LEVEL", "INFO"))
    app.run(host=host, port=port, ssl_context=_resolve_ssl_context())


if __name__ == "__main__":
    main()
