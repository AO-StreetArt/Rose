import base64
import json
from types import SimpleNamespace

import importlib
import sys

import pytest


@pytest.fixture
def server_module(monkeypatch):
    class StubConverter:
        def __init__(self, model_name, device):
            self.model_name = model_name
            self.device = device

    stub_processing = SimpleNamespace(ImageTo3DConverter=StubConverter)

    monkeypatch.setitem(sys.modules, "rose.processing.image_to_3d", stub_processing)
    sys.modules.pop("rose.exec.image_to_3d_server", None)

    server = importlib.import_module("rose.exec.image_to_3d_server")
    importlib.reload(server)

    server._converter_cache.clear()
    yield server
    server._converter_cache.clear()


@pytest.fixture
def client(server_module, monkeypatch):
    # Disable auth by default so tests can focus on functional behaviour.
    monkeypatch.setattr(server_module, "_AUTH_ENABLED", False)
    with server_module.app.test_client() as client:
        yield client


def _sample_prompt() -> str:
    return "a marble bust of a griffin"


def test_health_endpoint_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_ping_endpoint_returns_ok(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_invocations_uses_default_backend(server_module, client, monkeypatch):
    captured = {}

    class DummyConverter:
        def image_to_3d_3dtopia(self, prompt, **kwargs):  # noqa: D401 - test stub
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return {"result": "mesh"}

    dummy = DummyConverter()

    def fake_get_converter(backend, model_name, device):
        assert backend == "3dtopia"
        assert model_name is None
        assert device is None
        return dummy

    monkeypatch.setattr(server_module, "_get_converter", fake_get_converter)

    response = client.post(
        "/invocations",
        data=json.dumps({"prompt": _sample_prompt()}),
        content_type="application/json",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["backend"] == "3dtopia"
    assert payload["output"] == {"result": "mesh"}
    assert captured["prompt"] == _sample_prompt()


def test_backend_route_invocation(server_module, client, monkeypatch):
    class DummyConverter:
        def __init__(self):
            self.calls = []

        def image_to_3d_3dtopia(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return {"done": True}

    dummy = DummyConverter()

    monkeypatch.setattr(server_module, "_get_converter", lambda *args, **kwargs: dummy)

    response = client.post(
        "/3dtopia",
        data=json.dumps({"prompt": _sample_prompt()}),
        content_type="application/json",
    )

    assert response.status_code == 200
    assert response.get_json()["output"] == {"done": True}
    prompt, kwargs = dummy.calls[0]
    assert prompt == _sample_prompt()


def test_invocations_rejects_non_json(client):
    response = client.post("/invocations", data="{}", content_type="text/plain")
    assert response.status_code == 415


def test_invocations_unknown_backend(client):
    response = client.post(
        "/invocations",
        data=json.dumps({"backend": "non-existent", "prompt": _sample_prompt()}),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_get_converter_caches_instances(server_module, monkeypatch):
    created = []

    class DummyConverter:
        def __init__(self, model_name, device):
            created.append((model_name, device))

    monkeypatch.setattr(server_module, "ImageTo3DConverter", DummyConverter)
    server_module._converter_cache.clear()

    first = server_module._get_converter("3dtopia", None, None)
    second = server_module._get_converter("3dtopia", None, None)

    assert first is second
    assert created == [("3DTopia/3DTopia-XL", server_module._DEFAULT_DEVICE)]


def test_authentication_required_when_enabled(server_module, monkeypatch):
    monkeypatch.setattr(server_module, "_AUTH_ENABLED", True)
    monkeypatch.setattr(server_module, "_AUTH_USERNAME", "user")
    monkeypatch.setattr(server_module, "_AUTH_PASSWORD", "secret")

    dummy = SimpleNamespace(
        image_to_3d_3dtopia=lambda prompt, **kwargs: {"ok": True}
    )
    monkeypatch.setattr(server_module, "_get_converter", lambda *a, **k: dummy)

    with server_module.app.test_client() as test_client:
        response = test_client.post(
            "/invocations",
            data=json.dumps({"prompt": _sample_prompt()}),
            content_type="application/json",
        )
        assert response.status_code == 401

        token = base64.b64encode(b"user:secret").decode("ascii")
        response = test_client.post(
            "/invocations",
            headers={"Authorization": f"Basic {token}"},
            data=json.dumps({"prompt": _sample_prompt()}),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert response.get_json()["output"] == {"ok": True}
