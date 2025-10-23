import sys
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from image_to_3d import ImageTo3DConverter


dummy_image_path = "dummy.png"


def _ensure_dummy_torch(monkeypatch):
    dummy_torch = types.ModuleType("torch")
    dummy_torch.no_grad = lambda: nullcontext()
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)


def _build_dummy_runner():
    runner_instance = MagicMock()
    runner_instance.to.return_value = runner_instance
    runner_instance.eval.return_value = runner_instance
    runner_instance.renderer = MagicMock()
    runner_instance.renderer.set_chunk_size.return_value = None
    runner_instance.return_value = ["scene_code"]
    runner_instance.extract_mesh.return_value = ["dummy_mesh"]
    DummyRunnerClass = MagicMock()
    DummyRunnerClass.from_pretrained.return_value = runner_instance
    return DummyRunnerClass, runner_instance


def test_image_to_3d_triposr_official_backend_path(monkeypatch):
    _ensure_dummy_torch(monkeypatch)
    DummyRunnerClass, _ = _build_dummy_runner()

    dummy_image = MagicMock()
    dummy_image.convert.return_value = dummy_image
    image_context = MagicMock()
    image_context.__enter__.return_value = dummy_image
    image_context.__exit__.return_value = False

    with patch("image_to_3d.TripoSRRunnerClass", DummyRunnerClass), \
         patch("image_to_3d.TripoSRModel", None), \
         patch("image_to_3d.TripoSRProcessor", None), \
         patch("image_to_3d.Image", MagicMock(open=MagicMock(return_value=image_context))):
        converter = ImageTo3DConverter(device="cpu")
        result = converter.image_to_3d(dummy_image_path)
        assert result == "dummy_mesh"


def test_image_to_3d_triposr_official_backend_pil(monkeypatch):
    _ensure_dummy_torch(monkeypatch)
    DummyRunnerClass, _ = _build_dummy_runner()

    with patch("image_to_3d.TripoSRRunnerClass", DummyRunnerClass), \
         patch("image_to_3d.TripoSRModel", None), \
         patch("image_to_3d.TripoSRProcessor", None), \
         patch("image_to_3d.Image", PILImage):
        converter = ImageTo3DConverter(device="cpu")
        pil_image = PILImage.new("RGB", (4, 4), color="red")
        result = converter.image_to_3d(pil_image)
        assert result == "dummy_mesh"


def test_image_to_3d_triposr_transformers_backend(monkeypatch):
    _ensure_dummy_torch(monkeypatch)

    dummy_image = MagicMock()
    dummy_image.convert.return_value = dummy_image
    image_context = MagicMock()
    image_context.__enter__.return_value = dummy_image
    image_context.__exit__.return_value = False

    processor_instance = MagicMock()
    processor_call_output = MagicMock()
    processor_call_output.to.return_value = processor_call_output
    processor_instance.return_value = processor_call_output
    processor_instance.generate_mesh.return_value = "generated_mesh"
    DummyProcessor = MagicMock()
    DummyProcessor.from_pretrained.return_value = processor_instance

    model_instance = MagicMock()
    model_instance.to.return_value = model_instance
    model_instance.eval.return_value = model_instance
    model_output = MagicMock()
    model_output.meshes = ["mesh_tensor"]
    model_instance.return_value = model_output
    DummyModel = MagicMock()
    DummyModel.from_pretrained.return_value = model_instance

    with patch("image_to_3d.TripoSRRunnerClass", None), \
         patch("image_to_3d.TripoSRProcessor", DummyProcessor), \
         patch("image_to_3d.TripoSRModel", DummyModel), \
         patch("image_to_3d.Image", MagicMock(open=MagicMock(return_value=image_context))):
        converter = ImageTo3DConverter(device="cpu")
        result = converter.image_to_3d(dummy_image_path)
        assert result == "generated_mesh"
        DummyProcessor.from_pretrained.assert_called_once()
        DummyModel.from_pretrained.assert_called_once()


def test_image_to_3d_triposr_invalid_input(monkeypatch):
    _ensure_dummy_torch(monkeypatch)
    DummyRunnerClass, _ = _build_dummy_runner()

    with patch("image_to_3d.TripoSRRunnerClass", DummyRunnerClass), \
         patch("image_to_3d.TripoSRModel", None), \
         patch("image_to_3d.TripoSRProcessor", None), \
         patch("image_to_3d.Image", PILImage):
        converter = ImageTo3DConverter(device="cpu")
        with pytest.raises(TypeError):
            converter.image_to_3d(42)


def test_image_to_3d_triposr_missing_dependencies(monkeypatch):
    with patch("image_to_3d.TripoSRRunnerClass", None), \
         patch("image_to_3d.TripoSRModel", None), \
         patch("image_to_3d.TripoSRProcessor", None):
        with pytest.raises(ImportError):
            ImageTo3DConverter()


def test_image_to_3d_triposr_cube_real_execution(monkeypatch, tmp_path):
    cube_path = Path(__file__).with_name("CubeImage.png")
    if not cube_path.exists():
        pytest.skip("Cube test image missing")

    if not ImageTo3DConverter.is_triposr_available():
        pytest.skip("TripoSR dependencies not installed")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    converter = ImageTo3DConverter(model_name="stabilityai/TripoSR", device=device)
    mesh = converter.image_to_3d(str(cube_path))
    if mesh is None:
        pytest.skip("TripoSR model returned no mesh; ensure weights are available")

    output_file = tmp_path / "triposr_cube_mesh.txt"
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(str(mesh))

    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8").strip()
