import pytest
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

from PIL import Image as PILImage
from rose.processing.image_to_3d import ImageTo3DConverter

# Dummy image path and PIL image for tests
dummy_image_path = "dummy.png"


def test_image_to_3d_trellis(monkeypatch):
    # Patch the TrellisImageTo3DPipeline symbol directly
    DummyPipeline = MagicMock()
    DummyPipeline.from_pretrained.return_value.run.return_value = {"gaussian": "dummy_gaussian", "mesh": "dummy_mesh"}
    DummyPipeline.from_pretrained.return_value.cuda.return_value = None
    with patch("rose.processing.image_to_3d.TrellisImageTo3DPipeline", DummyPipeline):
        with patch("rose.processing.image_to_3d.Image", create=True):
            converter = ImageTo3DConverter()
            dummy_image = MagicMock()
            result = converter.image_to_3d(dummy_image)
            assert isinstance(result, dict)
            assert "gaussian" in result
            assert "mesh" in result

    # Test error handling: TrellisImageTo3DPipeline is None
    with patch("rose.processing.image_to_3d.TrellisImageTo3DPipeline", None):
        with pytest.raises(ImportError):
            ImageTo3DConverter()


def test_image_to_3d_vggt(monkeypatch):
    # Patch VGGT and load_and_preprocess_images symbols directly
    DummyVGGT = MagicMock()
    DummyVGGT.from_pretrained = MagicMock()
    DummyVGGT.from_pretrained.return_value.to = MagicMock()
    DummyVGGT.from_pretrained.return_value.to.return_value.return_value = {"depth": "dummy_depth", "point_map": "dummy_point_map"}
    DummyLoader = MagicMock()
    DummyLoader.return_value.to.return_value = "dummy_images"
    with patch("rose.processing.image_to_3d.VGGT", DummyVGGT), \
         patch("rose.processing.image_to_3d.load_and_preprocess_images", DummyLoader):
        converter = ImageTo3DConverter(model_name='facebook/VGGT-1B')
        with patch("torch.no_grad"), patch("torch.cuda.amp.autocast"):
            result = converter.image_to_3d_vggt(dummy_image_path)
            assert isinstance(result, dict)
            assert "depth" in result
            assert "point_map" in result

    # Test error handling: VGGT and loader are None
    with patch("rose.processing.image_to_3d.VGGT", None), patch("rose.processing.image_to_3d.load_and_preprocess_images", None):
        converter = ImageTo3DConverter(model_name='facebook/VGGT-1B')
        with pytest.raises(ImportError):
            converter.image_to_3d_vggt(dummy_image_path)


def test_image_to_3d_hunyuan(monkeypatch):
    # Patch Hunyuan3DDiTFlowMatchingPipeline symbol directly
    DummyPipeline = MagicMock()
    DummyPipeline.from_pretrained = MagicMock()
    DummyPipeline.from_pretrained.return_value.return_value = ["dummy_mesh"]
    with patch("rose.processing.image_to_3d.Hunyuan3DDiTFlowMatchingPipeline", DummyPipeline):
        converter = ImageTo3DConverter(model_name='tencent/Hunyuan3D')
        result = converter.image_to_3d_hunyuan(dummy_image_path)
        assert result == "dummy_mesh"

    # Test error handling: Hunyuan3DDiTFlowMatchingPipeline is None
    with patch("rose.processing.image_to_3d.Hunyuan3DDiTFlowMatchingPipeline", None):
        with pytest.raises(ImportError):
            ImageTo3DConverter(model_name='tencent/Hunyuan3D')


def test_image_to_3d_triposr(monkeypatch):
    dummy_image = MagicMock()
    dummy_image.convert.return_value = dummy_image
    image_context = MagicMock()
    image_context.__enter__.return_value = dummy_image
    image_context.__exit__.return_value = False

    runner_instance = MagicMock()
    runner_instance.to.return_value = runner_instance
    runner_instance.eval.return_value = runner_instance
    runner_instance.renderer = MagicMock()
    runner_instance.renderer.set_chunk_size.return_value = None
    runner_instance.return_value = ["scene_code"]
    runner_instance.extract_mesh.return_value = ["dummy_mesh"]

    DummyRunnerClass = MagicMock()
    DummyRunnerClass.from_pretrained.return_value = runner_instance

    with patch("rose.processing.image_to_3d.TripoSRRunnerClass", DummyRunnerClass), \
         patch("rose.processing.image_to_3d.TripoSRModel", None), \
         patch("rose.processing.image_to_3d.TripoSRProcessor", None), \
         patch("rose.processing.image_to_3d.Image", MagicMock(open=MagicMock(return_value=image_context))), \
         patch("torch.no_grad", return_value=nullcontext()):
        converter = ImageTo3DConverter(model_name='stabilityai/TripoSR', device='cpu')
        result = converter.image_to_3d_triposr(dummy_image_path)
        assert result == "dummy_mesh"

    with patch("rose.processing.image_to_3d.TripoSRRunnerClass", None), \
         patch("rose.processing.image_to_3d.TripoSRModel", None), \
         patch("rose.processing.image_to_3d.TripoSRProcessor", None):
        with pytest.raises(ImportError):
            ImageTo3DConverter(model_name='stabilityai/TripoSR')


def test_image_to_3d_triposr_cube_image(monkeypatch):
    cube_path = Path(__file__).with_name("CubeImage.png")

    runner_instance = MagicMock()
    runner_instance.to.return_value = runner_instance
    runner_instance.eval.return_value = runner_instance
    runner_instance.renderer = MagicMock()
    runner_instance.renderer.set_chunk_size.return_value = None
    runner_instance.return_value = ["scene_code"]
    runner_instance.extract_mesh.return_value = ["dummy_mesh"]

    DummyRunnerClass = MagicMock()
    DummyRunnerClass.from_pretrained.return_value = runner_instance

    with patch("rose.processing.image_to_3d.TripoSRRunnerClass", DummyRunnerClass), \
         patch("rose.processing.image_to_3d.TripoSRModel", None), \
         patch("rose.processing.image_to_3d.TripoSRProcessor", None), \
         patch("rose.processing.image_to_3d.Image", PILImage), \
         patch("torch.no_grad", return_value=nullcontext()):
        converter = ImageTo3DConverter(model_name='stabilityai/TripoSR', device='cpu')
        result = converter.image_to_3d_triposr(str(cube_path))
        assert result == "dummy_mesh"

    with patch("rose.processing.image_to_3d.TripoSRRunnerClass", None), \
         patch("rose.processing.image_to_3d.TripoSRModel", None), \
         patch("rose.processing.image_to_3d.TripoSRProcessor", None):
        with pytest.raises(ImportError):
            ImageTo3DConverter(model_name='stabilityai/TripoSR')


def test_image_to_3d_triposr_cube_real_execution(tmp_path):
    cube_path = Path(__file__).with_name("CubeImage.png")
    if not cube_path.exists():
        pytest.skip("Cube test image missing")

    if not ImageTo3DConverter.is_triposr_available():
        pytest.skip("TripoSR dependencies not installed")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    converter = ImageTo3DConverter(model_name='stabilityai/TripoSR', device=device)
    mesh = converter.image_to_3d_triposr(str(cube_path))
    if mesh is None:
        pytest.skip("TripoSR model returned no mesh; ensure weights are available")

    output_file = tmp_path / "triposr_cube_mesh.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(mesh))

    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8").strip()
