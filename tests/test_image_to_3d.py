import pytest
from unittest.mock import patch, MagicMock
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
    DummyVGGT.from_pretrained.return_value.to.return_value.__call__ = MagicMock(return_value={"depth": "dummy_depth", "point_map": "dummy_point_map"})
    DummyLoader = MagicMock()
    DummyLoader.return_value.to.return_value = "dummy_images"
    with patch("rose.processing.image_to_3d.VGGT", DummyVGGT), \
         patch("rose.processing.image_to_3d.load_and_preprocess_images", DummyLoader):
        converter = ImageTo3DConverter()
        with patch("torch.no_grad"), patch("torch.cuda.amp.autocast"):
            result = converter.image_to_3d_vggt(dummy_image_path)
            assert isinstance(result, dict)
            assert "depth" in result
            assert "point_map" in result

    # Test error handling: VGGT and loader are None
    with patch("rose.processing.image_to_3d.VGGT", None), patch("rose.processing.image_to_3d.load_and_preprocess_images", None):
        converter = ImageTo3DConverter()
        with pytest.raises(ImportError):
            converter.image_to_3d_vggt(dummy_image_path)

def test_image_to_3d_hunyuan(monkeypatch):
    # Patch Hunyuan3DDiTFlowMatchingPipeline symbol directly
    DummyPipeline = MagicMock()
    DummyPipeline.from_pretrained = MagicMock()
    DummyPipeline.from_pretrained.return_value.__call__ = MagicMock(return_value=["dummy_mesh"])
    with patch("rose.processing.image_to_3d.Hunyuan3DDiTFlowMatchingPipeline", DummyPipeline):
        converter = ImageTo3DConverter()
        result = converter.image_to_3d_hunyuan(dummy_image_path)
        assert result == "dummy_mesh"

    # Test error handling: Hunyuan3DDiTFlowMatchingPipeline is None
    with patch("rose.processing.image_to_3d.Hunyuan3DDiTFlowMatchingPipeline", None):
        converter = ImageTo3DConverter()
        with pytest.raises(ImportError):
            converter.image_to_3d_hunyuan(dummy_image_path)