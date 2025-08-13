import unittest
from unittest.mock import patch, MagicMock
from rose.processing.image_to_3d import ImageTo3DConverter


class TestImageTo3DConverter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.dummy_image_path = "dummy.png"

    def test_image_to_3d_trellis(self):
        """Test Trellis 3D conversion."""
        # Patch the TrellisImageTo3DPipeline symbol directly
        DummyPipeline = MagicMock()
        DummyPipeline.from_pretrained.return_value.run.return_value = {"gaussian": "dummy_gaussian", "mesh": "dummy_mesh"}
        DummyPipeline.from_pretrained.return_value.cuda.return_value = None
        
        with patch("rose.processing.image_to_3d.TrellisImageTo3DPipeline", DummyPipeline):
            with patch("rose.processing.image_to_3d.Image", create=True):
                converter = ImageTo3DConverter()
                dummy_image = MagicMock()
                result = converter.image_to_3d(dummy_image)
                
                self.assertIsInstance(result, dict)
                self.assertIn("gaussian", result)
                self.assertIn("mesh", result)

        # Test error handling: TrellisImageTo3DPipeline is None
        with patch("rose.processing.image_to_3d.TrellisImageTo3DPipeline", None):
            with self.assertRaises(ImportError):
                ImageTo3DConverter()

    def test_image_to_3d_vggt(self):
        """Test VGGT 3D conversion."""
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
                result = converter.image_to_3d_vggt(self.dummy_image_path)
                
                self.assertIsInstance(result, dict)
                self.assertIn("depth", result)
                self.assertIn("point_map", result)

        # Test error handling: VGGT and loader are None
        with patch("rose.processing.image_to_3d.VGGT", None), patch("rose.processing.image_to_3d.load_and_preprocess_images", None):
            converter = ImageTo3DConverter()
            with self.assertRaises(ImportError):
                converter.image_to_3d_vggt(self.dummy_image_path)

    def test_image_to_3d_hunyuan(self):
        """Test Hunyuan 3D conversion."""
        # Patch Hunyuan3DDiTFlowMatchingPipeline symbol directly
        DummyPipeline = MagicMock()
        DummyPipeline.from_pretrained = MagicMock()
        DummyPipeline.from_pretrained.return_value.__call__ = MagicMock(return_value=["dummy_mesh"])
        
        with patch("rose.processing.image_to_3d.Hunyuan3DDiTFlowMatchingPipeline", DummyPipeline):
            converter = ImageTo3DConverter()
            result = converter.image_to_3d_hunyuan(self.dummy_image_path)
            
            self.assertEqual(result, "dummy_mesh")

        # Test error handling: Hunyuan3DDiTFlowMatchingPipeline is None
        with patch("rose.processing.image_to_3d.Hunyuan3DDiTFlowMatchingPipeline", None):
            converter = ImageTo3DConverter()
            with self.assertRaises(ImportError):
                converter.image_to_3d_hunyuan(self.dummy_image_path)

    def test_image_to_3d_converter_init(self):
        """Test ImageTo3DConverter initialization."""
        # Test successful initialization
        with patch("rose.processing.image_to_3d.TrellisImageTo3DPipeline", MagicMock()), \
             patch("rose.processing.image_to_3d.VGGT", MagicMock()), \
             patch("rose.processing.image_to_3d.Hunyuan3DDiTFlowMatchingPipeline", MagicMock()):
            
            converter = ImageTo3DConverter()
            self.assertIsNotNone(converter)

    def test_image_to_3d_converter_with_different_inputs(self):
        """Test 3D conversion with different input types."""
        # Test with string path
        with patch("rose.processing.image_to_3d.TrellisImageTo3DPipeline", MagicMock()), \
             patch("rose.processing.image_to_3d.Image", create=True):
            
            converter = ImageTo3DConverter()
            dummy_image = MagicMock()
            
            result = converter.image_to_3d(dummy_image)
            self.assertIsInstance(result, dict)

        # Test with PIL Image
        with patch("rose.processing.image_to_3d.TrellisImageTo3DPipeline", MagicMock()):
            converter = ImageTo3DConverter()
            dummy_pil_image = MagicMock()
            
            result = converter.image_to_3d(dummy_pil_image)
            self.assertIsInstance(result, dict)

    def test_error_handling_missing_dependencies(self):
        """Test error handling when dependencies are missing."""
        # Test when all dependencies are missing
        with patch("rose.processing.image_to_3d.TrellisImageTo3DPipeline", None), \
             patch("rose.processing.image_to_3d.VGGT", None), \
             patch("rose.processing.image_to_3d.Hunyuan3DDiTFlowMatchingPipeline", None):
            
            with self.assertRaises(ImportError):
                ImageTo3DConverter()

    def test_performance_measurement(self):
        """Test that 3D conversion includes performance measurement."""
        import time
        
        DummyPipeline = MagicMock()
        DummyPipeline.from_pretrained.return_value.run.return_value = {"gaussian": "dummy_gaussian", "mesh": "dummy_mesh"}
        DummyPipeline.from_pretrained.return_value.cuda.return_value = None
        
        with patch("rose.processing.image_to_3d.TrellisImageTo3DPipeline", DummyPipeline):
            with patch("rose.processing.image_to_3d.Image", create=True):
                converter = ImageTo3DConverter()
                dummy_image = MagicMock()
                
                start_time = time.time()
                result = converter.image_to_3d(dummy_image)
                processing_time = time.time() - start_time
                
                # Should complete in reasonable time (less than 60 seconds)
                self.assertLess(processing_time, 60.0)
                self.assertIsInstance(result, dict)
