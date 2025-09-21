import unittest
from unittest.mock import patch
from PIL import Image
import numpy as np


class DummyControlNet:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()


class DummyPipe:
    def __init__(self, *args, **kwargs):
        self.called = False

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    # Diffusers API surface used in our code
    def enable_vae_slicing(self):
        pass

    def enable_vae_tiling(self):
        pass

    def enable_model_cpu_offload(self):
        pass

    def to(self, *_args, **_kwargs):
        return self

    def __call__(self, *args, **kwargs):
        # Return an object with an images list containing a PIL image
        class R:
            pass

        r = R()
        r.images = [Image.new("RGB", (64, 64), color=(128, 128, 128))]
        return r


class DummyDepthEstimator:
    def estimate_depth(self, img):
        # Simple plane depth gradient
        w, h = img.size
        x = np.linspace(0, 1, w, dtype=np.float32)
        y = np.linspace(0, 1, h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        return (xv + yv) / 2.0


class TestTextureGenerator(unittest.TestCase):
    @patch("rose.processing.texture_generator.DepthEstimator", DummyDepthEstimator)
    @patch("rose.processing.texture_generator.StableDiffusionXLControlNetPipeline", DummyPipe)
    @patch("rose.processing.texture_generator.ControlNetModel", DummyControlNet)
    def test_init_and_generate_albedo(self):
        from rose.processing.texture_generator import TextureGenerator

        tg = TextureGenerator(device="cpu")
        control = Image.new("RGB", (64, 64), color=(255, 255, 255))
        img = tg.generate_albedo(
            uv_condition_image=control,
            prompt="wood grain",
            seed=0,
            num_inference_steps=2,
        )
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.size, (64, 64))

    @patch("rose.processing.texture_generator.DepthEstimator", DummyDepthEstimator)
    @patch("rose.processing.texture_generator.StableDiffusionXLControlNetPipeline", DummyPipe)
    @patch("rose.processing.texture_generator.ControlNetModel", DummyControlNet)
    def test_infer_normal_and_roughness(self):
        from rose.processing.texture_generator import TextureGenerator

        tg = TextureGenerator(device="cpu")
        albedo = Image.new("RGB", (32, 32), color=(200, 200, 200))
        normal = tg.infer_normal_from_albedo(albedo)
        rough = tg.infer_roughness_from_albedo(albedo)
        self.assertIsInstance(normal, Image.Image)
        self.assertEqual(normal.mode, "RGB")
        self.assertEqual(normal.size, (32, 32))
        self.assertIsInstance(rough, Image.Image)
        self.assertEqual(rough.mode, "L")
        self.assertEqual(rough.size, (32, 32))

    @patch("rose.processing.texture_generator.DepthEstimator", DummyDepthEstimator)
    @patch("rose.processing.texture_generator.StableDiffusionXLControlNetPipeline", DummyPipe)
    @patch("rose.processing.texture_generator.ControlNetModel", DummyControlNet)
    def test_generate_textures_bundle(self):
        from rose.processing.texture_generator import TextureGenerator

        tg = TextureGenerator(device="cpu")
        control = Image.new("RGB", (48, 48), color=(255, 255, 255))
        out = tg.generate_textures(
            control_image=control,
            prompt="rusted metal",
            steps=2,
        )
        self.assertIn("albedo", out)
        self.assertIn("normal", out)
        self.assertIn("roughness", out)
        self.assertIsInstance(out["albedo"], Image.Image)
        self.assertIsInstance(out["normal"], Image.Image)
        self.assertIsInstance(out["roughness"], Image.Image)


if __name__ == "__main__":
    unittest.main()

