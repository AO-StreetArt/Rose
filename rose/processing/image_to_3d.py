import os
from typing import Any, Optional

try:
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
except ImportError:
    TrellisImageTo3DPipeline = None
    Image = None

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
except ImportError:
    VGGT = None
    load_and_preprocess_images = None

try:
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
except ImportError:
    Hunyuan3DDiTFlowMatchingPipeline = None


class ImageTo3DConverter:
    """
    Converts images to 3D data using the Microsoft TRELLIS pre-trained model.
    This class provides an interface to generate 3D representations (Gaussian point clouds, meshes, etc.) from a single image.
    """
    def __init__(self, model_name: str = "microsoft/TRELLIS-image-large", device: str = "cuda"):
        """
        Initializes the TRELLIS image-to-3D pipeline.
        Args:
            model_name (str): Name or path of the TRELLIS model to use.
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        if TrellisImageTo3DPipeline is None:
            raise ImportError("trellis package is not installed. Please install TRELLIS and its dependencies.")
        self.model_name = model_name
        self.device = device
        os.environ["SPCONV_ALGO"] = "native"  # Recommended for single runs
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
        if device == "cuda":
            self.pipeline.cuda()

    def image_to_3d(self, image: "Image.Image", seed: int = 1, **kwargs) -> Optional[Any]:
        """
        Converts a PIL image to 3D data using the TRELLIS pipeline.
        Args:
            image (PIL.Image.Image): Input image.
            seed (int): Random seed for generation.
            **kwargs: Additional parameters for the pipeline's run method.
        Returns:
            dict: Outputs containing 3D representations (e.g., 'gaussian', 'mesh', etc.), or None if failed.
        """
        try:
            outputs = self.pipeline.run(image, seed=seed, **kwargs)
            return outputs
        except Exception as e:
            print(f"Error during TRELLIS image-to-3D conversion: {e}")
            return None

    @staticmethod
    def is_vggt_available() -> bool:
        """
        Returns True if the VGGT model and utilities are available.
        """
        return VGGT is not None and load_and_preprocess_images is not None

    def image_to_3d_vggt(self, image_path: str, device: str = "cuda") -> Optional[dict]:
        """
        Converts an image to 3D data using the Facebook VGGT pre-trained model.
        Args:
            image_path (str): Path to the input image file.
            device (str): Device to run the model on ("cuda" or "cpu").
        Returns:
            dict: Outputs containing 3D attributes (e.g., depth, point map, camera params), or None if failed.
        """
        if VGGT is None or load_and_preprocess_images is None:
            raise ImportError("vggt package is not installed. Please install vggt and its dependencies.")
        import torch
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        images = load_and_preprocess_images([image_path]).to(device)
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images)
            return predictions
        except Exception as e:
            print(f"Error during VGGT image-to-3D conversion: {e}")
            return None

    @staticmethod
    def is_hunyuan3d_available() -> bool:
        """
        Returns True if the Hunyuan3D 2.0 pipeline is available.
        """
        return Hunyuan3DDiTFlowMatchingPipeline is not None

    def image_to_3d_hunyuan(self, image_path: str, model_name: str = "tencent/Hunyuan3D-2", **kwargs):
        """
        Converts an image to 3D data using the Tencent Hunyuan3D 2.0 pre-trained model.
        Args:
            image_path (str): Path to the input image file.
            model_name (str): Name or path of the Hunyuan3D model to use.
            **kwargs: Additional parameters for the pipeline's __call__ method.
        Returns:
            mesh: The generated 3D mesh object, or None if failed.
        """
        if Hunyuan3DDiTFlowMatchingPipeline is None:
            raise ImportError("Hunyuan3D 2.0 is not installed. Please install hunyuan3d2 and its dependencies.")
        try:
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_name)
            mesh = pipeline(image=image_path, **kwargs)[0]
            return mesh
        except Exception as e:
            print(f"Error during Hunyuan3D image-to-3D conversion: {e}")
            return None
