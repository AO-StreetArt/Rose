import importlib
import os
from typing import Any, Optional

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from trellis.pipelines import TrellisImageTo3DPipeline
except ImportError:
    TrellisImageTo3DPipeline = None

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

try:
    from tsr.system import TSR
except ImportError:
    TSR = None

try:
    from transformers import TripoSRModel, TripoSRProcessor
except ImportError:
    TripoSRModel = None
    TripoSRProcessor = None


def _import_triposr_runner() -> Optional[Any]:
    """Attempt to import the official TripoSR pipeline class from known locations."""
    if TSR is not None:
        return TSR
    candidates = (
        ("triposr.pipeline", "TripoSRPipeline"),
        ("triposr.pipelines.triposr", "TripoSRPipeline"),
        ("triposr", "TripoSR"),
        ("triposr.inference", "TripoSRInference"),
        ("tsr", "TSR"),
        ("tsr.system", "TSR"),
    )
    for module_path, attr in candidates:
        try:
            module = importlib.import_module(module_path)
            runner = getattr(module, attr)
            return runner
        except (ImportError, AttributeError):
            continue
    return None


TripoSRRunnerClass = _import_triposr_runner()


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
        self.model_name = model_name
        self.device = device
        self.triposr_processor: Optional[Any] = None
        self.triposr_model: Optional[Any] = None
        self.triposr_runner: Optional[Any] = None
        self.triposr_pipeline: Optional[Any] = None
        self.triposr_backend: Optional[str] = None
        self.triposr_model_name: Optional[str] = None
        self.pipeline: Optional[Any] = None
        if self.model_name == 'microsoft/TRELLIS-image-large':
            if TrellisImageTo3DPipeline is None:
                raise ImportError("TRELLIS image-to-3D pipeline is not installed.")
            os.environ["SPCONV_ALGO"] = "native"  # Recommended for single runs
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
        elif self.model_name == 'facebook/VGGT-1B':
            if VGGT is not None and load_and_preprocess_images is not None:
                self.pipeline = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        elif self.model_name == 'tencent/Hunyuan3D':
            if Hunyuan3DDiTFlowMatchingPipeline is None:
                raise ImportError("Hunyuan3D pipeline is not installed.")
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_name)
        elif self.model_name == 'stabilityai/TripoSR':
            if TripoSRRunnerClass is not None:
                self.triposr_backend = "official"
                self.triposr_model_name = None
            elif self.is_triposr_transformers_available():
                self.triposr_backend = "transformers"
                self.triposr_model_name = None
            else:
                raise ImportError(
                    "TripoSR is not installed. Install the packaged release via"
                    " `pip install triposr` or upgrade transformers to a version that"
                    " bundles TripoSR."
                )
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        if self.pipeline is not None and device == "cuda":
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
        images = load_and_preprocess_images([image_path]).to(device)
        if self.model_name == 'facebook/VGGT-1B' and self.pipeline is not None:
            model = self.pipeline
        else:
            model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
            if self.model_name == 'facebook/VGGT-1B':
                self.pipeline = model
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
            mesh = self.pipeline(image=image_path, **kwargs)[0]
            return mesh
        except Exception as e:
            print(f"Error during Hunyuan3D image-to-3D conversion: {e}")
            return None

    @staticmethod
    def is_triposr_available() -> bool:
        """Returns True if either the official or Transformers TripoSR bindings are available."""
        return TripoSRRunnerClass is not None or ImageTo3DConverter.is_triposr_transformers_available()

    @staticmethod
    def is_triposr_transformers_available() -> bool:
        """Returns True if the Transformers TripoSR model and processor are available."""
        return TripoSRModel is not None and TripoSRProcessor is not None

    def image_to_3d_triposr(self, image_path: str, model_name: str = "stabilityai/TripoSR") -> Optional[Any]:
        """
        Converts an image to a 3D mesh using the TripoSR model distributed via Hugging Face Transformers.
        Args:
            image_path (str): Path to the input image file.
            model_name (str): Name or path of the TripoSR model to use.
        Returns:
            Any: The generated 3D mesh object, or None if failed.
        """
        if not self.is_triposr_available():
            raise ImportError(
                "TripoSR is not installed. Install the packaged release via"
                " `pip install triposr` or upgrade transformers to include TripoSR."
            )
        if Image is None:
            raise ImportError("Pillow is required to load images for TripoSR.")

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except Exception as e:
            print(f"Error opening image for TripoSR conversion: {e}")
            return None

        if self.triposr_backend == "official":
            runner = self._ensure_official_triposr_runner(model_name)
            if runner is None:
                return None
            mesh = self._run_official_triposr(runner, image, image_path)
            return mesh
        else:
            return self._run_transformers_triposr(model_name, image)

    def _ensure_official_triposr_runner(self, model_name: str) -> Optional[Any]:
        if TripoSRRunnerClass is None:
            print("Official TripoSR pipeline is not available.")
            return None
        if self.triposr_runner is not None and self.triposr_model_name == model_name:
            return self.triposr_runner
        runner_cls = TripoSRRunnerClass
        try:
            if hasattr(runner_cls, "from_pretrained"):
                try:
                    runner = runner_cls.from_pretrained(model_name)
                except TypeError:
                    try:
                        runner = runner_cls.from_pretrained(
                            model_name,
                            config_name="config.yaml",
                            weight_name="model.ckpt",
                        )
                    except Exception as inner_exc:
                        print(f"Error loading TripoSR pipeline: {inner_exc}")
                        return None
            else:
                runner = runner_cls()
        except Exception as exc:
            print(f"Error loading TripoSR pipeline: {exc}")
            return None
        runner = self._move_triposr_runner_to_device(runner)
        if runner is None:
            return None
        self.triposr_runner = runner
        self.triposr_model_name = model_name
        return runner

    def _move_triposr_runner_to_device(self, runner: Any) -> Optional[Any]:
        try:
            if hasattr(runner, "to"):
                runner = runner.to(self.device)
            elif self.device == "cuda" and hasattr(runner, "cuda"):
                runner.cuda()
            if hasattr(runner, "eval"):
                runner.eval()
        except Exception as exc:
            print(f"Error moving TripoSR pipeline to device: {exc}")
            return None
        return runner

    def _run_official_triposr(self, runner: Any, image: "Image.Image", image_path: str) -> Optional[Any]:
        if hasattr(runner, "extract_mesh") and callable(getattr(runner, "extract_mesh")):
            mesh = self._run_tsr_pipeline(runner, image)
            if mesh is not None:
                return mesh
        call_attempts = (
            {"image": image},
            {"images": image},
            {"image_path": image_path},
            (image,),
            (image_path,),
            (),
        )
        for attempt in call_attempts:
            try:
                if isinstance(attempt, dict):
                    outputs = runner(**attempt)
                else:
                    outputs = runner(*attempt)
                return self._extract_triposr_mesh(outputs)
            except TypeError:
                continue
            except Exception as exc:
                print(f"Error during TripoSR official pipeline execution: {exc}")
                return None
        if hasattr(runner, "run"):
            try:
                outputs = runner.run(image=image)
                return self._extract_triposr_mesh(outputs)
            except Exception as exc:
                print(f"Error during TripoSR official pipeline execution: {exc}")
                return None
        print("Unable to execute TripoSR pipeline with provided interfaces.")
        return None

    def _run_tsr_pipeline(self, runner: Any, image: "Image.Image") -> Optional[Any]:
        try:
            import torch
        except ImportError:
            print("Torch is required to execute the TripoSR TSR pipeline.")
            return None

        try:
            if hasattr(runner, "renderer") and hasattr(runner.renderer, "set_chunk_size"):
                # Use a default chunk size that mirrors the reference CLI implementation
                runner.renderer.set_chunk_size(8192)
            with torch.no_grad():
                scene_codes = runner([image], device=self.device)
            meshes = runner.extract_mesh(scene_codes, has_vertex_color=True)
        except Exception as exc:
            print(f"Error during TSR pipeline execution: {exc}")
            return None

        if isinstance(meshes, (list, tuple)) and meshes:
            return meshes[0]
        return meshes

    def _run_transformers_triposr(self, model_name: str, image: "Image.Image") -> Optional[Any]:
        if not self.is_triposr_transformers_available():
            print("Transformers TripoSR backend not available.")
            return None
        if (
            self.triposr_processor is None
            or self.triposr_model is None
            or self.triposr_model_name != model_name
        ):
            self.triposr_processor = TripoSRProcessor.from_pretrained(model_name)
            self.triposr_model = TripoSRModel.from_pretrained(model_name).to(self.device)
            self.triposr_model.eval()
            self.triposr_model_name = model_name
        processor = self.triposr_processor
        model = self.triposr_model

        inputs = processor(images=image, return_tensors="pt").to(self.device)
        import torch

        with torch.no_grad():
            outputs = model(**inputs)
        meshes = getattr(outputs, "meshes", None)
        if not meshes:
            return None
        mesh = processor.generate_mesh(meshes[0])
        return mesh

    @staticmethod
    def _extract_triposr_mesh(outputs: Any) -> Optional[Any]:
        if outputs is None:
            return None
        if isinstance(outputs, dict):
            if "mesh" in outputs:
                return outputs["mesh"]
            meshes = outputs.get("meshes")
            if isinstance(meshes, (list, tuple)) and meshes:
                return meshes[0]
            return meshes
        if hasattr(outputs, "mesh"):
            mesh = getattr(outputs, "mesh")
            if mesh is not None:
                return mesh
        if hasattr(outputs, "meshes"):
            meshes = getattr(outputs, "meshes")
            if isinstance(meshes, (list, tuple)) and meshes:
                return meshes[0]
            return meshes
        if isinstance(outputs, (list, tuple)) and outputs:
            return outputs[0]
        return outputs
