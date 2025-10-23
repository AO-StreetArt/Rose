import importlib
from typing import Any, Optional, Union

try:
    from PIL import Image
except ImportError:
    Image = None

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
    """Converts images to 3D meshes using the TripoSR model."""

    def __init__(self, model_name: str = "stabilityai/TripoSR", device: str = "cuda"):
        """
        Initialise the TripoSR image-to-3D conversion utilities.
        Args:
            model_name (str): Name or path of the TripoSR model to use.
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        self.model_name = model_name
        self.device = device
        self.triposr_processor: Optional[Any] = None
        self.triposr_model: Optional[Any] = None
        self.triposr_runner: Optional[Any] = None
        self.triposr_backend: Optional[str] = None
        self.triposr_model_name: Optional[str] = None

        if TripoSRRunnerClass is not None:
            self.triposr_backend = "official"
        elif self.is_triposr_transformers_available():
            self.triposr_backend = "transformers"
        else:
            raise ImportError(
                "TripoSR is not installed. Install the packaged release via"
                " `pip install triposr` or upgrade transformers to a version that"
                " bundles TripoSR."
            )

    def image_to_3d(
        self,
        image: Union[str, "Image.Image"],
        model_name: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Converts an input image (path or PIL image) to a 3D mesh using TripoSR.
        Args:
            image (Union[str, PIL.Image.Image]): Input image path or PIL image.
            model_name (Optional[str]): Optional override for the TripoSR model identifier.
        Returns:
            Any: The generated 3D mesh object, or None if failed.
        """
        return self.image_to_3d_triposr(image, model_name=model_name)

    @staticmethod
    def is_triposr_available() -> bool:
        """Returns True if either the official or Transformers TripoSR bindings are available."""
        return TripoSRRunnerClass is not None or ImageTo3DConverter.is_triposr_transformers_available()

    @staticmethod
    def is_triposr_transformers_available() -> bool:
        """Returns True if the Transformers TripoSR model and processor are available."""
        return TripoSRModel is not None and TripoSRProcessor is not None

    def image_to_3d_triposr(
        self,
        image_input: Union[str, "Image.Image"],
        model_name: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Converts an image to a 3D mesh using the TripoSR model.
        Args:
            image_input (Union[str, PIL.Image.Image]): Input image (path or PIL image).
            model_name (Optional[str]): Name or path of the TripoSR model to use.
        Returns:
            Any: The generated 3D mesh object, or None if failed.
        """
        if not self.is_triposr_available():
            raise ImportError(
                "TripoSR is not installed. Install the packaged release via"
                " `pip install triposr` or upgrade transformers to include TripoSR."
            )
        resolved_model_name = model_name or self.model_name

        pil_image: Optional["Image.Image"] = None
        image_path: Optional[str] = None

        if isinstance(image_input, str):
            image_path = image_input
            if Image is None:
                raise ImportError("Pillow is required to load images for TripoSR.")
            try:
                with Image.open(image_input) as loaded_image:
                    pil_image = loaded_image.convert("RGB")
            except Exception as exc:
                print(f"Error opening image for TripoSR conversion: {exc}")
                return None
        else:
            if Image is None or not isinstance(image_input, Image.Image):
                raise TypeError(
                    "image_input must be a path to an image file or a PIL.Image.Image instance."
                )
            pil_image = image_input.convert("RGB")

        if pil_image is None:
            print("No image data available for TripoSR conversion.")
            return None

        if self.triposr_backend == "official":
            runner = self._ensure_official_triposr_runner(resolved_model_name)
            if runner is None:
                return None
            return self._run_official_triposr(runner, pil_image, image_path)
        return self._run_transformers_triposr(resolved_model_name, pil_image)

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

    def _run_official_triposr(
        self,
        runner: Any,
        image: "Image.Image",
        image_path: Optional[str],
    ) -> Optional[Any]:
        if hasattr(runner, "extract_mesh") and callable(getattr(runner, "extract_mesh")):
            mesh = self._run_tsr_pipeline(runner, image)
            if mesh is not None:
                return mesh
        call_attempts: list[Union[dict[str, Any], tuple[Any, ...]]] = []
        if image is not None:
            call_attempts.extend(
                (
                    {"image": image},
                    {"images": image},
                    (image,),
                )
            )
        if image_path:
            call_attempts.extend(
                (
                    {"image_path": image_path},
                    (image_path,),
                )
            )
        call_attempts.append(())

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

    def _run_transformers_triposr(
        self,
        model_name: str,
        image: "Image.Image",
    ) -> Optional[Any]:
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
