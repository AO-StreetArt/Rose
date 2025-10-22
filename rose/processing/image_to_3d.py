import importlib
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

try:
    import torch
except ImportError:
    torch = None

try:
    from diffusers import DiffusionPipeline
except ImportError:
    DiffusionPipeline = None

try:
    from threedtopia import load_pipeline as load_3dtopia_pipeline  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    try:
        import inference as _threedtopia  # type: ignore

        load_3dtopia_pipeline = getattr(_threedtopia, "load_pipeline", None)
    except ImportError:
        load_3dtopia_pipeline = None  # type: ignore


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


def _import_3dtopia_pipeline_class() -> Optional[Any]:
    """Attempt to import the 3DTopia-XL image-to-3D pipeline from known modules."""
    candidates = (
        ("threedtopia.pipelines.image_to_3d", "ThreeDTopiaXLPipeline"),
        ("threedtopia.pipeline.image_to_3d", "ThreeDTopiaXLPipeline"),
        ("threedtopia.pipeline", "ThreeDTopiaXLPipeline"),
        ("threedtopia", "ThreeDTopiaXLPipeline"),
        ("threedtopia", "ThreeDTopiaXL"),
        ("threedtopia.runners", "ImageTo3DPipeline"),
        ("threedtopia.runners", "ThreeDTopiaXLPipeline"),
    )
    for module_path, attr in candidates:
        try:
            module = importlib.import_module(module_path)
            runner = getattr(module, attr)
            return runner
        except (ImportError, AttributeError):
            continue
    return None


ThreeDTopiaPipelineClass = _import_3dtopia_pipeline_class()

SUPPORTED_3DTOPIA_MODELS = {"3DTopia/3DTopia-XL", "3DTopia-XL"}


class ImageTo3DConverter:
    """
    Converts images to 3D data using a selection of state-of-the-art image-to-3D pipelines.
    Supports TRELLIS, Facebook VGGT, Tencent Hunyuan3D, Stability AI TripoSR, and 3DTopia-XL backends
    to generate meshes and related assets from a single image.
    """
    def __init__(self, model_name: str = "microsoft/TRELLIS-image-large", device: str = "cuda"):
        """
        Initializes the requested image-to-3D pipeline backend.
        Args:
            model_name (str): Identifier for the supported model to use (e.g. TRELLIS, VGGT, Hunyuan3D, TripoSR, 3DTopia-XL).
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
        self._3dtopia_backend: Optional[str] = None
        self._3dtopia_model_name: Optional[str] = None
        self._3dtopia_last_error: Optional[Exception] = None
        self._3dtopia_pipeline: Optional[Any] = None
        self._3dtopia_cached_identifier: Optional[str] = None
        resolved_model_name = self._resolve_model_reference(self.model_name)

        if self.model_name == 'microsoft/TRELLIS-image-large':
            if TrellisImageTo3DPipeline is None:
                raise ImportError("TRELLIS image-to-3D pipeline is not installed.")
            os.environ["SPCONV_ALGO"] = "native"  # Recommended for single runs
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(resolved_model_name)
        elif self.model_name == 'facebook/VGGT-1B':
            if VGGT is not None and load_and_preprocess_images is not None:
                resolved = self._resolve_model_reference("facebook/VGGT-1B")
                self.pipeline = VGGT.from_pretrained(resolved).to(device)
        elif self.model_name == 'tencent/Hunyuan3D':
            if Hunyuan3DDiTFlowMatchingPipeline is None:
                raise ImportError("Hunyuan3D pipeline is not installed.")
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(resolved_model_name)
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
        elif self.model_name in {"3DTopia/3DTopia-XL", "3DTopia-XL"}:
            self._3dtopia_pipeline = self._load_3dtopia_pipeline(resolved_model_name)
            self.pipeline = self._3dtopia_pipeline
            self._3dtopia_model_name = self.model_name
            self._3dtopia_cached_identifier = resolved_model_name
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        if (
            self.pipeline is not None
            and isinstance(self.device, str)
            and self.device.startswith("cuda")
        ):
            if hasattr(self.pipeline, "cuda"):
                self.pipeline.cuda()
            elif hasattr(self.pipeline, "to"):
                self.pipeline.to(self.device)

    def image_to_3d(self, image: "Image.Image", seed: int = 1, **kwargs) -> Optional[Any]:
        """
        Converts a PIL image to 3D data using the TRELLIS pipeline backend.
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
    def is_3dtopia_available() -> bool:
        """Returns True if the dependencies required for 3DTopia-XL inference are available."""
        return load_3dtopia_pipeline is not None and torch is not None

    def image_to_3d_3dtopia(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        seed: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        num_inference_steps: Optional[int] = 75,
        guidance_scale: Optional[float] = 7.5,
        return_visualizations: bool = False,
        return_video: bool = False,
        image: Optional[Image.Image] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Generates a textured mesh using the 3DTopia-XL pipeline."""
        if not self.is_3dtopia_available():
            raise ImportError("3DTopia-XL backend is not available. Install the '3dtopia-xl' package.")

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("3DTopia-XL prompt must be a non-empty string")

        backend_identifier = model_name or self._3dtopia_model_name or self.model_name or "3DTopia/3DTopia-XL"
        if backend_identifier not in SUPPORTED_3DTOPIA_MODELS:
            raise ValueError(f"Unsupported 3DTopia-XL model identifier: {backend_identifier}")

        resolved_model = self._resolve_model_reference(backend_identifier)
        pipeline = self._ensure_3dtopia_pipeline(resolved_model, display_name=backend_identifier)
        if self.model_name in SUPPORTED_3DTOPIA_MODELS:
            self.pipeline = pipeline

        call_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "return_visualizations": return_visualizations,
            "return_video": return_video,
        }
        if image is not None:
            if Image is None or not isinstance(image, Image.Image):
                raise TypeError("image must be a PIL.Image when provided")
            call_kwargs["image"] = image
        call_kwargs.update(kwargs)

        try:
            outputs = pipeline(**call_kwargs)
        except Exception as exc:  # pragma: no cover - depends on vendor pipeline
            self._3dtopia_last_error = exc
            raise RuntimeError(f"3DTopia-XL pipeline execution failed: {exc}") from exc

        self._3dtopia_last_error = None

        if outputs is None:
            return None

        result = dict(outputs)
        if output_dir is not None:
            result.setdefault("asset_dir", str(output_dir))
        return result

    def _ensure_3dtopia_pipeline(self, resolved_identifier: str, display_name: Optional[str] = None) -> Any:
        if self._3dtopia_pipeline is not None and self._3dtopia_cached_identifier == resolved_identifier:
            return self._3dtopia_pipeline
        pipeline = self._load_3dtopia_pipeline(resolved_identifier)
        self._3dtopia_pipeline = pipeline
        self._3dtopia_cached_identifier = resolved_identifier
        if display_name is not None:
            self._3dtopia_model_name = display_name
        elif self._3dtopia_model_name is None:
            self._3dtopia_model_name = resolved_identifier
        return pipeline

    def _format_3dtopia_outputs(
        self,
        outputs: Any,
        output_dir: Optional[Union[str, Path]],
    ) -> Optional[Dict[str, Any]]:
        if outputs is None:
            return None

        result: Dict[str, Any] = {}
        if isinstance(outputs, dict):
            result.update(outputs)
        else:
            attr_map = {
                "mesh": ("mesh", "geometry", "tri_mesh", "mesh_output"),
                "meshes": ("meshes", "mesh_list", "geometries"),
                "texture_images": ("texture_images", "textures", "texture", "texture_map", "uv_textures"),
                "materials": ("materials", "material", "mtl"),
                "mesh_path": ("mesh_path", "mesh_file", "mesh_filepath", "geometry_path"),
                "texture_paths": ("texture_paths", "texture_files", "texture_filepaths"),
                "material_paths": ("material_paths", "mtl_path", "materials_path"),
            }
            for target, candidates in attr_map.items():
                for candidate in candidates:
                    if hasattr(outputs, candidate):
                        value = getattr(outputs, candidate)
                        if value is None:
                            continue
                        if target in {"texture_paths", "material_paths"}:
                            result[target] = [str(v) for v in self._as_list(value)]
                        elif target == "meshes" and isinstance(value, (list, tuple)):
                            result[target] = list(value)
                        else:
                            result[target] = value
                        break

        if "mesh" not in result and isinstance(result.get("meshes"), (list, tuple)) and result["meshes"]:
            result["mesh"] = result["meshes"][0]

        if output_dir is not None:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            saved = self._maybe_save_3dtopia_assets(outputs, destination)
            for key, value in saved.items():
                if isinstance(value, Path):
                    value = str(value)
                elif isinstance(value, (list, tuple)):
                    value = [str(v) for v in value]
                result[key] = value
            result.setdefault("asset_dir", str(destination))

        for key in ("mesh_path", "asset_dir"):
            if isinstance(result.get(key), Path):
                result[key] = str(result[key])
        for key in ("texture_paths", "material_paths"):
            if key in result:
                result[key] = [str(v) for v in self._as_list(result[key])]

        result.setdefault("raw_output", outputs)
        return result

    def _maybe_save_3dtopia_assets(self, outputs: Any, destination: Path) -> Dict[str, Any]:
        if outputs is None:
            return {}

        for method_name in (
            "save",
            "save_assets",
            "export",
            "write",
            "write_assets",
            "save_mesh",
            "save_obj",
        ):
            method = getattr(outputs, method_name, None)
            if callable(method):
                try:
                    result = method(destination)
                except TypeError:
                    try:
                        result = method(str(destination))
                    except Exception as exc:
                        print(f"Warning: 3DTopia-XL {method_name} export failed: {exc}")
                        return {}
                except Exception as exc:
                    print(f"Warning: 3DTopia-XL {method_name} export failed: {exc}")
                    return {}

                if result is None:
                    return {"asset_dir": str(destination)}
                if isinstance(result, dict):
                    return result
                if isinstance(result, (list, tuple)):
                    return {"exported_files": [str(item) for item in result]}
                return {"export_result": result}
        return {}

    def _ensure_rgb_image(self, image: Union[str, "Image.Image"]) -> "Image.Image":
        if Image is None:
            raise ImportError("Pillow is required for 3DTopia-XL image inputs.")
        if isinstance(image, str):
            with Image.open(image) as img:
                return img.convert("RGB")
        if hasattr(image, "convert") and callable(image.convert):
            if getattr(image, "mode", "") != "RGB":
                return image.convert("RGB")
            return image
        raise TypeError("image must be a file path or a PIL.Image.Image instance")

    def _model_root(self) -> Optional[Path]:
        root = os.getenv("IMAGE_TO_3D_MODEL_ROOT") or os.getenv("SM_MODEL_DIR")
        if not root:
            return None
        candidate = Path(root)
        if candidate.exists():
            _logger.info("Model root resolved to %s", candidate)
            return candidate
        return None

    def _resolve_model_reference(self, identifier: str) -> str:
        root = self._model_root()
        if root is None:
            return identifier
        normalized = identifier.strip()
        if not normalized:
            return identifier
        normalized = normalized.replace("\\", "/").strip("/")
        candidates = []
        path_candidate = Path(normalized)
        candidates.append(root / path_candidate)
        name = path_candidate.name
        if name:
            candidates.append(root / name)
        underscored = normalized.replace("/", "_")
        if underscored != name:
            candidates.append(root / underscored)
        seen = set()
        for candidate in candidates:
            key = candidate.as_posix()
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                _logger.info("Resolved %s to %s", identifier, candidate)
                return str(candidate)
        if root.exists():
            root_sentinels = (
                "model_index.json",
                "config.json",
                "pipeline_config.json",
            )
            for sentinel in root_sentinels:
                if (root / sentinel).exists():
                    _logger.info("Resolved %s to model root %s via sentinel %s", identifier, root, sentinel)
                    return str(root)
            # Even if sentinel files are absent, stay on the local model root so SageMaker
            # deployments without internet never attempt Hugging Face downloads.
            _logger.info("Resolved %s to model root %s via default fallback", identifier, root)
            return str(root)
        return identifier

    def _as_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _is_cuda_device(self) -> bool:
        return isinstance(self.device, str) and self.device.startswith("cuda")

    def _load_3dtopia_pipeline(self, model_name: str) -> Any:
        if load_3dtopia_pipeline is None:
            raise ImportError(
                "3DTopia-XL loader is not available. Ensure the forked `3dtopia-xl` package is installed."
            )

        model_path = self._resolve_model_reference(model_name)
        path_obj = Path(model_path)
        if not path_obj.exists():
            raise RuntimeError(f"3DTopia checkpoint directory does not exist: {path_obj}")

        config_path = os.getenv("THREEDTOPIA_CONFIG_PATH")
        output_dir = os.getenv("THREEDTOPIA_OUTPUT_DIR")

        try:
            pipeline = load_3dtopia_pipeline(
                checkpoint_dir=path_obj,
                device=self.device,
                config_path=config_path,
                output_dir=output_dir,
            )
        except Exception as exc:  # pragma: no cover - relies on vendor package
            raise RuntimeError(
                f"Unable to load 3DTopia-XL pipeline from {path_obj}: {exc}"
            ) from exc

        self._3dtopia_backend = "native"
        return pipeline

    def _should_use_fp16(self) -> bool:
        if os.getenv("THREEDTOPIA_FORCE_FP32") == "1":
            return False
        if os.getenv("THREEDTOPIA_FORCE_FP16") == "1":
            return torch is not None
        if torch is None or not self._is_cuda_device():
            return False
        cuda_module = getattr(torch, "cuda", None)
        if cuda_module is not None:
            is_available = getattr(cuda_module, "is_available", None)
            if callable(is_available):
                return bool(is_available())
        return True

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

        resolved_model = self._resolve_model_reference(model_name)

        if self.triposr_backend == "official":
            runner = self._ensure_official_triposr_runner(resolved_model)
            if runner is None:
                return None
            mesh = self._run_official_triposr(runner, image, image_path)
            return mesh
        else:
            return self._run_transformers_triposr(resolved_model, image)

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

_logger = logging.getLogger(__name__)
