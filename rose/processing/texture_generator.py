from typing import Optional, Dict, Any

try:
    import torch
    from PIL import Image, ImageFilter
    import numpy as np
except ImportError as e:
    raise

try:
    from diffusers import (
        StableDiffusionXLControlNetPipeline,
        ControlNetModel,
    )
except ImportError:
    StableDiffusionXLControlNetPipeline = None
    ControlNetModel = None

from .depth_estimator import DepthEstimator


class TextureGenerator:
    """
    Generates texture maps for 3D assets using SDXL + ControlNet and infers
    basic PBR maps (normal, roughness) from the generated albedo.

    - Albedo generation: Stable Diffusion XL conditioned by a control image
      (e.g., UV layout lines, AO, or edges) to encourage structure alignment.
    - Normal map: estimated from monocular depth (DPT) and converted to normals.
    - Roughness map: simple heuristic from local contrast/variance of albedo.

    Notes:
    - Requires diffusers and Hugging Face model access for SDXL + ControlNet.
    - Works on CUDA/ROCm if torch.cuda.is_available(); otherwise CPU.
    """

    def __init__(
        self,
        base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet_id: str = "diffusers/controlnet-canny-sdxl-1.0",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Initialize the texture generator pipeline.

        Args:
            base_model_id: HF model id for SDXL base.
            controlnet_id: HF model id for SDXL ControlNet (lineart/canny/depth, etc.).
            device: "cuda" or "cpu". Default chooses CUDA if available.
            dtype: torch dtype for diffusion. Defaults to float16 on CUDA else float32.
        """
        if StableDiffusionXLControlNetPipeline is None or ControlNetModel is None:
            raise ImportError(
                "diffusers is not installed. Please `pip install diffusers transformers accelerate safetensors`"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if dtype is None:
            dtype = torch.float16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32
        self.dtype = dtype

        # Load ControlNet and SDXL base
        self.controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=self.dtype)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,
        )

        # Memory optimizations
        try:
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
        except Exception:
            pass

        # Move to device or offload
        if self.device == "cuda" and torch.cuda.is_available():
            self.pipe.to(self.device)
        else:
            # CPU/offload path
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception:
                pass

        # Depth estimator for normal inference
        self.depth_estimator = DepthEstimator()

    @staticmethod
    def _ensure_rgb(img: Image.Image) -> Image.Image:
        return img.convert("RGB") if img.mode != "RGB" else img

    @staticmethod
    def _pil_to_np(img: Image.Image) -> np.ndarray:
        return np.array(img)

    @staticmethod
    def _np_to_pil(arr: np.ndarray) -> Image.Image:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def generate_albedo(
        self,
        uv_condition_image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = 1,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        conditioning_scale: float = 1.0,
        output_size: Optional[tuple] = None,
    ) -> Image.Image:
        """
        Generate an albedo (base color) texture using SDXL + ControlNet.

        Args:
            uv_condition_image: Control image (e.g., UV layout overlay, AO or edge map) in UV space.
            prompt: Text prompt describing the desired material/texture.
            negative_prompt: Optional negative prompt.
            seed: RNG seed for reproducibility.
            num_inference_steps: Diffusion steps.
            guidance_scale: Classifier-free guidance scale.
            conditioning_scale: ControlNet conditioning strength.
            output_size: Optional (width, height) to resize the output; defaults to control image size.

        Returns:
            PIL.Image: Generated albedo texture.
        """
        if uv_condition_image is None:
            raise ValueError("uv_condition_image must be provided for ControlNet conditioning")

        control_img = self._ensure_rgb(uv_condition_image)
        if output_size is not None:
            control_img = control_img.resize(output_size, Image.LANCZOS)

        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))
            except Exception:
                generator = torch.Generator().manual_seed(int(seed))

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_img,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=conditioning_scale,
            generator=generator,
        )

        albedo = result.images[0]
        return self._ensure_rgb(albedo)

    def infer_normal_from_albedo(self, albedo: Image.Image) -> Image.Image:
        """
        Infer a tangent-space normal map from the albedo by first predicting monocular
        depth (DPT) and converting depth gradients to normals. Assumes orthographic
        projection approximation for UV-space textures.
        """
        # Estimate depth
        pil_img = self._ensure_rgb(albedo)
        depth = self.depth_estimator.estimate_depth(pil_img)  # HxW float

        # Normalize depth to [0, 1]
        d = depth.astype(np.float32)
        d = d - d.min()
        if d.max() > 1e-6:
            d = d / d.max()

        # Compute gradients
        # Sobel filters via numpy kernels
        kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

        def conv2(x: np.ndarray, k: np.ndarray) -> np.ndarray:
            from scipy.signal import convolve2d  # local import to keep dependency optional in runtime
            return convolve2d(x, k, mode="same", boundary="symm")

        try:
            dx = conv2(d, kx)
            dy = conv2(d, ky)
        except Exception:
            # Fallback with simple finite differences if scipy is unavailable
            dx = np.zeros_like(d)
            dy = np.zeros_like(d)
            dx[:, 1:-1] = (d[:, 2:] - d[:, :-2]) * 0.5
            dy[1:-1, :] = (d[2:, :] - d[:-2, :]) * 0.5

        # Build normals: n = normalize([-dx, -dy, 1])
        nx = -dx
        ny = -dy
        nz = np.ones_like(d)
        n = np.stack([nx, ny, nz], axis=-1)
        n_norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
        n = n / n_norm

        # Convert to 8-bit tangent space RGB
        n_rgb = ((n * 0.5 + 0.5) * 255.0).astype(np.uint8)
        return Image.fromarray(n_rgb, mode="RGB")

    def infer_roughness_from_albedo(self, albedo: Image.Image, strength: float = 1.0) -> Image.Image:
        """
        Infer a roughness map heuristically from local contrast of the albedo.
        Higher local detail -> lower roughness; smoother regions -> higher roughness.
        Output is a single-channel 8-bit image where white = rough.
        """
        img = self._ensure_rgb(albedo)
        gray = img.convert("L")

        # Local variance proxy: high-pass via unsharp mask
        # roughness ~ 1 - normalized(high_frequency)
        low = gray.filter(ImageFilter.GaussianBlur(radius=3))
        high = np.clip(np.array(gray, dtype=np.float32) - np.array(low, dtype=np.float32), -255, 255)

        hf = np.abs(high)
        # Normalize
        hf -= hf.min()
        if hf.max() > 1e-6:
            hf /= hf.max()
        # Invert for roughness and apply strength
        rough = np.clip(1.0 - strength * hf, 0.0, 1.0)
        rough_img = (rough * 255.0).astype(np.uint8)
        return Image.fromarray(rough_img, mode="L")

    def generate_textures(
        self,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = 1,
        steps: int = 30,
        guidance: float = 5.0,
        conditioning_scale: float = 1.0,
        output_size: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """
        High-level convenience method: generate albedo then infer normal and roughness.

        Returns a dict with PIL Images under keys: 'albedo', 'normal', 'roughness'.
        """
        albedo = self.generate_albedo(
            uv_condition_image=control_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=steps,
            guidance_scale=guidance,
            conditioning_scale=conditioning_scale,
            output_size=output_size,
        )
        normal = self.infer_normal_from_albedo(albedo)
        roughness = self.infer_roughness_from_albedo(albedo)
        return {"albedo": albedo, "normal": normal, "roughness": roughness}
