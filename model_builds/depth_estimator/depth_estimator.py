import os
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor


class DepthEstimator:
    """
    Estimates depth information for elements in images, using monocular or stereo depth estimation models.
    """

    _SUPPORTED_BACKENDS = {"dpt", "zoedepth"}

    def __init__(
        self,
        default_backend: str = "dpt",
        dpt_model_id: str = "Intel/dpt-large",
        zoedepth_model_id: str = "Intel/zoedepth-nyu-kitti",
        *,
        local_files_only: bool = False,
        cache_dir: Optional[str] = None,
        dpt_revision: Optional[str] = None,
        zoedepth_revision: Optional[str] = None,
    ):
        backend = default_backend.strip().lower()
        if backend not in self._SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend '{default_backend}'. Choose from {self._SUPPORTED_BACKENDS}.")

        # Lazily initialised model handles to keep startup lightweight unless requested.
        self.processor: Optional[DPTImageProcessor] = None
        self.model: Optional[DPTForDepthEstimation] = None
        self.zoedepth_processor = None
        self.zoedepth_model = None
        self._default_backend = backend
        self._dpt_model_id = dpt_model_id
        self._zoedepth_model_id = zoedepth_model_id
        self._local_files_only = local_files_only
        self._cache_dir = cache_dir
        self._dpt_revision = dpt_revision
        self._zoedepth_revision = zoedepth_revision

        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)

        # Eagerly load the chosen default backend so the first request does not pay the loading cost.
        if backend == "dpt":
            self._ensure_dpt_loaded()
        else:
            self._ensure_zoedepth_loaded()

    @property
    def default_backend(self) -> str:
        return self._default_backend

    def _hf_load_kwargs(self, *, revision: Optional[str], model_id: str) -> Dict[str, object]:
        kwargs: Dict[str, object] = {}
        is_local_path = os.path.isdir(model_id)

        if self._cache_dir and not is_local_path:
            kwargs["cache_dir"] = self._cache_dir
        if revision and not is_local_path:
            kwargs["revision"] = revision
        if self._local_files_only and not is_local_path:
            kwargs["local_files_only"] = True
        return kwargs

    def _ensure_dpt_loaded(self) -> None:
        if self.processor is None or self.model is None:
            model_id = self._dpt_model_id
            load_kwargs = self._hf_load_kwargs(revision=self._dpt_revision, model_id=model_id)
            self.processor = DPTImageProcessor.from_pretrained(model_id, **load_kwargs)
            self.model = DPTForDepthEstimation.from_pretrained(model_id, **load_kwargs)

    def _ensure_zoedepth_loaded(self) -> None:
        if self.zoedepth_processor is None or self.zoedepth_model is None:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            model_id = self._zoedepth_model_id
            load_kwargs = self._hf_load_kwargs(revision=self._zoedepth_revision, model_id=model_id)
            self.zoedepth_processor = AutoImageProcessor.from_pretrained(model_id, **load_kwargs)
            self.zoedepth_model = AutoModelForDepthEstimation.from_pretrained(model_id, **load_kwargs)

    def estimate_depth(self, img: Image.Image) -> np.ndarray:
        """
        Accepts a PIL Image object and returns a depth map using the Intel/dpt-large model.
        Args:
            img (PIL.Image.Image): Input image.
        Returns:
            np.ndarray: Depth map of the image.
        """
        self._ensure_dpt_loaded()

        inputs = self.processor(images=img, return_tensors="pt")  # type: ignore[operator]
        with torch.no_grad():
            outputs = self.model(**inputs)  # type: ignore[operator]
            predicted_depth = outputs.predicted_depth
            # Resize to original image size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        return prediction

    def estimate_depth_zoedepth(self, img: Image.Image) -> Optional[np.ndarray]:
        """
        Accepts a PIL Image object and returns a depth map using the Intel/zoedepth-nyu-kitti model.
        Args:
            img (PIL.Image.Image): Input image.
        Returns:
            np.ndarray: Depth map of the image, or None if estimation fails.
        """
        try:
            self._ensure_zoedepth_loaded()

            inputs = self.zoedepth_processor(images=img, return_tensors="pt")  # type: ignore[operator]
            with torch.no_grad():
                outputs = self.zoedepth_model(**inputs)  # type: ignore[operator]
                predicted_depth = outputs.predicted_depth
            depth_map = predicted_depth.squeeze().cpu().numpy()
            return depth_map
        except Exception as e:
            print(f"Error during ZoeDepth estimation: {e}")
            return None
