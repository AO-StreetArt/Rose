#!/usr/bin/env python3
"""CLI utility to convert a single photo into a segmented 3D scene.

Pipeline:
1. Detect objects in the photo.
2. Segment each detected object to obtain tight masks.
3. Estimate global depth for reference assets.
4. For every object: crop & mask, run image-to-3D conversion, generate textures,
   and emit simple PBR material metadata.

Outputs are organised per object inside the chosen output directory.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# flake8: noqa: E402
import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from rose.processing.depth_estimator import DepthEstimator
from rose.processing.image_segmenter import ImageSegmenter
from rose.processing.image_to_3d import ImageTo3DConverter
from rose.processing.object_detector import ObjectDetector
from rose.postprocessing.image_creator import ImageCreator
from rose.preprocessing.image_utils import ImagePreprocessor

try:
    from rose.processing.texture_generator import TextureGenerator
except ImportError:  # pragma: no cover - heavy dependency optional
    TextureGenerator = None  # type: ignore


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str
    class_id: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        bbox = data.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            bbox = [0, 0, 0, 0]
        processed_bbox = tuple(int(round(v)) for v in bbox)  # type: ignore[arg-type]
        return cls(
            bbox=processed_bbox,
            confidence=float(data.get("confidence", 0.0)),
            class_name=str(data.get("class_name") or f"class_{data.get('class_id', 0)}"),
            class_id=int(data.get("class_id", 0)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3D assets from a single photograph.")
    parser.add_argument("image_path", type=Path, help="Path to the input image (jpg/png/etc.)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write results. Defaults to <image_stem>_scene next to the input image.",
    )
    parser.add_argument(
        "--object-model",
        default="faster_rcnn",
        choices=["faster_rcnn", "ssd"],
        help="Object detection backbone to use.",
    )
    parser.add_argument(
        "--object-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence (0-1).",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=8,
        help="Maximum number of detections to process (highest confidence first).",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold in [0,1] for converting segmentation logits to a binary mask.",
    )
    parser.add_argument(
        "--depth-backend",
        choices=["dpt", "zoedepth"],
        default="dpt",
        help="Depth estimation model to use for the reference depth map.",
    )
    parser.add_argument(
        "--depth-colormap",
        default="viridis",
        help="Matplotlib colormap name when visualising depth.",
    )
    parser.add_argument(
        "--image-to-3d-model",
        default="stabilityai/TripoSR",
        help="Image-to-3D model identifier supported by ImageTo3DConverter (defaults to TripoSR).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device string for heavy models (e.g. cuda or cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed forwarded to models that support it.",
    )
    parser.add_argument(
        "--skip-textures",
        action="store_true",
        help="Skip texture and PBR generation to speed up runs.",
    )
    parser.add_argument(
        "--texture-prompt",
        default=None,
        help="Optional diffusion prompt guiding texture generation. Defaults to the detected label.",
    )
    parser.add_argument(
        "--texture-negative-prompt",
        default=None,
        help="Optional negative prompt for texture generation.",
    )
    parser.add_argument(
        "--background-color",
        default="0,0,0",
        help="RGB background (comma separated) used where masks remove pixels (default: 0,0,0).",
    )
    return parser.parse_args()


def resolve_device(preferred: str) -> str:
    device = preferred.lower()
    if device != "cuda":
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return preferred
        print("CUDA requested but unavailable; falling back to cpu.")
    except Exception as exc:
        print(f"Unable to query CUDA availability ({exc}); falling back to cpu.")
    return "cpu"


def sanitise_name(name: str) -> str:
    clean = "".join(c.lower() if c.isalnum() else "_" for c in name)
    clean = clean.strip("_")
    return clean or "object"


def clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def ensure_output_dir(image_path: Path, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    default_dir = image_path.with_name(f"{image_path.stem}_scene")
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir


def load_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    image = Image.open(image_path)
    return ImagePreprocessor.ensure_rgb_pil_image(image)


def detect_objects(image_np: np.ndarray, args: argparse.Namespace) -> List[Detection]:
    # Start with a permissive threshold so we can apply custom fallbacks without reloading models
    initial_threshold = min(args.object_confidence, 0.05)
    detector = ObjectDetector(
        model_type=args.object_model,
        confidence_threshold=initial_threshold,
    )
    detections_raw = detector.detect_objects(image_np)

    def filter_detections(threshold: float) -> List[Detection]:
        filtered = [Detection.from_dict(d) for d in detections_raw if float(d.get("confidence", 0.0)) >= threshold]
        filtered.sort(key=lambda det: det.confidence, reverse=True)
        if args.max_objects and args.max_objects > 0:
            return filtered[: args.max_objects]
        return filtered

    detections = filter_detections(args.object_confidence)

    if not detections:
        fallback_thresholds = [0.3, 0.2, 0.1, initial_threshold]
        for threshold in fallback_thresholds:
            if threshold >= args.object_confidence:
                continue
            fallback = filter_detections(threshold)
            if fallback:
                print(
                    "No detections above "
                    f"{args.object_confidence:.2f}; using threshold {threshold:.2f} instead."
                )
                detections = fallback
                break

    return detections


def compute_depth_map(image: Image.Image, depth_backend: str) -> np.ndarray:
    depth_estimator = DepthEstimator()
    if depth_backend == "zoedepth":
        depth = depth_estimator.estimate_depth_zoedepth(image)
        if depth is not None:
            return depth
        print("Warning: ZoeDepth failed; falling back to DPT.")
    return depth_estimator.estimate_depth(image)


def segment_objects(image: Image.Image, detections: List[Detection]) -> Dict[str, np.ndarray]:
    prompts = []
    seen = set()
    for det in detections:
        name = det.class_name or f"class_{det.class_id}"
        if name not in seen:
            prompts.append(name)
            seen.add(name)
    if not prompts:
        return {}
    segmenter = ImageSegmenter()
    masks = segmenter.segment(image, prompts)
    return {prompt: mask for prompt, mask in zip(prompts, masks)}


def parse_background(color_str: str) -> Tuple[int, int, int]:
    try:
        parts = [int(p.strip()) for p in color_str.split(",")]
        if len(parts) != 3:
            raise ValueError
        return tuple(max(0, min(255, v)) for v in parts)  # type: ignore[return-value]
    except Exception as exc:
        raise ValueError(f"Invalid --background-color value '{color_str}'. Expected R,G,B") from exc


def create_object_images(
    base_image: Image.Image,
    mask: Optional[np.ndarray],
    bbox: Tuple[int, int, int, int],
    mask_threshold: float,
    background_rgb: Tuple[int, int, int],
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    width, height = base_image.size
    x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
    crop_rgb = base_image.crop((x1, y1, x2, y2)).convert("RGB")
    crop_np = np.array(crop_rgb)

    if mask is not None:
        mask_crop = mask[y1:y2, x1:x2]
        binary = mask_crop >= mask_threshold
        if not binary.any():
            binary = np.ones_like(binary, dtype=bool)
    else:
        binary = np.ones((crop_np.shape[0], crop_np.shape[1]), dtype=bool)

    mask_uint8 = (binary.astype(np.uint8) * 255)
    crop_np[~binary] = background_rgb
    masked_rgb = Image.fromarray(crop_np, mode="RGB")
    mask_image = Image.fromarray(mask_uint8, mode="L")
    masked_rgba = masked_rgb.convert("RGBA")
    masked_rgba.putalpha(mask_image)
    return masked_rgb, masked_rgba, mask_image


def export_mesh(mesh: Any, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Attempt common interfaces first
    try:
        if hasattr(mesh, "export"):
            export_method = getattr(mesh, "export")
            try:
                export_method(str(output_path), file_type=output_path.suffix.lstrip("."))
            except TypeError:
                export_method(str(output_path))
            return True
        if hasattr(mesh, "write"):
            mesh.write(str(output_path))
            return True
    except Exception as exc:
        print(f"Warning: mesh export failed: {exc}")

    try:
        output_path.write_text(str(mesh), encoding="utf-8")
        print(f"Warning: wrote textual mesh representation to {output_path}")
        return False
    except Exception as exc:
        print(f"Error: could not persist mesh data at {output_path}: {exc}")
        return False


def prepare_texture_generator(device: str) -> Optional[TextureGenerator]:
    if TextureGenerator is None:
        print("Warning: TextureGenerator unavailable (missing diffusers and related deps). Skipping textures.")
        return None
    try:
        return TextureGenerator(device=device)
    except Exception as exc:
        print(f"Warning: failed to initialise TextureGenerator: {exc}")
        return None


def run_pipeline(args: argparse.Namespace) -> None:
    image_path: Path = args.image_path
    output_dir = ensure_output_dir(image_path, args.output_dir)
    background_rgb = parse_background(args.background_color)

    print(f"Loading image: {image_path}")
    base_image = load_image(image_path)
    base_np = np.array(base_image)

    print("Detecting objects...")
    detections = detect_objects(base_np, args)
    if not detections:
        width, height = base_image.size
        print("No objects detected; treating entire frame as a single object.")
        detections = [Detection(bbox=(0, 0, width, height), confidence=1.0, class_name="scene", class_id=-1)]

    print(f"Detected {len(detections)} objects.")

    print("Estimating depth map...")
    depth_map = compute_depth_map(base_image, args.depth_backend)
    depth_png = output_dir / "depth_map.png"
    depth_raw = output_dir / "depth_map.npy"
    ImageCreator.save_depth_map_as_image(depth_map, str(depth_png), colormap=args.depth_colormap)
    np.save(depth_raw, depth_map)

    print("Running segmentation for detected classes...")
    class_masks = segment_objects(base_image, detections)

    resolved_device = resolve_device(args.device)

    print("Initialising image-to-3D model...")
    converter = ImageTo3DConverter(model_name=args.image_to_3d_model, device=resolved_device)
    use_triposr = "triposr" in args.image_to_3d_model.lower()

    texture_generator = None
    if not args.skip_textures:
        texture_generator = prepare_texture_generator(device=resolved_device)

    summary: List[Dict[str, Any]] = []

    for index, det in enumerate(detections, start=1):
        label = det.class_name or f"class_{det.class_id}"
        safe_label = sanitise_name(label)
        object_dir = output_dir / f"{index:02d}_{safe_label}"
        object_dir.mkdir(parents=True, exist_ok=True)

        mask = class_masks.get(label)
        masked_rgb, masked_rgba, mask_image = create_object_images(
            base_image,
            mask,
            det.bbox,
            args.mask_threshold,
            background_rgb,
        )

        object_source = object_dir / "source.png"
        object_mask = object_dir / "mask.png"
        object_rgba = object_dir / "source_rgba.png"
        masked_rgb.save(object_source)
        mask_image.save(object_mask)
        masked_rgba.save(object_rgba)

        print(f"[{index}/{len(detections)}] Converting {label} to 3D...")
        mesh_result: Optional[Any] = None
        try:
            if use_triposr and hasattr(converter, "image_to_3d_triposr"):
                mesh_result = converter.image_to_3d_triposr(
                    str(object_source),
                    model_name=args.image_to_3d_model,
                )
            else:
                mesh_result = converter.image_to_3d(masked_rgb, seed=args.seed)
        except Exception as exc:
            print(f"Warning: image-to-3D conversion failed for {label}: {exc}")

        mesh_object = None
        additional_outputs: Dict[str, Any] = {}
        if isinstance(mesh_result, dict):
            additional_outputs = {k: v for k, v in mesh_result.items() if k != "mesh"}
            mesh_object = mesh_result.get("mesh") or mesh_result.get("meshes")
        else:
            mesh_object = mesh_result

        mesh_written = False
        mesh_path = object_dir / "mesh.gltf"
        if mesh_object is not None:
            mesh_written = export_mesh(mesh_object, mesh_path)
        else:
            mesh_path.write_text("// Mesh generation failed", encoding="utf-8")
            print(f"Warning: no mesh produced for {label}.")

        texture_files: Dict[str, str] = {}
        material_settings: Dict[str, Any] = {
            "name": label,
            "albedo": None,
            "normal": None,
            "roughness": None,
            "metallic": 0.0,
            "roughness_value": 0.5,
            "occlusion": None,
        }

        if texture_generator is not None:
            try:
                prompt = args.texture_prompt or f"texture of {label}"
                textures = texture_generator.generate_textures(
                    control_image=masked_rgb,
                    prompt=prompt,
                    negative_prompt=args.texture_negative_prompt,
                    seed=args.seed,
                )
                for tex_name, image_obj in textures.items():
                    tex_path = object_dir / f"{tex_name}.png"
                    image_obj.save(tex_path)
                    texture_files[tex_name] = tex_path.name
                    material_settings[tex_name] = tex_path.name
            except Exception as exc:
                print(f"Warning: texture generation failed for {label}: {exc}")
        else:
            prompt_stub = args.texture_prompt or f"texture of {label}"
            material_settings["notes"] = (
                "Texture generation skipped. Provide textures manually. Suggested prompt: " + prompt_stub
            )

        metadata = {
            "label": label,
            "confidence": det.confidence,
            "bbox": det.bbox,
            "mesh_file": mesh_path.name if mesh_written else None,
            "textures": texture_files,
            "material_settings": material_settings,
            "additional_outputs": list(additional_outputs.keys()),
        }
        with open(object_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        summary.append(metadata)

    summary_path = output_dir / "scene_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "image": str(image_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "depth_files": {
                    "colourised": depth_png.name,
                    "raw": depth_raw.name,
                },
                "objects": summary,
            },
            handle,
            indent=2,
        )

    print(f"Scene assets written to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    run_pipeline(parse_args())
