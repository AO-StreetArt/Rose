import torch
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class ImageSegmenter:
    """
    Performs zero-shot image segmentation using the pre-trained CLIPSeg model (CIDAS/clipseg-rd64-refined).
    Supports text prompts for segmentation.
    """
    def __init__(self, device: str = None):
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)

    def segment(self, image: Image.Image, prompts: list):
        """
        Segments the image based on the provided text prompts.
        Args:
            image (PIL.Image.Image): Input image.
            prompts (list of str): List of text prompts to segment.
        Returns:
            np.ndarray: Segmentation masks, shape (num_prompts, H, W), values in [0, 1].
        """
        # Prepare inputs
        inputs = self.processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (num_prompts, 352, 352)
            # Resize masks to original image size
            masks = torch.nn.functional.interpolate(
                logits.unsqueeze(1),
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False
            ).squeeze(1)
            masks = torch.sigmoid(masks)
        return masks.cpu().numpy() 