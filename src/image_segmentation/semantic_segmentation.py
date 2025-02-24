#!/usr/bin/env python
"""
This script loads a DINOv2-based semantic segmentation model,
applies it to a local image file, and saves the rendered segmentation mask.
It uses Meta’s DINOv2 backbone and segmentation head configuration.
Dependencies include mmcv, mmseg, torch, PIL, cv2, and the dinov2 repository.
"""

import sys
import math
import itertools
import urllib.request
from functools import partial

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

# mmcv and mmseg for model initialization and inference
import mmcv
from mmseg.apis import init_segmentor, inference_segmentor

# Import the DINOv2 segmentation modules (make sure dinov2 is installed or the repo path is added)
import dinov2.eval.segmentation.models
import dinov2.eval.segmentation.utils.colormaps as colormaps

# Define dataset colormaps for rendering segmentation results.
DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}

# -----------------------------------------------------------------------------
# Utility Classes and Functions
# -----------------------------------------------------------------------------
class CenterPadding(torch.nn.Module):
    """
    Pads an input tensor so that each spatial dimension is a multiple of a given number.
    """
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        return pad_left, pad_right

    @torch.inference_mode()
    def forward(self, x):
        # x: (B, C, H, W)
        pad_w = self._get_pad(x.shape[-1])
        pad_h = self._get_pad(x.shape[-2])
        pads = pad_w + pad_h  # Order: (left, right, top, bottom)
        return F.pad(x, pads)

def load_config_from_url(url: str) -> str:
    """
    Downloads the configuration file (Python script) from the provided URL.
    """
    with urllib.request.urlopen(url) as f:
        config_str = f.read().decode()
    return config_str

def create_segmenter(cfg, backbone_model):
    """
    Creates a segmentation model by combining the segmentation head configuration with the DINOv2 backbone.
    It replaces the backbone's forward method to output intermediate features.
    """
    model = init_segmentor(cfg)
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_size)(x[0])
        )
    model.init_weights()
    return model

def render_segmentation(segmentation_logits, dataset):
    """
    Maps segmentation logits to a colored image using the appropriate colormap.
    """
    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    # Offset labels by 1 if needed (depends on labeling scheme)
    segmentation_values = colormap_array[segmentation_logits + 1]
    return Image.fromarray(segmentation_values)

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main():
    # ------------------------------
    # Configuration: file paths and parameters
    # ------------------------------
    # Local image file to segment
    image_path = "path/to/your_local_image.jpg"  # <-- Replace with your local image path
    # Output file where the segmented image will be saved
    output_path = "segmented_output.png"
    # Dataset used for the segmentation head's colormap (choose "voc2012" or "ade20k")
    dataset_name = "voc2012"
    # Segmentation head type: "ms" (multi-scale) or "linear"
    head_type = "ms"
    # Dataset used for segmentation head configuration (affects config and weights)
    head_dataset = "voc2012"
    # DINOv2 backbone size: "small", "base", "large", or "giant"
    backbone_size = "small"

    # ------------------------------
    # Step 1: Load the DINOv2 Backbone Model
    # ------------------------------
    # Map backbone sizes to corresponding architecture names
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[backbone_size]
    backbone_name = f"dinov2_{backbone_arch}"

    print("Loading DINOv2 backbone model...")
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    if torch.cuda.is_available():
        backbone_model.cuda()
    else:
        print("CUDA is not available. Running on CPU.")

    # ------------------------------
    # Step 2: Load Segmentation Head Configuration and Weights
    # ------------------------------
    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{head_dataset}_{head_type}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{head_dataset}_{head_type}_head.pth"

    print("Loading segmentation head configuration from URL...")
    cfg_str = load_config_from_url(head_config_url)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
    if head_type == "ms":
        HEAD_SCALE_COUNT = 3  # Adjust number of scales for multi-scale segmentation
        cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
        print("Using multi-scale head with scales:", cfg.data.test.pipeline[1]["img_ratios"])

    print("Creating segmentation model...")
    segmenter = create_segmenter(cfg, backbone_model=backbone_model)
    print("Loading segmentation head weights from URL...")
    mmcv.runner.load_checkpoint(segmenter, head_checkpoint_url, map_location="cpu")
    if torch.cuda.is_available():
        segmenter.cuda()
    segmenter.eval()

    # ------------------------------
    # Step 3: Load Local Image and Run Segmentation
    # ------------------------------
    print(f"Loading local image from {image_path}...")
    image = Image.open(image_path).convert("RGB")
    # Convert the image from RGB to BGR, as expected by mmseg (OpenCV format)
    image_array = np.array(image)[:, :, ::-1]

    print("Performing segmentation inference...")
    segmentation_logits = inference_segmentor(segmenter, image_array)[0]  # inference_segmentor returns a list; we take the first output

    # ------------------------------
    # Step 4: Render and Save Segmentation Output
    # ------------------------------
    print("Rendering segmentation output...")
    segmented_image = render_segmentation(segmentation_logits, dataset_name)
    segmented_image.save(output_path)
    print(f"Segmented image saved to {output_path}")

if __name__ == "__main__":
    main()
