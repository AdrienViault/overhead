import torch
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# Import SAM2 model builder and automatic mask generator
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# =============================================================================
# Set paths
# =============================================================================
sam2_checkpoint = "//home/adrien/Documents/Dev/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "//home/adrien/Documents/Dev/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" 
image_path = "/home/adrien/Documents/Dev/overhead/data/images/test_images/reprojected/perspective_0_GSAC0346.JPG"  

# Output folder for segmented images
output_folder = "/home/adrien/Documents/Dev/overhead/data/images/test_images/segmentation/"
os.makedirs(output_folder, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Build the SAM2 model (postprocessing disabled for clarity)
# =============================================================================
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

# =============================================================================
# Helper function to overlay segmentation masks on an image.
# =============================================================================
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    # Create an RGBA image for overlaying masks
    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0  # start fully transparent
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
    ax.imshow(img)

# =============================================================================
# Function to load and preprocess the image.
# =============================================================================
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Load the image
image = preprocess_image(image_path)

# =============================================================================
# Base parameters for the mask generator
# =============================================================================
base_params = {
    'points_per_side': 64,
    'points_per_batch': 128,
    'pred_iou_thresh': 0.7,
    'stability_score_thresh': 0.92,
    'stability_score_offset': 0.7,
    'crop_n_layers': 1,
    'box_nms_thresh': 0.7,
    'crop_n_points_downscale_factor': 2,
    'min_mask_region_area': 5.0,
    'use_m2m': True,
}

# =============================================================================
# Parameter variations to test
# =============================================================================
param_variations = {
    'points_per_side': {
         'values': [16, 32, 64],
         'comment': 'Grid points per image side. Lower values yield coarser sampling; higher values provide denser proposals.'
    },
    'points_per_batch': {
         'values': [64, 128, 256],
         'comment': 'Number of points processed per batch.'
    },
    'pred_iou_thresh': {
         'values': [0.5, 0.7, 0.9],
         'comment': 'Threshold for predicted IoU to filter masks.'
    },
    'stability_score_thresh': {
         'values': [0.8, 0.92, 0.98],
         'comment': 'Minimum stability score required for a mask to be kept.'
    },
    'stability_score_offset': {
         'values': [0.5, 0.7, 0.9],
         'comment': 'Offset applied to the stability score during evaluation.'
    },
    'crop_n_layers': {
         'values': [1, 2, 3],
         'comment': 'Number of cropping layers used to capture multi-scale details.'
    },
    'box_nms_thresh': {
         'values': [0.5, 0.7, 0.9],
         'comment': 'NMS threshold for overlapping boxes. Lower values lead to more aggressive suppression.'
    },
    'crop_n_points_downscale_factor': {
         'values': [1, 2, 4],
         'comment': 'Factor to downscale the number of crop points, trading off detail for speed.'
    },
    'min_mask_region_area': {
         'values': [5.0, 10.0, 20.0],
         'comment': 'Minimum pixel area for a mask region to be considered valid.'
    },
    'use_m2m': {
         'values': [True, False],
         'comment': 'Enable (True) or disable (False) mask-to-mask post-processing refinement.'
    }
}

# =============================================================================
# Precompute masks for each parameter and value variation.
# This avoids generating masks multiple times.
# =============================================================================
masks_results = {}  # {param_name: {value: masks}}
for param_name, param_info in param_variations.items():
    values = param_info['values']
    masks_results[param_name] = {}
    for value in values:
        params = base_params.copy()
        params[param_name] = value
        mask_generator = SAM2AutomaticMaskGenerator(model=sam2, **params)
        masks = mask_generator.generate(image)
        masks_results[param_name][value] = masks

# =============================================================================
# PART 1: Save one figure per parameter showing all its variation values.
# =============================================================================
for param_name, param_info in param_variations.items():
    values = param_info['values']
    comment = param_info['comment']
    print(f"\n--- Saving figure for parameter '{param_name}': {comment} ---")
    
    n_variations = len(values)
    fig, axes = plt.subplots(1, n_variations, figsize=(5 * n_variations, 5))
    if n_variations == 1:
        axes = [axes]
    
    for i, value in enumerate(values):
        plt.sca(axes[i])
        axes[i].imshow(image)
        show_anns(masks_results[param_name][value])
        axes[i].set_title(f"{param_name} = {value}", fontsize=10)
        axes[i].axis("off")
    
    fig.suptitle(f"Effect of '{param_name}': {comment}", fontsize=14)
    plt.tight_layout()
    
    # Save the figure for this parameter group
    output_filename = f"segmentation_{param_name}_variations.png"
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    print(f"Saved: {output_path}")
    plt.show()
    plt.close(fig)