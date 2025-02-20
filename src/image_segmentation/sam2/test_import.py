import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("YAY we have access to gpu")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

sam2_checkpoint = "//home/adrien/Documents/Dev/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "//home/adrien/Documents/Dev/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" 
image_path = "/home/adrien/Documents/Dev/overhead/data/images/test_images/reprojected/perspective_0_GSAC0346.JPG"  


print("runnin model load")
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
print("creating predictor")
predictor = SAM2ImagePredictor(sam2_model)
