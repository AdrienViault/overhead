import cv2
from PIL import Image
import requests
import torch
from matplotlib import rcParams
import matplotlib.pyplot as plt

# Set figure size
%matplotlib inline
rcParams['figure.figsize'] = 11 ,8

# Target image
target_image_path = "data/images/test_images/reprojected/perspective_70.0deg.jpg"
target_image = Image.open(target_image_path).convert("RGB")
target_sizes = torch.Tensor([target_image.size[::-1]])

# Source image
source_url = "http://images.cocodataset.org/val2017/000000058111.jpg"
source_image = Image.open(requests.get(source_url, stream=True).raw)

# Display input image and query image
fig, ax = plt.subplots(1,2)
ax[0].imshow(target_image)
ax[1].imshow(source_image)