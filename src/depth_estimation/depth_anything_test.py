from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt

# Load image from URL and ensure it's in RGB
image_path = "data/images/test_images/reprojected/perspective_-90deg.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Initialize device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the image processor and model, then move the model to the device
image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf").to(device)

# Prepare image for the model and send tensors to the device
inputs = image_processor(images=image, return_tensors="pt")
inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth  # shape: (batch, height, width)

# Interpolate the prediction to match the original image size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],  # image.size returns (width, height)
    mode="bicubic",
    align_corners=False,
)

# Access raw depth values from the prediction (still in metric units)
# Let's choose the center pixel as an example
height, width = prediction.shape[2], prediction.shape[3]
center_x, center_y = width // 2, height // 2
raw_depth_value = prediction[0, 0, center_y, center_x].item()
print(f"Raw depth value at center pixel ({center_x}, {center_y}): {raw_depth_value:.4f} (in model's metric units)")

# Squeeze to remove extra dimensions and move to CPU for further processing
depth_map = prediction.squeeze().cpu().numpy()

# Create a color-mapped visualization using matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(depth_map, cmap='plasma')
plt.axis('off')
plt.title('Depth Estimation')

plt.show()

# Save the depth map as an output image
depth_image = Image.fromarray(depth_map)
depth_image.save("depth_estimation_output.png")
print("Depth image saved as depth_estimation_output.png")
print("Values seem to be 3 times too big")