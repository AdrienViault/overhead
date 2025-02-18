from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

def load_depth_model(device):
    """
    Loads the image processor and depth estimation model, and moves the model to the specified device.
    
    Args:
        device (torch.device): Device to load the model on (e.g. "cuda" or "cpu").
        
    Returns:
        image_processor: The image processor.
        model: The depth estimation model on the given device.
        load_time (float): Time taken to load and transfer the model.
    """
    start_model = time.time()
    
    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
    
    # Move the model to the designated device
    if device.type == "cuda":
        torch.cuda.synchronize()
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_time = time.time() - start_model
    print(f"Model loaded to {device} in {load_time:.3f} seconds.")
    
    # Print GPU memory info after model loading
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print(f"GPU memory allocated after loading model: {allocated:.0f} MB")
        print(f"GPU memory reserved after loading model: {reserved:.0f} MB")
    
    return image_processor, model

def run_depth_inference(model, image_processor, image, device):
    """
    Prepares the image, runs inference, and returns the interpolated depth map.
    
    Args:
        model: The depth estimation model.
        image_processor: The corresponding image processor.
        image (PIL.Image): The input image in RGB.
        device (torch.device): The device for inference.
        
    Returns:
        depth_map (np.ndarray): The computed depth map (interpolated to original image size).
        raw_center_depth (float): Depth value at the center pixel.
        inference_time (float): Time taken for the inference.
    """
    # Prepare inputs and move them to the device
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    
    # Run inference and time it
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_inference = time.time()
        outputs = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time = time.time() - start_inference
    print(f"Inference completed in {inference_time:.3f} seconds.")
    
    # Retrieve predicted depth and interpolate to original image size
    predicted_depth = outputs.predicted_depth  # shape: (batch, height, width)
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],  # image.size returns (width, height), so reverse for (height, width)
        mode="bicubic",
        align_corners=False,
    )
    
    # Squeeze extra dimensions and move to CPU for further processing
    depth_map = prediction.squeeze().cpu().numpy()
    
    return depth_map

def visualize_depth_output(depth_map, image, save_path="depth_estimation_output.png"):
    """
    Visualizes the depth map using matplotlib and saves the output as an image.
    
    Args:
        depth_map (np.ndarray): The depth map.
        image (PIL.Image): The original image (used to compare size/resolution).
        save_path (str): Path where the output image will be saved.
    """
    # Create a color-mapped visualization using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map, cmap='plasma')
    plt.axis('off')
    plt.title('Depth Estimation')
    plt.show()
    
    # Normalize the depth map for visualization and save it as an image (8-bit conversion)
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_image = Image.fromarray((depth_norm * 255).astype(np.uint8))
    depth_image.save(save_path)
    print(f"Depth image saved as {save_path}")
    print("Values seem to be 3.25 times too big")

def main():
    # Load image from file and ensure it's in RGB format
    image_path = "data/images/test_images/reprojected/GSAC0346_perspective_right.jpg"  # Replace with your image path
    image = Image.open(image_path).convert("RGB")
    
    # Initialize device: GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory / (1024 ** 2)  # in MB
        print(f"Total GPU memory: {total_memory:.0f} MB")
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print(f"GPU memory allocated before loading model: {allocated:.0f} MB")
        print(f"GPU memory reserved before loading model: {reserved:.0f} MB")
    
    # Load model and image processor
    image_processor, model = load_depth_model(device)
    
    # Run depth estimation inference
    depth_map = run_depth_inference(model, image_processor, image, device)
    
    # Visualize and save the depth output
    visualize_depth_output(depth_map, image)

if __name__ == "__main__":
    main()
