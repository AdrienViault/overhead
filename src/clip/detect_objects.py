import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the labels for electricity management hardware
labels = ["electricity pole", "transformer", "power line", "electric meter", "substation"]

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

# Function to detect objects in the image
def detect_objects(image_path):
    image = preprocess_image(image_path)
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().numpy()
    return probs

# Function to visualize the detection results
def visualize_detection(image_path, probs):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")

    for i, label in enumerate(labels):
        prob = probs[0][i]
        plt.text(10, 30 + i * 30, f"{label}: {prob:.2f}", fontsize=12, color="white", backgroundcolor="black")

    plt.show()

# Example usage
image_path = "/home/adrien/Documents/Dev/overhead/data/images/test_images/reprojected/perspective_0_GSAC0346.JPG"  # Replace with your image path
probs = detect_objects(image_path)
visualize_detection(image_path, probs)
