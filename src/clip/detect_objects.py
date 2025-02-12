import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load the SAM2 model
sam2_checkpoint = "/home/adrien/Documents/Dev/sam2/checkpoints/sam2.1_hiera_large.pt"    # Replace with the path to the SAM2 checkpoint
model_cfg = "/home/adrien/Documents/Dev/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"  # Replace with the path to the SAM2 config file
device = "cuda" if torch.cuda.is_available() else "cpu"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Function to segment objects using SAM2 with different settings
def segment_objects_sam2(image_path, point_coords, point_labels, multimask_output=True):
    image = preprocess_image(image_path)
    predictor.set_image(image)

    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=multimask_output,
    )
    return masks, image

# Function to visualize the segmentation results
def visualize_segmentation(image, masks, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)

    for mask in masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(image, [contour], -1, np.random.randint(0, 255, 3).tolist(), 2)

    plt.imshow(image)
    plt.show()

# Example usage with different settings
image_path = "path/to/your/image.jpg"  # Replace with your image path

# Setting 1: Coarse granularity with few points
point_coords_1 = np.array([[x, y] for x in range(0, 500, 100) for y in range(0, 500, 100)])
point_labels_1 = np.ones(point_coords_1.shape[0])
masks_1, image_1 = segment_objects_sam2(image_path, point_coords_1, point_labels_1, multimask_output=True)
visualize_segmentation(image_1, masks_1, "Coarse Granularity")

# Setting 2: Fine granularity with more points
point_coords_2 = np.array([[x, y] for x in range(0, 500, 50) for y in range(0, 500, 50)])
point_labels_2 = np.ones(point_coords_2.shape[0])
masks_2, image_2 = segment_objects_sam2(image_path, point_coords_2, point_labels_2, multimask_output=True)
visualize_segmentation(image_2, masks_2, "Fine Granularity")

# Setting 3: Single mask output
point_coords_3 = np.array([[x, y] for x in range(0, 500, 100) for y in range(0, 500, 100)])
point_labels_3 = np.ones(point_coords_3.shape[0])
masks_3, image_3 = segment_objects_sam2(image_path, point_coords_3, point_labels_3, multimask_output=False)
visualize_segmentation(image_3, masks_3, "Single Mask Output")
