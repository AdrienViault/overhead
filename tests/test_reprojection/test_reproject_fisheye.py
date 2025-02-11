import os
import cv2
import pytest
from src.image_preprocessing.reproject_fisheye_distortion import equirectangular_to_perspective

def load_image(image_path):
    """Utility function to load an image from disk."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

@pytest.fixture
def test_images_dir():
    return os.path.join("data", "images", "test_images")

def test_reprojection_on_two_images(test_images_dir):
    # List of test image filenames (adjust as needed).
    image_files  = ["GSAC0346.JPG", "GSBF9163.JPG"]
    
    # Define parameters for the perspective view.
    fov = 90         # horizontal field of view in degrees
    phi = 0          # pitch remains 0 (no vertical tilt)
    out_hw = (600, 800)  # output image size (height, width)
    
    # Define the set of directions (yaw angles) to generate.
    directions = [0, 90, 180, 270]
    
    # Directory where the reprojected images will be saved.
    output_dir = os.path.join(test_images_dir, "reprojected")
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in image_files:
        input_path = os.path.join(test_images_dir, img_file)
        img = load_image(input_path)
        for theta in directions:
            persp = equirectangular_to_perspective(img, fov=fov, theta=theta, phi=0, out_hw=out_hw)
            output_filename = f"perspective_{theta}_{img_file}"
            output_path = os.path.join(output_dir, output_filename)
            success = cv2.imwrite(output_path, persp)
            assert success, f"Failed to save {output_path}"
            print(f"Saved perspective image for theta={theta} to: {output_path}")
