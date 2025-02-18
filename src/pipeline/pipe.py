import os
import random
import argparse
import time
from pathlib import Path
import cv2
from src.image_preprocessing.reproject_fisheye_distortion import project_equirectangular_left_right
from src.object_detection.object_detection import detect_objects
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image



def load_detection_model():
    # Load processor and model (assume the model is then moved to GPU)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # Print GPU memory info before model loading
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory / (1024 ** 2)  # in MB
        print(f"Total GPU memory: {total_memory:.0f} MB")
    else:
        print("CUDA is not available.")
        device = torch.device("cpu")

    # Time the model transfer to GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_model = time.time()
        model.to("cuda")  # Move the model to GPU
        torch.cuda.synchronize()
        model_transfer_time = time.time() - start_model
        print(f"Object detection model loaded to GPU in {model_transfer_time:.3f} seconds.")
    else:
        model.to("cpu")

    # Print GPU memory info after model loading
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # in MB
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)    # in MB
        print(f"GPU memory allocated by PyTorch after loading model: {allocated:.0f} MB")
        print(f"GPU memory reserved by PyTorch: {reserved:.0f} MB")
        print("Note: GPU VRAM is used for storing the model parameters, intermediate activations during inference, "
              "and also the input data (like your image batch). Maximum occupancy will include all these factors.")
        print("Ensure there is enough free memory for additional tasks (like depth estimation) to run concurrently.")
    return model, processor

# Import your image processing libraries (e.g., Pillow, OpenCV)
# from PIL import Image

# Placeholder functions – replace these with your actual implementations or module imports.
def projection(
        equi_img_dir_path="data/images/test_images/",
        equi_img_name= 'GSAC0346',
        equi_img_extension= '.JPG',
        out_width = 1080,
        horizontal_fov_deg = 90,
        vertical_fov_deg = 140,
        keep_top_crop_factor = 2/3,
        out_dir_path = "data/images/test_images/reprojected/",
):
    """
    Given an equirectangular image from a GoPro Max Sphere,
    return the left and right projected images.
    """
    # -------------------------
    # 1. Load the equirectangular image.
    # -------------------------
    equi_img = cv2.imread(equi_img_dir_path+equi_img_name+equi_img_extension)

    if equi_img is None:
        print("Error: Could not load the equirectangular image!")
        return
    
    persp_images = project_equirectangular_left_right(
        equi_img, 
        out_width, 
        horizontal_fov_deg,
        vertical_fov_deg,
        keep_top_crop_factor
        )
    pic_suffixes = ['left', 'right']
    proj_img_file_names = []
    for pic_suffix in pic_suffixes:
        proj_img_file_names.append(f"{equi_img_name}_perspective_{pic_suffix}.jpg")

    for persp_image, pic_name, pic_suffix in zip(persp_images, proj_img_file_names, pic_suffixes):    
        filename = out_dir_path + pic_name
        cv2.imwrite(filename, persp_image)
        print(f"Saved perspective image for {pic_suffix} side as {filename}")
    pass

def detect_objects(
        image_path, 
        detection_model, 
        detection_processor
        ):
    """
    Apply your object detection algorithm on the image using the provided detection model.
    Return a list of detected objects with their positions.
    """

    threshold = 0.1

    # Load an image (ensure it's in RGB format)
    image = Image.open(image_path).convert("RGB")

    # Define text labels to search for
    text_labels = [[
        "a photo of a street lamp", 
        "a photo of an overhead utility power distribution line",
        "a photo of a n overhead tram power line",
        "a photo of a safety cone",
        "a photo of a Single-phase low-voltage pole",
        "a photo of a Three-phase low-voltage pole with neutral",
        "a photo of a Three-phase low-voltage pole without neutral",
        "a photo of a Three-phase medium-voltage pole",
        "a photo of a Three-phase medium-voltage pole with ground wire",
        "a photo of a Three-phase high-voltage transmission tower",
        "a photo of a Combined utility pole (power + telecom)",
        "a photo of a Pole-Mounted Transformers",
        "a photo of a switchgear",
        "a photo of an underground Distribution Box",
        "a photo of a Remote Terminal Units",
        "a photo of a transformer",
        "a photo of a substation",
        "a photo of a secondary substation",
        "a photo of a busbar",
        "a photo of a surge arrester",
        "a photo of a grounding system",
        ]]

    # Call the detection function
    boxes, scores, detected_labels = detect_objects(
        detection_model, 
        detection_processor, 
        image, 
        text_labels, 
        threshold=threshold,
    )
    return boxes, scores, detected_labels

def crop_object(image_path, obj):
    """
    Crop the detected object region from the original image.
    """
    # cropped_img = ...
    # return cropped_img
    pass

def calculate_angle(obj, image_path):
    """
    Calculate the angle (relative to the sphere) for the detected object.
    """
    # angle = ...
    # return angle
    pass

def estimate_depth(cropped_img):
    """
    Estimate the metric depth for the cropped image of the object.
    """
    # depth = ...
    # return depth
    pass

def extract_metadata(image_path):
    """
    Extract metadata such as the observer's position and direction from the image.
    """
    # metadata = ...
    # return metadata
    pass

def save_metadata(processed_dir, image_stem, obj, angle, depth, metadata):
    """
    Save or log the metadata for the processed object.
    This might be in a CSV, JSON, or a database.
    """
    # For example, write to a JSON or CSV file:
    # with open(os.path.join(processed_dir, f"{image_stem}_metadata.json"), "a") as f:
    #     json.dump({...}, f)
    pass

def process_image(image_path, output_base, detection_model):
    """
    Process a single image:
    - Parse the directory structure to determine the user, sequence, and batch.
    - Create a corresponding output folder under the processed tree.
    - Run projection, detection, cropping, angle calculation, and depth estimation.
    - Extract and save metadata.
    """
    # Convert to Path object for easier manipulation
    path_obj = Path(image_path)
    parts = path_obj.parts

    # Expecting structure: .../raw/Grenoble/<user>/<sequence>/<batch>/image_name.JPG
    try:
        raw_index = parts.index("raw")
    except ValueError:
        print(f"Warning: 'raw' not found in path {image_path}")
        return

    # Check if the path has enough parts after 'raw'
    if len(parts) < raw_index + 5:
        print(f"Invalid path structure for {image_path}")
        return

    # Extract the subdirectories (assuming fixed order)
    user     = parts[raw_index + 2]
    sequence = parts[raw_index + 3]
    batch    = parts[raw_index + 4]

    # Build the corresponding processed directory:
    processed_dir = os.path.join(output_base, "Grenoble", user, sequence, batch)
    os.makedirs(processed_dir, exist_ok=True)

    # --- Projection step ---
    projections = projection(image_path)
    if projections is None:
        print(f"Projection failed for {image_path}")
        return
    left_img, right_img = projections

    left_path = os.path.join(processed_dir, path_obj.stem + "_left.jpg")
    right_path = os.path.join(processed_dir, path_obj.stem + "_right.jpg")
    # Example: left_img.save(left_path) and right_img.save(right_path)
    # Uncomment above lines after integrating your image library

    # --- Object detection and processing ---
    objects = detect_objects(image_path, detection_model)
    metadata = extract_metadata(image_path)
    for idx, obj in enumerate(objects or []):
        cropped_img = crop_object(image_path, obj)
        angle = calculate_angle(obj, image_path)
        depth = estimate_depth(cropped_img)
        obj_path = os.path.join(processed_dir, f"{path_obj.stem}_obj{idx}.jpg")
        # Example: cropped_img.save(obj_path)
        save_metadata(processed_dir, path_obj.stem, obj, angle, depth, metadata)

def main():
    parser = argparse.ArgumentParser(description="Process GoPro Max Sphere images through the pipeline.")
    parser.add_argument(
        "--root",
        type=str,
        default="/media/adrien/Space/Datasets/Overhead/raw/Grenoble",
        help="Root directory of the raw images."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/media/adrien/Space/Datasets/Overhead/processed/Grenoble",
        help="Root directory where processed images will be stored."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode: process only 10 random images from the dataset."
    )
    args = parser.parse_args()

    # Initialize heavy models once outside the loop.
    # For instance, load your object detection model (assuming a function load_detection_model exists)
    # detection_model = load_detection_model()
    detection_model = None  # Replace with your model loading code

    # Recursively search for image files (assuming JPG/JPEG)
    image_files = []
    for dirpath, _, filenames in os.walk(args.root):
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg")):
                image_files.append(os.path.join(dirpath, fname))

    print(f"Found {len(image_files)} image(s) in the dataset.")

    # In test mode, sample 10 random images from the tree.
    if args.test:
        image_files = random.sample(image_files, min(10, len(image_files)))
        print("Test mode enabled: processing 10 random images.")

    # Process each image in the list, timing each iteration.
    for image_path in image_files:
        start_time = time.time()
        print(f"Processing image: {image_path}")
        process_image(image_path, args.output, detection_model)
        iteration_time = time.time() - start_time
        print(f"Processed {image_path} in {iteration_time:.2f} seconds.")

if __name__ == "__main__":
    main()
