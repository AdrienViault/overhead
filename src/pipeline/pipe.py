import os
import random
import argparse
import time
from pathlib import Path
import cv2
from src.image_preprocessing.reproject_fisheye_distortion import project_equirectangular_left_right
from src.object_detection.object_detection import detect_objects
from src.image_metadata_extraction.metadata_extraction import get_gps_info
from src.object_detection.object_crop import crop_object_image
from src.object_detection.object_horizontal_angle_position import pixel_to_angle_perspective
from src.depth_estimation.depth_estimation import load_depth_model, run_depth_inference
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import json
import numpy as np


def decompose_image_path(image_path):
    """
    Decomposes the given image path into its directory, base name (without extension),
    and file extension.

    Args:
        image_path (str): The full path to the image.

    Returns:
        tuple: A tuple containing:
            - image_dir (str): The directory containing the image.
            - image_name (str): The name of the image file without its extension.
            - image_ext (str): The image file's extension (e.g., '.jpg').
    """
    image_dir = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    image_name, image_ext = os.path.splitext(base_name)
    return image_dir, image_name, image_ext

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
        equi_img_path,
        out_width ,
        horizontal_fov_deg ,
        vertical_fov_deg ,
        keep_top_crop_factor ,
):
    """
    Given an equirectangular image from a GoPro Max Sphere,
    return the left and right projected images.
    """
    # -------------------------
    # 1. Load the equirectangular image.
    # -------------------------
    equi_img = cv2.imread(equi_img_path)

    if equi_img is None:
        print("Error: Could not load the equirectangular image!")
        return
    
    persp_images, yaw_angles = project_equirectangular_left_right(
        equi_img, 
        out_width, 
        horizontal_fov_deg,
        vertical_fov_deg,
        keep_top_crop_factor
        )

    return persp_images, yaw_angles

def detect_elec_objects(
        image_path, 
        detection_model, 
        detection_processor
        ):
    """
    Apply your object detection algorithm on the image using the provided detection model.
    Return a list of detected objects with their positions.
    """

    threshold = 0.15

    # Load an image (ensure it's in RGB format)
    image = Image.open(image_path).convert("RGB")


    items_names =  [[
        "trash can",
        "letter box",
        "post box",
        "street lamp",
        "traffic light pole",
        "overhead utility power distribution line",
        "utility pole",
        "electricity management box",
    ]]

    ##create text_labels that is sames as items_names + "a photo of a" before or "a photo of an" if items starts with a vowel
    text_labels = []
    for items in items_names:
        for item in items:
            if item[0] in ['a', 'e', 'i', 'o', 'u']:
                text_labels.append(f"a photo of an {item}")
            else:
                text_labels.append(f"a photo of a {item}")
    
    text_labels = [text_labels]


    # Call the detection function
    boxes, scores, detected_labels = detect_objects(
        detection_model, 
        detection_processor, 
        image, 
        text_labels, 
        threshold=threshold,
    )
    return boxes, scores, detected_labels

def calculate_angle(image_path, box):
    #verify if box is a tensor
    if hasattr(box, 'tolist'):
        box = box.tolist()
    # box attributes
    xmin, ymin, xmax, ymax = box
    # get image dimensions
    image = Image.open(image_path)
    image_width, image_height = image.size
    # get center of the box
    x_center = (xmin + xmax) // 2
    y_center = (ymin + ymax) // 2
    # get angle
    angle = pixel_to_angle_perspective(
        image_width, 
        x_center
        )
    return angle

def get_leftside_image_relative_angle(
    yaw_angle,
    horizontal_fov_deg
):

    return yaw_angle - horizontal_fov_deg/2



def estimate_depth(depth_model, depth_processor, image_path, device):
    """
    Estimate the metric depth for the cropped image of the object.
    """
    image = Image.open(image_path).convert("RGB")
    depth_map = run_depth_inference(depth_model, depth_processor, image, device)
    return depth_map

def empirical_depth_correction(predicted_depth):
    return predicted_depth/3.25


def get_pixel_depth(depth, box):
    """
    for a depth map depth, and a box box, get middle pixel of the box and return depth value
    TODO for later : improve this method to make sure the depth correspounds to the detected object
    maybe center pixel is not in the object ?
    """
    #if necessary convert box from tensor to array
    if hasattr(box, 'tolist'):
        xmin, ymin, xmax, ymax = map(int, box.tolist())
    else:
        xmin, ymin, xmax, ymax = map(int, box)

    #get central pixel in the box
    x_center = (xmin + xmax) // 2
    y_center = (ymin + ymax) // 2

    # Ensure the coordinates are within the depth map dimensions
    x_center = min(max(x_center, 0), depth.shape[1] - 1)
    y_center = min(max(y_center, 0), depth.shape[0] - 1)

    # Get the depth value at the central pixel
    depth_value = depth[y_center, x_center]

    # Apply empirical depth correction
    corrected_depth_value = empirical_depth_correction(depth_value)

    return corrected_depth_value

# def add_point_to_depth_image(cropped_depth_image):
#     #get the center of the image
#     center = (cropped_depth_image.shape[1] // 2, cropped_depth_image.shape[0] // 2)
#     #add a point at the center of the image
#     cv2.circle(cropped_depth_image, center, 5, (255, 255, 255), -1)
#     #write the value of the point
#     cv2.putText(cropped_depth_image, str(cropped_depth_image[center[1], center[0]]), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#     return cropped_depth_image, cropped_depth_image[center[1], center[0]]





def init_image_metadata(        
        image_path,
        source_gps_metadata,
        side
        ):
    # verify that source_gps_metadata is a dict
    if not isinstance(source_gps_metadata, dict):
        raise ValueError("source_gps_metadata should be a dictionary")
    
    #parse components of source_gps_metadata and convert if there are binary elements
    # then convert them to string and print a warning
    for key, value in source_gps_metadata.items():
        if isinstance(value, bytes):
            source_gps_metadata[key] = value.decode()
            print(f"Warning: {key} is binary, converting to string")

    #create dictionary with source image information
    # initiate dictionary with a source element that will contain
    # gps data from source image
    # path to source image
    metadata_dict = {
        'source': {
            'GPS_metadata': source_gps_metadata,
            'path': image_path
        },
        'side': side
    }
    
    return metadata_dict

def update_metadata(
        image_relative_metadata,
        projection_path,
        cropped_path,
        croped_depth_path,
        processed_dir, 
        detected_label,
        score,
        idx,
        box,
        relative_angle, 
        absolute_angle,
        depth_value,
        
    ):
    """
    add to image_relative_metadata objects a new one with the info given in input

    """
    # verify that image_relative_metadata is a dict
    if not isinstance(image_relative_metadata, dict):
        raise ValueError("image_relative_metadata should be a dictionary")
    #check if box is a tensor
    if hasattr(box, 'tolist'):
        box = box.tolist()

     # Ensure numeric values are plain Python numbers
    try:
        score = float(score)
    except Exception:
        score = score  # or handle the conversion error as needed

    try:
        relative_angle = float(relative_angle)
    except Exception:
        relative_angle = relative_angle

    try:
        absolute_angle = float(absolute_angle)
    except Exception:
        absolute_angle = absolute_angle

    try:
        depth_value = float(depth_value)
    except Exception:
        depth_value = depth_value

    # Ensure paths and label are strings
    projection_path = str(projection_path)
    cropped_path = str(cropped_path)
    croped_depth_path = str(croped_depth_path)
    detected_label = str(detected_label)


    #create dictionary with object information
    obj_dict = {
        'projection_path': projection_path,
        'crop_path': cropped_path,
        'depth_path': croped_depth_path,
        'label': detected_label,
        'score': score,
        'index': idx,
        'bounding_box': {
            'xmin': box[0],
            'ymin': box[1],
            'xmax': box[2],
            'ymax': box[3]
        },
        'relative_angle': relative_angle,
        'absolute_angle': absolute_angle,
        'depth': depth_value,
    }
    #verify if objects key exists in image metadata
    if 'objects' not in image_relative_metadata:
        image_relative_metadata['objects'] = []
    #add object to image metadata
    image_relative_metadata['objects'].append(obj_dict)

    return image_relative_metadata



def save_metadata(processed_dir, image_source_name, metadata, side):
    """
    Save or log the metadata for the processed object in a JSON file.
    Before saving, print out the type information for all keys/values.
    """
    # Verify that metadata is a dictionary  
    if not isinstance(metadata, dict):
        raise ValueError("metadata should be a dictionary")
    
    metadata_path = os.path.join(processed_dir, f"{image_source_name}_{side}_metadata.json")
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved metadata for {image_source_name} in {metadata_path}")
    except TypeError as e:
        print("TypeError during json.dump:", e)
        # Optionally, reprint the metadata structure for further debugging.
        raise  

def process_image(
        image_path, 
        output_base, 
        detection_model,
        detection_processor,
        depth_model,
        depth_processor,
        device
        ):
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
    processed_dir = os.path.join(output_base, user, sequence, batch)
    os.makedirs(processed_dir, exist_ok=True)
    image_dir, image_name, image_ext = decompose_image_path(image_path)
   
    # Check if a left projection image already exists.
    left_proj_img_file_name = f"{image_name}_perspective_left.jpg"
    left_img_path = os.path.join(processed_dir, left_proj_img_file_name)
    if os.path.exists(left_img_path):
        print(f"Projection already exists for {image_path}. Skipping processing.")
        return

    source_gps_metadata = get_gps_info(image_path)
    # Get image angle from GPS data; default to 0 if not present.
    if 'GPSImgDirection' not in source_gps_metadata:
        source_angle = 0
        print(f"Warning: No GPSImgDirection found in {image_path}. Setting source angle to {source_angle}")
    else:
        source_angle = source_gps_metadata['GPSImgDirection']

    horizontal_fov_deg = 90
    # --- Projection step ---
    projections, yaw_angles = projection(
        equi_img_path=image_path,
        out_width=1080,
        horizontal_fov_deg=horizontal_fov_deg,
        vertical_fov_deg=140,
        keep_top_crop_factor=2 / 3,
    )
    if projections is None:
        print(f"Projection failed for {image_path}")
        return
    left_img, right_img = projections

    right_proj_img_file_name = f"{image_name}_perspective_right.jpg"
    right_img_path = os.path.join(processed_dir, right_proj_img_file_name)
    cv2.imwrite(left_img_path, left_img)
    cv2.imwrite(right_img_path, right_img)
    
    print(f"Saved perspective images for {image_path} as {left_proj_img_file_name} and {right_proj_img_file_name} in {processed_dir}")

    for side, proj_img_name, proj_img_path, yaw in zip(
        ['left', 'right'],
        [left_proj_img_file_name, right_proj_img_file_name],
        [left_img_path, right_img_path],
        yaw_angles
        ):
        side_image_relative_metadata = init_image_metadata(
            image_path,
            source_gps_metadata, 
            side
            )

        # Get image leftside relative angle.
        relative_leftside_angle = get_leftside_image_relative_angle(yaw, horizontal_fov_deg)

        # --- Object detection and processing ---
        boxes, scores, detected_labels  = detect_elec_objects(
            proj_img_path, 
            detection_model,
            detection_processor
            )
        
        if len(scores) > 0:
            depth_map = estimate_depth(
                depth_model,
                depth_processor,
                proj_img_path,
                device
                )
            # Normalize the depth map for visualization and save it as an image (8-bit conversion)
            depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_image = Image.fromarray((depth_norm * 255).astype(np.uint8))
            depth_map_file_name = f"{path_obj.stem}_{side}_depth_map.jpg"
            depth_map_path = os.path.join(processed_dir, depth_map_file_name)
            depth_image.save(depth_map_path)

            for idx, box, score, detected_label in zip(
                range(len(boxes)),
                boxes,
                scores,
                detected_labels
            ):
                # Remove "a photo of a" or "a photo of an" from beginning of detected_label.
                if detected_label.startswith("a photo of an "):
                    detected_label = detected_label[13:]
                elif detected_label.startswith("a photo of a "):
                    detected_label = detected_label[12:]

                detected_label_nospace = detected_label.replace(" ", "_")

                # Crop the object from the image.
                cropped_img = crop_object_image(proj_img_path, box)
                cropped_depth_img = crop_object_image(depth_map_path, box)
                cropped_img_file_name = f"{path_obj.stem}_{side}_obj{idx}_{detected_label_nospace}_img.jpg"
                cropped_depth_map_name = f"{path_obj.stem}_{side}_obj{idx}_{detected_label_nospace}_depth_map.jpg"
                cropped_img_path = os.path.join(processed_dir, cropped_img_file_name)
                cropped_depth_map_path = os.path.join(processed_dir, cropped_depth_map_name)
                cv2.imwrite(cropped_img_path, cropped_img)
                cv2.imwrite(cropped_depth_map_path, cropped_depth_img)

                object_relative_angle = calculate_angle(image_path, box)
                object_absolute_angle = source_angle + relative_leftside_angle + object_relative_angle
                
                depth_value = get_pixel_depth(depth_map, box)

                # Update metadata with the detected object info.
                update_metadata(
                    side_image_relative_metadata,
                    proj_img_path,
                    cropped_img_path,
                    cropped_depth_map_path,
                    processed_dir,
                    detected_label,
                    score,
                    idx,
                    box,
                    object_relative_angle,
                    object_absolute_angle,
                    depth_value,
                    )
        save_metadata(
            processed_dir,
            image_name,
            side_image_relative_metadata,
            side
            )


def main():
    start_time = time.time()
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
    

    # Initialize the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize heavy models once outside the loop.
    # For instance, load your object detection model (assuming a function load_detection_model exists)
    detection_model, detection_processor = load_detection_model()

    # Load the depth estimation model
    depth_processor, depth_model = load_depth_model(device)

    # Recursively search for image files (assuming JPG/JPEG)
    image_files = []
    for dirpath, _, filenames in os.walk(args.root):
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg")):
                image_files.append(os.path.join(dirpath, fname))

    print(f"Found {len(image_files)} image(s) in the dataset.")

    # In test mode, sample 100 random images from the tree.
    if args.test:
        image_files = random.sample(image_files, min(100, len(image_files)))
        print("Test mode enabled: processing 100 random images.")

    # Process each image in the list, timing each iteration.
    for image_path in image_files:
        print(f"Processing image: {image_path}")
        process_image(
            image_path, 
            args.output, 
            detection_model,
            detection_processor,
            depth_model,
            depth_processor,
            device
            )
    stop_time = time.time()
    print(f"Processing complete in {stop_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
