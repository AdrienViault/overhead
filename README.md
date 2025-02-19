# Overhead Image Processing Pipeline

This project implements a state-of-the-art image processing pipeline designed for aerial imagery. It leverages transformer-based AI models for two key tasks:

- **Object Detection:** Uses a zero-shot approach with the OwlViT model to detect various electrical objects.
- **Depth Estimation:** Utilizes a state-of-the-art depth model (referred to as "deth anything zero shot outdoor") for estimating metric depth.

The pipeline processes equirectangular images (typically from GoPro Max Sphere cameras) by projecting them into left/right perspective views, detecting objects, cropping around detected objects, estimating depth, and saving metadata.

## Table of Contents

- [Overview](#overview)
- [Data Acquisition](#data-acquisition)
- [Key Components](#key-components)
  - [Utility Functions](#utility-functions)
  - [Model Loading](#model-loading)
  - [Image Projection](#image-projection)
  - [Object Detection and Post-processing](#object-detection-and-post-processing)
  - [Depth Estimation](#depth-estimation)
  - [Metadata Handling](#metadata-handling)
- [Pipeline Execution](#pipeline-execution)
- [Visualization](#visualization)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Notes](#notes)

## Overview

The pipeline is executed via the `main()` function. It:
- Loads the necessary models and processors.
- Recursively searches through a raw image directory for JPG/JPEG files.
- Optionally samples a subset of images for testing.
- Processes each image by:
  - Projecting the equirectangular image into two perspective views.
  - Running object detection to identify electrical objects (e.g., street lamps, utility poles, transformers, etc.).
  - Calculating angles for detected objects using image coordinates and GPS metadata.
  - Estimating the depth of detected objects.
  - Cropping objects from both perspective images and depth maps.
  - Logging and saving all relevant metadata in JSON format.

All functions work together to transform raw image data into processed images with associated metadata for further analysis or visualization.

## Data Acquisition

The images used in this project are street view open-source images sourced from **Panoramax OpenStreetMap** data in the Grenoble area. Scripts in the `src/panoramax_get` folder download and preprocess these images, providing a realistic urban environment for testing object detection and depth estimation models.

## Key Components

### Utility Functions

- **`decompose_image_path(image_path)`**  
  Splits an image path into its directory, base name (without extension), and file extension.

### Model Loading

- **`load_detection_model()`**  
  Loads the OwlViT processor and model for object detection. It checks for GPU availability and transfers the model to GPU if possible, printing GPU memory information before and after transfer.

- **`load_depth_model(device)`**  
  Loads the depth estimation model and processor (imported from an external module). Depth estimation is optimized for outdoor scenes using a transformer-based approach.

### Image Projection

- **`projection(equi_img_path, out_width, horizontal_fov_deg, vertical_fov_deg, keep_top_crop_factor)`**  
  Projects a GoPro Max Sphere equirectangular image into left/right perspective images using fisheye distortion re-projection. Returns the two images and their corresponding yaw angles.

### Object Detection and Post-processing

- **`detect_elec_objects(image_path, detection_model, detection_processor)`**  
  Runs object detection on an image, searching for predefined text labels corresponding to electrical objects. Returns bounding boxes, confidence scores, and detected labels.

- **`calculate_angle(image_path, box)`**  
  Converts the center of a detected object’s bounding box into an angle relative to the image perspective, to later compute the object's absolute orientation using GPS data.

- **`get_leftside_image_relative_angle(yaw_angle, horizontal_fov_deg)`**  
  Computes the relative angle for the left side of the projected image using the yaw angle and horizontal field-of-view.

### Depth Estimation

- **`estimate_depth(depth_model, depth_processor, image_path, device)`**  
  Performs depth inference on an image using the loaded depth model and processor.

- **`empirical_depth_correction(predicted_depth)`**  
  Applies a correction factor (dividing by 3.25) to the predicted depth values.

- **`get_pixel_depth(depth, box)`**  
  Extracts the depth value from the depth map at the center of the bounding box and applies the empirical correction.

- **`add_point_to_depth_image(cropped_depth_image)`**  
  Marks the center of the cropped depth image by drawing a point and overlaying the depth value for visualization.

### Metadata Handling

- **`init_image_metadata(image_path, source_gps_metadata, side)`**  
  Initializes a metadata dictionary using GPS data from the source image.

- **`update_metadata(image_relative_metadata, projection_path, cropped_path, croped_depth_path, processed_dir, detected_label, score, idx, box, relative_angle, absolute_angle, depth_value)`**  
  Updates the metadata with details for each detected object, including file paths, bounding box coordinates, detection confidence, calculated angles, and depth estimates.

- **`save_metadata(processed_dir, image_source_name, metadata, side)`**  
  Saves the aggregated metadata as a JSON file in the specified processed directory.

## Pipeline Execution

- **`process_image(image_path, output_base, detection_model, detection_processor, depth_model, depth_processor, device)`**  
  Processes a single image by:
  1. Parsing the image path to determine the directory structure (user, sequence, batch).
  2. Creating the corresponding output folder.
  3. Extracting GPS metadata.
  4. Performing projection to generate left/right perspective views.
  5. Running object detection on each perspective image.
  6. Estimating depth for detected objects.
  7. Cropping objects from both perspective images and depth maps.
  8. Updating and saving metadata.

- **`main()`**  
  The main function:
  1. Parses command-line arguments for the raw image directory, output directory, and test mode.
  2. Loads the detection and depth estimation models.
  3. Recursively finds image files in the raw directory.
  4. Optionally samples a subset of images if test mode is enabled.
  5. Iterates over images and processes each one.
  6. Reports the total processing time.

## Visualization

Scripts in the `src/visualization` folder help visualize processed outputs on a map. These tools overlay detected objects and metadata (GPS coordinates, angles, depth estimates) onto geographic maps for easier spatial analysis.

## Usage

To run the pipeline, open your terminal and execute the following command:

    python pipe.py --root /path/to/raw/images --output /path/to/processed/output --test

Options:
- **--root:** Path to the directory containing raw images.
- **--output:** Path to the directory where processed images and metadata will be saved.
- **--test:** (Optional) Flag to process a random subset (up to 1000 images) for quick testing.

## Dependencies

Ensure you have installed the following:
- Python 3.x
- OpenCV
- Pillow (PIL)
- PyTorch
- Transformers (by Hugging Face)
- Additional custom modules for:
  - Image preprocessing (fisheye distortion re-projection)
  - GPS metadata extraction
  - Object cropping
  - Depth estimation

To install packages from PyPI, run:

    pip install opencv-python pillow torch transformers

Also, ensure that the custom modules under `src/` are accessible in your PYTHONPATH.

## Notes

- **GPU Acceleration:** The pipeline automatically detects a CUDA-compatible GPU and uses it for heavy model inference.
- **Error Handling:** The code includes checks (e.g., verifying image loading and path validation) to prevent runtime errors.
- **Modularity:** Each function is designed to be independent, allowing for easy updates or component replacements.
- **Visualization Tools:** The visualization scripts in `src/visualization` enable you to display processed results on a map, facilitating further spatial analysis.
