# Overhead Image Processing Pipeline

This project implements a state-of-the-art image processing pipeline designed for aerial imagery. It leverages transformer-based AI models for two key tasks:

- **Object Detection:** Utilizes a zero-shot approach with the OwlViT model for detecting a variety of electrical objects.
- **Depth Estimation:** Uses a state-of-the-art depth model (referred to as "deth anything zero shot outdoor") for estimating metric depth.

The pipeline processes equirectangular images (typically from GoPro Max Sphere cameras) by projecting them into left/right perspective views, detecting objects, cropping the images around these objects, estimating depth, and finally saving all the metadata.

## Table of Contents

- [Overview](#overview)
- [Key Components](#key-components)
  - [Utility Functions](#utility-functions)
  - [Model Loading](#model-loading)
  - [Image Projection](#image-projection)
  - [Object Detection and Post-processing](#object-detection-and-post-processing)
  - [Depth Estimation](#depth-estimation)
  - [Metadata Handling](#metadata-handling)
- [Pipeline Execution](#pipeline-execution)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Notes](#notes)

## Overview

The pipeline is executed via the `main()` function, which:
- Loads the necessary models and processors.
- Recursively searches through a raw image directory for JPG/JPEG files.
- Optionally samples a subset of images for test runs.
- Processes each image by:
  - Projecting the equirectangular image into two perspective views.
  - Running object detection to identify electrical objects (e.g., street lamps, utility poles, transformers, etc.).
  - Calculating angles for detected objects using both image coordinates and GPS metadata.
  - Estimating the depth of the detected objects.
  - Cropping objects from the perspective and depth maps.
  - Logging and saving all relevant metadata in JSON format.

All functions are designed to work together to transform raw image data into a set of processed images with associated metadata, suitable for further analysis or visualization.

## Key Components

### Utility Functions

- **`decompose_image_path(image_path)`**  
  Splits the full image path into directory, base name (without extension), and extension. This function is used to generate filenames and directories for processed output.

### Model Loading

- **`load_detection_model()`**  
  Loads the OwlViT processor and model for object detection. It also checks for GPU availability and transfers the model to GPU if possible. GPU memory information is printed both before and after the model transfer for debugging purposes.

- **`load_depth_model(device)`**  
  (Imported from an external module) Loads the depth estimation model and processor for the pipeline. Depth estimation is performed using a transformer-based approach optimized for outdoor scenes.

### Image Projection

- **`projection(equi_img_path, out_width, horizontal_fov_deg, vertical_fov_deg, keep_top_crop_factor)`**  
  Projects a GoPro Max Sphere equirectangular image into left and right perspective images using a fisheye distortion re-projection function. Returns the two perspective images and their corresponding yaw angles.

### Object Detection and Post-processing

- **`detect_elec_objects(image_path, detection_model, detection_processor)`**  
  Runs the object detection algorithm on a given image. The function searches for a predefined list of text labels corresponding to various electrical objects. It returns bounding boxes, confidence scores, and detected labels.

- **`calculate_angle(image_path, box)`**  
  Converts the center pixel position of a detected object bounding box into an angle relative to the image perspective. This angle is later used to compute the object's absolute orientation using GPS data.

- **`get_leftside_image_relative_angle(yaw_angle, horizontal_fov_deg)`**  
  Computes the relative angle for the left side of the projected image based on the yaw angle and the horizontal field-of-view.

### Depth Estimation

- **`estimate_depth(depth_model, depth_processor, image_path, device)`**  
  Performs depth inference on a given image using the loaded depth model and processor.

- **`empirical_depth_correction(predicted_depth)`**  
  Applies an empirical correction factor (dividing by 3.25) to the predicted depth values to adjust the metric depth estimates.

- **`get_pixel_depth(depth, box)`**  
  Extracts the depth value from the depth map at the center of the object’s bounding box and applies the empirical correction.

- **`add_point_to_depth_image(cropped_depth_image)`**  
  Marks the center of the cropped depth image by drawing a point and overlaying the depth value. This helps visualize the depth measurement on the cropped object image.

### Metadata Handling

- **`init_image_metadata(image_path, source_gps_metadata, side)`**  
  Initializes a metadata dictionary using GPS data extracted from the source image. This metadata serves as a container for additional object-specific information.

- **`update_metadata(image_relative_metadata, projection_path, cropped_path, croped_depth_path, processed_dir, detected_label, score, idx, box, relative_angle, absolute_angle, depth_value)`**  
  Updates the metadata dictionary with details for each detected object, including file paths, bounding box coordinates, detection confidence, calculated angles, and depth estimates.

- **`save_metadata(processed_dir, image_source_name, metadata, side)`**  
  Saves the aggregated metadata as a JSON file in the specified processed directory.

## Pipeline Execution

- **`process_image(image_path, output_base, detection_model, detection_processor, depth_model, depth_processor, device)`**  
  The main processing function for a single image. It:
  1. Parses the image path to determine the directory structure (user, sequence, batch).
  2. Creates the corresponding output folder.
  3. Extracts GPS metadata from the image.
  4. Performs the projection step to generate left and right perspective views.
  5. Runs object detection on each perspective image.
  6. Estimates the depth for each detected object.
  7. Crops the object from both the perspective image and depth map.
  8. Updates and saves the metadata for further analysis.

- **`main()`**  
  The entry point of the pipeline:
  1. Parses command-line arguments to set the raw image directory, output directory, and test mode.
  2. Loads the object detection and depth estimation models.
  3. Recursively finds image files in the raw directory.
  4. Optionally samples a subset of images if test mode is enabled.
  5. Iterates over the images, calling `process_image` for each.
  6. Reports the total processing time.

## Usage

To run the pipeline, use the command-line interface. For example:

```bash
python pipe.py --root /path/to/raw/images --output /path/to/processed/output --test
