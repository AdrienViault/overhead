import os
import glob
import json
import random
import cv2
import shutil
import matplotlib.pyplot as plt
from collections import defaultdict

# ==========================
# Step 1: Load all detections from metadata JSON files.
# ==========================
def load_all_objects(root_dir):
    """Recursively search for *_metadata.json files in root_dir and return a list of all detected objects."""
    metadata_files = glob.glob(os.path.join(root_dir, "**", "*_metadata.json"), recursive=True)
    all_objects = []
    for meta_file in metadata_files:
        try:
            with open(meta_file, "r") as f:
                data = json.load(f)
            # Each metadata file is expected to have a key "objects" which is a list of detections.
            if "objects" in data:
                for obj in data["objects"]:
                    # Record which metadata file the detection came from.
                    obj["metadata_file"] = meta_file
                    all_objects.append(obj)
        except Exception as e:
            print(f"Error loading {meta_file}: {e}")
    return all_objects

# ==========================
# Step 2: Group objects by label and report counts.
# ==========================
def group_by_label(objects):
    """Return a dictionary mapping label -> list of objects."""
    objects_by_label = defaultdict(list)
    for obj in objects:
        label = obj.get("label", "unknown")
        objects_by_label[label].append(obj)
    return objects_by_label

def print_label_counts(objects_by_label):
    """Print the number of detections per label."""
    print("Number of detections per class:")
    for label, obj_list in objects_by_label.items():
        print(f"  {label}: {len(obj_list)}")

# ==========================
# Step 3: Plot confidence score distributions.
# ==========================
def plot_confidence_distributions(objects_by_label):
    """For each label, plot a histogram of the confidence scores."""
    for label, obj_list in objects_by_label.items():
        scores = [obj.get("score", 0) for obj in obj_list]
        plt.figure(figsize=(6,4))
        plt.hist(scores, bins=20, color="blue", alpha=0.7)
        plt.title(f"Confidence Distribution for '{label}'")
        plt.xlabel("Confidence Score")
        plt.ylabel("Frequency")
        plt.show()

# ==========================
# Step 4: Interactive selection of good samples.
# ==========================
def interactive_selection(sampled_detections, desired_count=50):
    """
    For each label, display the cropped image and let the user decide if the sample is a good example.
    Records the full projection image and bounding box if accepted.
    
    sampled_detections: dict mapping label -> list of candidate detections (sampled already)
    desired_count: number of accepted samples per class.
    
    Returns a dict mapping label -> list of accepted samples.
    """
    selected_samples = {}
    for label, det_list in sampled_detections.items():
        print(f"\n--- Reviewing samples for label: '{label}' ---")
        selected_samples[label] = []
        for obj in det_list:
            crop_path = obj.get("crop_path")
            if not crop_path or not os.path.exists(crop_path):
                print(f"  [Warning] Crop image not found: {crop_path}")
                continue

            # Load the cropped image.
            img = cv2.imread(crop_path)
            if img is None:
                print(f"  [Warning] Unable to load image at: {crop_path}")
                continue

            # Display the cropped image using matplotlib (convert BGR to RGB).
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6,4))
            plt.imshow(img_rgb)
            plt.title(f"Label: {label} | Confidence: {obj.get('score', 0):.2f}")
            plt.axis("off")
            plt.show()

            # Ask user whether to keep the sample.
            keep = input("Keep this sample? (y/n): ").strip().lower()
            if keep == "y":
                # Record the sample using the full projection image and bounding box.
                sample_entry = {
                    "projection_path": obj.get("projection_path"),
                    "bounding_box": obj.get("bounding_box"),
                    "label": label,
                    "score": obj.get("score")
                }
                selected_samples[label].append(sample_entry)
            if len(selected_samples[label]) >= desired_count:
                print(f"Collected {desired_count} samples for '{label}'.")
                break
    return selected_samples

# ==========================
# Step 5: Convert selected samples to a COCO-style dataset and copy images.
# ==========================
def get_image_dimensions(image_path):
    """Return (width, height) of the image. Uses default values if image cannot be loaded."""
    img = cv2.imread(image_path)
    if img is None:
        return 640, 480
    h, w = img.shape[:2]
    return w, h

def convert_to_coco(selected_samples, output_dir='data/coco_like_object_dataset', output_file='annotations.json'):
    """
    Convert the selected samples (dict mapping label -> list of samples) into a COCO-style JSON.
    For each image used (the full projection image), copy it to output_dir/images with a new name prefixed
    by an 8-digit increment (e.g., "00000001_imgname.jpg"). The "file_name" in the JSON is updated accordingly.
    """
    coco = {
        "info": {
            "year": 2025,
            "version": "1.0",
            "description": "Selected representative instances from detections",
            "contributor": "Your Name or Organization",
            "url": "",
            "date_created": "2025-02-26"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Create a mapping for categories.
    category_mapping = {}
    for label in selected_samples.keys():
        cid = len(category_mapping) + 1
        category_mapping[label] = cid
        coco["categories"].append({
            "id": cid,
            "name": label,
            "supercategory": ""
        })
    
    image_id = 1
    annotation_id = 1
    images_dict = {}  # Keyed by the original projection_path.
    
    # Prepare the output directories.
    os.makedirs(output_dir, exist_ok=True)
    images_output_dir = os.path.join(output_dir, "images")
    os.makedirs(images_output_dir, exist_ok=True)
    
    # We'll assign an 8-digit increment to each unique image.
    new_filename_mapping = {}
    counter = 1

    for label, sample_list in selected_samples.items():
        for sample in sample_list:
            proj_path = sample.get("projection_path")
            if not proj_path or not os.path.exists(proj_path):
                continue
            # If we haven't processed this image yet, add it.
            if proj_path not in images_dict:
                width, height = get_image_dimensions(proj_path)
                original_name = os.path.basename(proj_path)
                new_name = f"{counter:08d}_{original_name}"
                # Copy the image to the new dataset folder.
                destination_path = os.path.join(images_output_dir, new_name)
                try:
                    shutil.copy(proj_path, destination_path)
                except Exception as e:
                    print(f"Error copying {proj_path} to {destination_path}: {e}")
                    continue
                # Record the new filename.
                new_filename_mapping[proj_path] = new_name
                # Create the image entry for COCO.
                images_dict[proj_path] = {
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": new_name,
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "2025-02-26"
                }
                image_id += 1
                counter += 1
            
            # Convert the bounding box [xmin, ymin, xmax, ymax] to [x, y, width, height].
            bbox_info = sample.get("bounding_box", {})
            xmin = bbox_info.get("xmin", 0)
            ymin = bbox_info.get("ymin", 0)
            xmax = bbox_info.get("xmax", 0)
            ymax = bbox_info.get("ymax", 0)
            width_box = xmax - xmin
            height_box = ymax - ymin
            area = width_box * height_box

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": images_dict[proj_path]["id"],
                "category_id": category_mapping[label],
                "segmentation": [],  # Empty unless you have polygon data.
                "area": area,
                "bbox": [xmin, ymin, width_box, height_box],
                "iscrowd": 0
            })
            annotation_id += 1

    coco["images"] = list(images_dict.values())
    
    # Write the JSON annotations file.
    output_json_path = os.path.join(output_dir, output_file)
    with open(output_json_path, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"COCO JSON saved to {output_json_path}")

# ==========================
# Main workflow
# ==========================
def main():
    # Set the root directory of your processed files.
    processed_root = "/media/adrien/Space/Datasets/Overhead/processed/Grenoble"
    
    # 1. Load all object instances from metadata JSON files.
    all_objects = load_all_objects(processed_root)
    print(f"Loaded a total of {len(all_objects)} object instances.")
    
    # 2. Group objects by label and print counts.
    objects_by_label = group_by_label(all_objects)
    print_label_counts(objects_by_label)
    
    # 3. Plot confidence score distributions for each class.
    plot_confidence_distributions(objects_by_label)
    
    # 4. For each class, randomly sample up to 300 instances.
    nb_instances = 300
    sampled_detections = {}
    for label, det_list in objects_by_label.items():
        if len(det_list) > nb_instances:
            sampled_detections[label] = random.sample(det_list, nb_instances)
        else:
            sampled_detections[label] = det_list
        print(f"Sampled {len(sampled_detections[label])} instances for label '{label}'.")
    
    # 5. Interactive selection: manually review each cropped image sample.
    desired_count = 30
    selected_samples = interactive_selection(sampled_detections, desired_count=desired_count)
    
    # 6. Convert the selected samples into a COCO-style dataset,
    #    copy the full projection images into data/coco_like_object_dataset/images,
    #    and update the JSON file accordingly.
    convert_to_coco(selected_samples, output_dir='data/coco_like_object_dataset', output_file='annotations.json')

if __name__ == "__main__":
    main()
