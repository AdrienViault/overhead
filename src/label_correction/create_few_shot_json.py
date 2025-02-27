import os
import json
import random

# Paths to the train annotations file and the output file.
TRAIN_ANN_PATH = os.path.join("data", "coco_like_object_dataset", "train", "annotations", "annotations.json")
OUTPUT_PATH = os.path.join("data", "coco_like_object_dataset", "train", "annotations")

# Load the train annotations in COCO format.
with open(TRAIN_ANN_PATH, "r") as f:
    train_data = json.load(f)

train_images = train_data.get("images", [])
train_annotations = train_data.get("annotations", [])
categories = train_data.get("categories", [])
info = train_data.get("info", {})
licenses = train_data.get("licenses", [])

# Build a mapping from category_id to its annotations.
cat_to_annotations = {}
for ann in train_annotations:
    cat_id = ann["category_id"]
    cat_to_annotations.setdefault(cat_id, []).append(ann)

selected_annotations = []
# For each category, select exactly 10 annotations from different images.
nb_shots = 1

for cat in categories:
    cat_id = cat["id"]
    anns = cat_to_annotations.get(cat_id, [])
    # Create a mapping from image_id to annotation (using the first occurrence for each image).
    unique_anns = {}
    for ann in anns:
        img_id = ann["image_id"]
        if img_id not in unique_anns:
            unique_anns[img_id] = ann

    unique_list = list(unique_anns.values())
    if len(unique_list) < nb_shots:
        print(f"Warning: Category '{cat['name']}' (id {cat_id}) has only {len(unique_list)} unique images. Skipping.")
        continue

    # Randomly sample exactly nb_shots annotations.
    sampled = random.sample(unique_list, nb_shots)
    selected_annotations.extend(sampled)

# Determine the set of image ids referenced by the selected annotations.
selected_image_ids = set(ann["image_id"] for ann in selected_annotations)
selected_images = [img for img in train_images if img["id"] in selected_image_ids]

# Assemble the new COCO JSON.
shot_data = {
    "info": info,
    "licenses": licenses,
    "images": selected_images,
    "annotations": selected_annotations,
    "categories": categories
}

output_file_path = os.path.join(OUTPUT_PATH, f"{nb_shots}_shot_annotations.json")
# Save the new nb_shots-shot annotations JSON.
with open(output_file_path, "w") as f:
    json.dump(shot_data, f, indent=4)

print(f"{nb_shots}-shot JSON saved to {output_file_path}")