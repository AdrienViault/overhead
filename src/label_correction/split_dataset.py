import os
import json
import random
import shutil

# Set base directories.
BASE_DIR = "data/coco_like_object_dataset"
ORIG_ANNOTATIONS_PATH = os.path.join(BASE_DIR, "annotations.json")
ORIG_IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Output directories for train and test.
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_ANN_DIR = os.path.join(TRAIN_DIR, "annotations")
TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, "images")
TEST_ANN_DIR = os.path.join(TEST_DIR, "annotations")
TEST_IMG_DIR = os.path.join(TEST_DIR, "images")

# Create output directories if they do not exist.
for d in [TRAIN_ANN_DIR, TRAIN_IMG_DIR, TEST_ANN_DIR, TEST_IMG_DIR]:
    os.makedirs(d, exist_ok=True)

# Load the original COCO annotations.
with open(ORIG_ANNOTATIONS_PATH, "r") as f:
    coco = json.load(f)

images = coco.get("images", [])
annotations = coco.get("annotations", [])
categories = coco.get("categories", [])

# Build a mapping from category_id to a list of annotations.
cat_to_annotations = {}
for cat in categories:
    cat_to_annotations[cat["id"]] = []

for ann in annotations:
    cat_id = ann["category_id"]
    cat_to_annotations[cat_id].append(ann)

# For each category, randomly sample 30 annotations (if available).
# Then, from these, randomly select 20 for training and 10 for testing.
train_annotations = []
test_annotations = []

for cat in categories:
    cat_id = cat["id"]
    ann_list = cat_to_annotations.get(cat_id, [])
    if len(ann_list) < 30:
        print(f"Warning: Category '{cat['name']}' (id {cat_id}) has only {len(ann_list)} annotations.")
        # Option: use all available annotations (splitting proportionally)
        sampled = ann_list
        # Decide split proportionally (e.g. 2/3 for train, 1/3 for test)
        n_train = int(len(sampled) * 2 / 3)
        n_test = len(sampled) - n_train
    else:
        sampled = random.sample(ann_list, 30)
        n_train = 20
        n_test = 10

    # Randomly choose n_train for training.
    train_sample = random.sample(sampled, n_train)
    # The remaining (from the sampled 30) will be used for test.
    test_sample = [ann for ann in sampled if ann not in train_sample]
    # In case there are more than n_test remaining, sample exactly n_test.
    if len(test_sample) > n_test:
        test_sample = random.sample(test_sample, n_test)
    train_annotations.extend(train_sample)
    test_annotations.extend(test_sample)

# Now, collect the unique image ids referenced by the train and test annotations.
train_image_ids = set(ann["image_id"] for ann in train_annotations)
test_image_ids = set(ann["image_id"] for ann in test_annotations)

# Build new images lists for train and test.
train_images = [img for img in images if img["id"] in train_image_ids]
test_images = [img for img in images if img["id"] in test_image_ids]

# Create new COCO dictionaries.
train_coco = {
    "info": coco.get("info", {}),
    "licenses": coco.get("licenses", []),
    "images": train_images,
    "annotations": train_annotations,
    "categories": categories
}

test_coco = {
    "info": coco.get("info", {}),
    "licenses": coco.get("licenses", []),
    "images": test_images,
    "annotations": test_annotations,
    "categories": categories
}

# Save new annotations JSON files.
train_json_path = os.path.join(TRAIN_ANN_DIR, "annotations.json")
test_json_path = os.path.join(TEST_ANN_DIR, "annotations.json")

with open(train_json_path, "w") as f:
    json.dump(train_coco, f, indent=4)
print(f"Train annotations saved to {train_json_path}")

with open(test_json_path, "w") as f:
    json.dump(test_coco, f, indent=4)
print(f"Test annotations saved to {test_json_path}")

# Copy images for train and test sets.
def copy_images(image_list, src_dir, dst_dir):
    for img in image_list:
        filename = img["file_name"]
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Image file not found: {src_path}")

copy_images(train_images, ORIG_IMAGES_DIR, TRAIN_IMG_DIR)
copy_images(test_images, ORIG_IMAGES_DIR, TEST_IMG_DIR)

print("Image copying completed.")
