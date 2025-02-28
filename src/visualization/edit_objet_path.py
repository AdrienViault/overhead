import os
import json
import glob

# Define the drive path prefix and the target local prefix.
DRIVE_PATH = "/media/adrien/Space/Datasets/Overhead/processed/"
# Since the JSON files already include "Grenoble" after the drive path,
# removing DRIVE_PATH will yield a path starting with "Grenoble/"
LOCAL_PREFIX = ""  # We simply remove DRIVE_PATH so that "Grenoble/..." remains.

def update_paths_in_object(obj):
    """
    For an object dictionary, update its path keys (projection_path, crop_path, depth_path)
    so that they contain only the local path (starting with "Grenoble/...") instead of the full drive path.
    """
    for key in ["projection_path", "crop_path", "depth_path"]:
        if key in obj and isinstance(obj[key], str):
            full_path = obj[key]
            if full_path.startswith(DRIVE_PATH):
                # Remove the drive path prefix.
                obj[key] = LOCAL_PREFIX + full_path[len(DRIVE_PATH):]
    return obj

def process_metadata_file(filepath):
    """
    Open a metadata JSON file, update the object paths and add a drive_path key,
    then save the updated file.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    # Add (or update) the drive_path key at the top level.
    data["drive_path"] = DRIVE_PATH

    # Process objects if available.
    if "objects" in data and isinstance(data["objects"], list):
        updated_objects = []
        for obj in data["objects"]:
            updated_objects.append(update_paths_in_object(obj))
        data["objects"] = updated_objects
    else:
        print(f"No objects found in {filepath}")

    # Write the updated metadata back to the same file.
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Updated {filepath}")
    except Exception as e:
        print(f"Error writing {filepath}: {e}")

def main():
    # Update this directory to where your JSON metadata files are stored.
    metadata_dir = "/media/adrien/Space/Datasets/Overhead/processed/"
    # Look for all JSON files ending with _metadata.json recursively.
    pattern = os.path.join(metadata_dir, "**", "*_metadata.json")
    metadata_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(metadata_files)} metadata file(s) to update.")

    for filepath in metadata_files:
        process_metadata_file(filepath)

if __name__ == "__main__":
    main()
