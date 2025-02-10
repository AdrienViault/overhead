import requests
import json
import csv
import os

# Define file paths
filtered_sequences_filename = "data/sequences_metadata/sequences_selected.json"
storage_path = "/media/adrien/Space/Datasets/Overhead/Grenoble"

# Ensure the storage directory exists
os.makedirs(storage_path, exist_ok=True)

# Load selected sequences from JSON
with open(filtered_sequences_filename, "r", encoding="utf-8") as json_file:
    sequences_data = json.load(json_file)
    sequences = sequences_data.get("collections", [])

print(f"Total sequences to download: {len(sequences)}")

total_bytes_downloaded = 0
current_user = None

# Function to download images
def download_images(sequence):
    global total_bytes_downloaded, current_user
    sequence_id = sequence.get("id")
    provider_name = sequence.get("providers", [{}])[0].get("name", "Unknown_Provider")
    
    # Print when switching to a new user
    if provider_name != current_user:
        print(f"\nStarting downloads for user: {provider_name}")
        current_user = provider_name
    
    # Create a directory for this provider and sequence
    sequence_dir = os.path.join(storage_path, provider_name, sequence_id)
    os.makedirs(sequence_dir, exist_ok=True)
    print(f"  -> Downloading sequence {sequence_id}")
    
    # Find the link to images
    for link in sequence.get("links", []):
        if link.get("rel") == "items":
            items_url = link.get("href")
            response = requests.get(items_url)
            if response.status_code == 200:
                items_data = response.json()
                
                for item in items_data.get("features", []):
                    image_url = item.get("assets", {}).get("image", {}).get("href")
                    if image_url:
                        image_filename = os.path.join(sequence_dir, os.path.basename(image_url))
                        
                        # Download the image
                        img_response = requests.get(image_url, stream=True)
                        if img_response.status_code == 200:
                            with open(image_filename, "wb") as img_file:
                                for chunk in img_response.iter_content(1024):
                                    img_file.write(chunk)
                                    total_bytes_downloaded += len(chunk)
                            print(f"    -> Downloaded: {image_filename}")
                        else:
                            print(f"    !! Failed to download: {image_url}")
            else:
                print(f"    !! Failed to fetch images for sequence {sequence_id}")
    
    print(f"  -> Finished sequence {sequence_id}")

# Iterate over sequences and download images
for sequence in sequences:
    download_images(sequence)

# Convert bytes to gigabytes
total_gb = total_bytes_downloaded / (1024 ** 3)
print(f"\nTotal data downloaded: {total_gb:.2f} GB")
print("All images downloaded successfully!")
