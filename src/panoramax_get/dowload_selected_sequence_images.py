import os
import csv
import subprocess
import json

# Define file paths
filtered_sequences_filename = "data/sequences_metadata/sequences_selected.json"
storage_path = "/media/adrien/Space/Datasets/Overhead/Grenoble"
panoramax_api_url = "https://panoramax.openstreetmap.fr"

# Ensure the storage directory exists
os.makedirs(storage_path, exist_ok=True)

# Log in to Panoramax CLI (User will be prompted for token if required)
print("Logging into Panoramax...")
subprocess.run(["panoramax_cli", "login", "--api-url", panoramax_api_url])
print("Login successful! Now starting the download process...")

# Load selected sequences from JSON
with open(filtered_sequences_filename, "r", encoding="utf-8") as json_file:
    sequences_data = json.load(json_file)
    sequences = sequences_data.get("collections", [])

print(f"Total sequences to download: {len(sequences)}")

# Download only specific sequences for selected users
for sequence in sequences:
    sequence_id = sequence.get("id")
    provider_name = sequence.get("providers", [{}])[0].get("name", "Unknown_Provider")
    print(f"\nStarting download for sequence {sequence_id} (User: {provider_name})")
    
    sequence_storage_path = os.path.join(storage_path, provider_name, sequence_id)
    os.makedirs(sequence_storage_path, exist_ok=True)
    
    command = [
        "panoramax_cli", "download",
        "--api-url", panoramax_api_url,
        "--collection", sequence_id,
        "--path", sequence_storage_path,
        "--quality", "hd"
    ]
    
    print(f"Executing command: {' '.join(command)}")
    
    # Execute command and stream output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Stream CLI output in real time
    for line in process.stdout:
        print(line.strip())
    for line in process.stderr:
        print(line.strip())
    
    process.wait()
    
    if process.returncode != 0:
        print(f"Error: Download failed for sequence {sequence_id}")
    else:
        print(f"Download completed for sequence {sequence_id}")

print("All selected sequences downloaded successfully!")
