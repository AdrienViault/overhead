import requests
import json
import csv
from collections import defaultdict

# Define file paths
json_filename = "data/sequences_metadata/sequences_all.json"
selected_users_csv_filename = "data/sequences_metadata/selected_users.csv"
filtered_sequences_filename = "data/sequences_metadata/sequences_selected.json"

# Load selected users from CSV
selected_users = set()
with open(selected_users_csv_filename, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        user, download_flag = row
        if download_flag.strip().lower() == "yes":
            selected_users.add(user)

print(f"Users selected for download: {selected_users}")

# Load all sequences from JSON
with open(json_filename, "r", encoding="utf-8") as json_file:
    sequences_data = json.load(json_file)
    sequences = sequences_data.get("collections", [])

# Filter sequences for selected users
filtered_sequences = []
for collection in sequences:
    for provider in collection.get("providers", []):
        provider_name = provider.get("name", "Unknown Provider")
        if provider_name in selected_users:
            filtered_sequences.append(collection)
            break  # No need to check other providers for the same sequence

# Save filtered sequences to JSON
with open(filtered_sequences_filename, "w", encoding="utf-8") as json_file:
    json.dump({"collections": filtered_sequences}, json_file, indent=4, ensure_ascii=False)

print(f"Filtered sequences saved to {filtered_sequences_filename}")
print(f"Total sequences selected: {len(filtered_sequences)}")
