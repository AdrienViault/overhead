import requests
import json
from collections import defaultdict

# Define API endpoint and parameters
base_url = "https://panoramax.openstreetmap.fr/api/collections"
bbox = "5.60,45.11,5.80,45.20"
datetime_range = "2015-01-01/2025-12-31"  # Define the date range

# Function to fetch all pages of sequences
def fetch_all_sequences():
    all_sequences = []
    next_url = f"{base_url}?bbox={bbox}&datetime={datetime_range}"
    page_count = 0
    
    while next_url:
        page_count += 1
        print(f"Fetching page {page_count}: {next_url}")
        response = requests.get(next_url)
        if response.status_code == 200:
            data = response.json()
            all_sequences.extend(data.get("collections", []))
            
            # Check if there's another page of results
            next_url = None
            for link in data.get("links", []):
                if link.get("rel") == "next":
                    next_url = link.get("href")
        else:
            print(f"Error fetching data: {response.status_code}")
            break
    
    print(f"Total pages fetched: {page_count}")
    return all_sequences

# Fetch sequences
sequences = fetch_all_sequences()

# Save the JSON response to a file
with open("panoramax_sequences.json", "w", encoding="utf-8") as json_file:
    json.dump({"collections": sequences}, json_file, indent=4, ensure_ascii=False)

print(f"Total sequences retrieved: {len(sequences)}")

# Parse JSON to count sequences per provider
provider_stats = defaultdict(int)

for collection in sequences:
    for provider in collection.get("providers", []):
        provider_name = provider.get("name", "Unknown Provider")
        provider_stats[provider_name] += 1

# Print results
print("Provider Statistics:")
for provider, count in provider_stats.items():
    print(f"Provider: {provider}, Sequences: {count}")
