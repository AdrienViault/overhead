import requests
import json

# Define API endpoint and parameters
url = "https://panoramax.openstreetmap.fr/api/collections"
params = {
    "bbox": "5.60,45.11,5.80,45.20"
}

# Perform the API request
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    
    # Save the JSON response to a file
    with open("panoramax_sequences.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    
    print("Data successfully saved to panoramax_sequences.json")
else:
    print(f"Error: {response.status_code}")
