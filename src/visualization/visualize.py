pip install geopy folium matplotlib
```) and adjust the paths as needed.

Below is the complete script:

---

```python
#!/usr/bin/env python
import os
import json
import glob
from geopy.distance import distance
import folium
import matplotlib.pyplot as plt

def compute_destination(gps_metadata, angle_deg, depth_m):
    """
    Compute a destination coordinate from a source GPS point given a bearing (angle_deg) 
    and distance (depth_m in meters).

    Args:
        gps_metadata (dict): Should contain 'GPSLatitude' and 'GPSLongitude'.
        angle_deg (float): Bearing in degrees (absolute angle).
        depth_m (float): Distance in meters.

    Returns:
        (lat, lon): Destination latitude and longitude.
    """
    try:
        lat = float(gps_metadata.get("GPSLatitude"))
        lon = float(gps_metadata.get("GPSLongitude"))
    except (TypeError, ValueError):
        raise ValueError("Invalid GPS metadata: ensure GPSLatitude and GPSLongitude are provided.")

    # Compute the destination using geopy; note: distance.destination takes bearing in degrees.
    dest_point = distance(meters=depth_m).destination((lat, lon), angle_deg)
    return dest_point.latitude, dest_point.longitude

def load_metadata_files(metadata_dir):
    """
    Load all JSON metadata files from the specified directory (and subdirectories).

    Args:
        metadata_dir (str): Path to the folder containing the JSON files.

    Returns:
        List of metadata dictionaries.
    """
    metadata_files = glob.glob(os.path.join(metadata_dir, "**", "*_metadata.json"), recursive=True)
    metadata_list = []
    for filepath in metadata_files:
        try:
            with open(filepath, "r") as f:
                metadata = json.load(f)
                metadata["metadata_file"] = filepath  # record source file if needed
                metadata_list.append(metadata)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return metadata_list

def process_metadata(metadata_list):
    """
    For each metadata entry, compute the object location based on the source GPS,
    absolute angle, and object depth.
    
    Returns a list of objects enriched with their computed location.
    """
    objects_list = []
    for meta in metadata_list:
        # Extract source GPS metadata
        source_info = meta.get("source", {})
        gps_metadata = source_info.get("GPS_metadata", {})
        # Skip if GPS data not available
        try:
            src_lat = float(gps_metadata.get("GPSLatitude"))
            src_lon = float(gps_metadata.get("GPSLongitude"))
        except (TypeError, ValueError):
            print(f"Skipping metadata file {meta.get('metadata_file')} due to missing GPS data.")
            continue

        # Process each detected object
        for obj in meta.get("objects", []):
            try:
                angle = float(obj.get("absolute_angle", 0))
                depth = float(obj.get("depth", 0))
                # Compute destination using the provided bearing (absolute angle) and depth
                dest_lat, dest_lon = compute_destination(gps_metadata, angle, depth)
            except Exception as e:
                print(f"Error computing location for an object: {e}")
                continue

            # Enrich the object dictionary with computed location and source GPS for reference
            obj["computed_location"] = {"latitude": dest_lat, "longitude": dest_lon}
            obj["source_location"] = {"latitude": src_lat, "longitude": src_lon}
            objects_list.append(obj)
    return objects_list

def create_map(objects_list, output_map="objects_map.html"):
    """
    Create a folium map visualizing each object.
    Each marker popup includes the detection confidence score and, if available, the path to the cropped image.
    """
    if not objects_list:
        print("No objects to map.")
        return

    # Compute average location to center the map
    avg_lat = sum(obj["computed_location"]["latitude"] for obj in objects_list) / len(objects_list)
    avg_lon = sum(obj["computed_location"]["longitude"] for obj in objects_list) / len(objects_list)
    fmap = folium.Map(location=[avg_lat, avg_lon], zoom_start=14)

    for obj in objects_list:
        lat = obj["computed_location"]["latitude"]
        lon = obj["computed_location"]["longitude"]
        score = obj.get("score", 0)
        crop_path = obj.get("crop_path", "")
        # Build a simple HTML popup
        popup_html = f"<b>Confidence:</b> {score:.2f}"
        if crop_path and os.path.exists(crop_path):
            # The image is shown as an HTML image tag; adjust width as desired.
            popup_html += f"<br><img src='{crop_path}' width='100'/>"
        folium.Marker([lat, lon], popup=folium.Popup(popup_html, max_width=250)).add_to(fmap)

    fmap.save(output_map)
    print(f"Map saved to {output_map}")

def plot_confidence_distribution(objects_list, output_hist="confidence_distribution.png"):
    """
    Plot and save a histogram of object detection confidence scores.
    """
    scores = [obj.get("score", 0) for obj in objects_list if obj.get("score") is not None]
    if not scores:
        print("No confidence scores available for plotting.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Object Detection Confidence Scores")
    plt.savefig(output_hist)
    plt.show()
    print(f"Confidence score distribution saved as {output_hist}")

def main():
    # Update this directory to where your JSON metadata files are stored.
    metadata_dir = "/media/adrien/Space/Datasets/Overhead/processed/"
    metadata_list = load_metadata_files(metadata_dir)
    print(f"Loaded {len(metadata_list)} metadata file(s).")
    
    objects_list = process_metadata(metadata_list)
    print(f"Processed {len(objects_list)} detected object(s).")
    
    # Create a map with markers for each detected object.
    create_map(objects_list)
    
    # Plot and show the distribution of detection confidence scores.
    plot_confidence_distribution(objects_list)

if __name__ == "__main__":
    main()
