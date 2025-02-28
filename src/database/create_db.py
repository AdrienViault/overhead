import os
import json
import glob
import psycopg2
from psycopg2.extras import execute_values

def convert_dms_to_decimal(dms, ref):
    """
    Convert a DMS dictionary to a decimal degree value.
    dms should be a dict with keys "degrees", "minutes", "seconds".
    """
    degrees = float(dms.get("degrees", 0))
    minutes = float(dms.get("minutes", 0))
    seconds = float(dms.get("seconds", 0))
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def load_markers_from_metadata(metadata_dir):
    """
    Scans through all JSON metadata files in metadata_dir (and subdirectories)
    and builds a list of marker dictionaries. For each detected object, it computes
    decimal latitude and longitude from the DMS computed_location.
    """
    markers = []
    metadata_files = glob.glob(os.path.join(metadata_dir, "**", "*_metadata.json"), recursive=True)
    
    for filepath in metadata_files:
        try:
            with open(filepath, "r") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
        
        # Each metadata file may contain several detected objects.
        for obj in meta.get("objects", []):
            comp = obj.get("computed_location", {})
            # Get DMS and reference for latitude and longitude.
            lat_dms = comp.get("GPSLatitude", {})
            lon_dms = comp.get("GPSLongitude", {})
            lat_ref = comp.get("GPSLatitudeRef", "N")
            lon_ref = comp.get("GPSLongitudeRef", "E")
            try:
                decimal_lat = convert_dms_to_decimal(lat_dms, lat_ref)
                decimal_lon = convert_dms_to_decimal(lon_dms, lon_ref)
            except Exception as e:
                print(f"Error converting DMS to decimal in {filepath}: {e}")
                continue
            
            # Store the computed decimals in the object for later use.
            obj.setdefault("computed_location", {})["decimal_lat"] = decimal_lat
            obj["computed_location"]["decimal_lon"] = decimal_lon
            markers.append(obj)
    
    return markers

# Define the directory where your JSON metadata files are stored.
metadata_dir = "/media/adrien/Space/Datasets/Overhead/processed/"
markers = load_markers_from_metadata(metadata_dir)
print(f"Loaded {len(markers)} markers from metadata.")

# Connect to PostgreSQL database.
conn = psycopg2.connect(dbname="geodb", user="postgres", password="D^A@cn5W", host="localhost")
cur = conn.cursor()

# Create a list of records to insert.
records = []
for marker in markers:
    try:
        decimal_lat = marker["computed_location"]["decimal_lat"]
        decimal_lon = marker["computed_location"]["decimal_lon"]
    except KeyError:
        print("Skipping marker with missing computed_location:", marker)
        continue
    
    record = (
        marker.get("label", "Unknown"),
        marker.get("score", 0.0),
        # Use the well-known text (WKT) format for the point.
        f'SRID=4326;POINT({decimal_lon} {decimal_lat})',
        marker.get("projection_path", ""),
        marker.get("detection_path", marker.get("projection_path", "")),
        marker.get("depth_path", "")
    )
    records.append(record)

# Prepare the INSERT query.
query = """
INSERT INTO markers (label, score, geom, projection_path, detection_path, depth_path)
VALUES %s;
"""


#clear database
print("Clearing the markers table...")
cur.execute("TRUNCATE TABLE markers;")
#verify if the table is empty
cur.execute("SELECT * FROM markers;")
print(f"Number of rows in the table: {cur.rowcount}")

# Insert the marker records using execute_values for efficiency.
try:
    execute_values(cur, query, records)
    conn.commit()
    print(f"Inserted {len(records)} markers into the database.")
except Exception as e:
    conn.rollback()
    print("Error inserting markers:", e)

cur.close()
conn.close()
