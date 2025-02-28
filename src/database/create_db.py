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
    and builds a list of marker dictionaries.
    It computes decimal latitude and longitude from the DMS values in computed_location.
    Additionally, it extracts bounding box information (if present) under the key "bbox".
    """
    markers = []
    metadata_files = glob.glob(os.path.join(metadata_dir, "**", "*_metadata.json"), recursive=True)
    print(f"Found {len(metadata_files)} metadata files in {metadata_dir}.")

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
            
            # Update the computed location with decimal values.
            obj.setdefault("computed_location", {})["decimal_lat"] = decimal_lat
            obj["computed_location"]["decimal_lon"] = decimal_lon

            # Extract bounding box if available.
            # Expected structure: { "xmin": value, "ymin": value, "xmax": value, "ymax": value }
            bbox = obj.get("bbox")
            if bbox:
                try:
                    xmin = float(bbox.get("xmin"))
                    ymin = float(bbox.get("ymin"))
                    xmax = float(bbox.get("xmax"))
                    ymax = float(bbox.get("ymax"))
                    # Create a WKT polygon for the bounding box.
                    bbox_wkt = f'POLYGON(({xmin} {ymin}, {xmax} {ymin}, {xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}))'
                    obj["bbox_wkt"] = bbox_wkt
                except Exception as e:
                    print(f"Error processing bounding box in {filepath}: {e}")
                    obj["bbox_wkt"] = None
            else:
                obj["bbox_wkt"] = None

            markers.append(obj)
    
    return markers

# Define the directory where your JSON metadata files are stored.
metadata_dir = "/media/adrien/Space/Datasets/Overhead/processed/"
markers = load_markers_from_metadata(metadata_dir)
print(f"Loaded {len(markers)} markers from metadata.")

# Connect to PostgreSQL database.
try:
    conn = psycopg2.connect(dbname="geodb", user="postgres", password="D^A@cn5W", host="localhost")
    cur = conn.cursor()
    print("Connected to the database.")
except Exception as e:
    print("Database connection error:", e)
    exit(1)

# Drop the existing markers table if it exists.
drop_table_query = "DROP TABLE IF EXISTS markers;"
print("Dropping the existing markers table if it exists...")
cur.execute(drop_table_query)
conn.commit()

# Create the markers table with an extra column for bounding box geometry.
create_table_query = """
CREATE TABLE markers (
    id SERIAL PRIMARY KEY,
    label TEXT,
    score REAL,
    geom geometry(Point,4326),
    bounding_box geometry(Polygon,4326),
    projection_path TEXT,
    detection_path TEXT,
    depth_path TEXT
);
"""
print("Creating the markers table with the bounding_box column...")
cur.execute(create_table_query)
conn.commit()
print("Table created successfully.")

# Prepare a list of records for insertion.
records = []
for marker in markers:
    try:
        decimal_lat = marker["computed_location"]["decimal_lat"]
        decimal_lon = marker["computed_location"]["decimal_lon"]
    except KeyError:
        print("Skipping marker with missing computed_location:", marker)
        continue

    # Use WKT format for the point geometry.
    point_wkt = f'SRID=4326;POINT({decimal_lon} {decimal_lat})'
    
    # Get bounding box WKT if available; it should be in valid POLYGON WKT format.
    bbox_wkt = marker.get("bbox_wkt")
    if bbox_wkt:
        bbox_wkt = f"SRID=4326;{bbox_wkt}"
    else:
        bbox_wkt = None

    record = (
        marker.get("label", "Unknown"),
        marker.get("score", 0.0),
        point_wkt,
        bbox_wkt,
        marker.get("projection_path", ""),
        marker.get("detection_path", marker.get("projection_path", "")),
        marker.get("depth_path", "")
    )
    records.append(record)

print(f"Prepared {len(records)} records for insertion.")

# Prepare the INSERT query.
insert_query = """
INSERT INTO markers (label, score, geom, bounding_box, projection_path, detection_path, depth_path)
VALUES %s;
"""

# Insert the records using execute_values for efficiency.
try:
    print("Inserting records into the database...")
    execute_values(cur, insert_query, records)
    conn.commit()
    print(f"Inserted {len(records)} markers into the database.")
except Exception as e:
    conn.rollback()
    print("Error inserting markers:", e)

cur.close()
conn.close()
print("Database connection closed.")
