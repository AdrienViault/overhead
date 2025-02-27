import os
import json
import glob
from geopy.distance import distance

def convert_gps_to_decimal(gps_entry):
    """
    Convert a GPS coordinate entry (with degrees, minutes, seconds) to decimal degrees.
    """
    try:
        degrees = float(gps_entry.get("degrees", 0))
        minutes = float(gps_entry.get("minutes", 0))
        seconds = float(gps_entry.get("seconds", 0))
    except Exception as e:
        raise ValueError("Invalid GPS entry: " + str(e))
    decimal = degrees + minutes / 60 + seconds / 3600
    return decimal

def compute_destination(gps_metadata, bearing_deg, depth_m):
    """
    Compute a destination coordinate (decimal degrees) from a source GPS point given a bearing and distance.
    """
    try:
        lat_entry = gps_metadata.get("GPSLatitude", {})
        lon_entry = gps_metadata.get("GPSLongitude", {})
        src_lat = convert_gps_to_decimal(lat_entry)
        src_lon = convert_gps_to_decimal(lon_entry)
    except Exception as e:
        raise ValueError("Error converting GPS data: " + str(e))
    
    dest_point = distance(meters=depth_m).destination((src_lat, src_lon), bearing_deg)
    return dest_point.latitude, dest_point.longitude

def convert_decimal_to_dms(decimal_value, is_lat=True):
    """
    Convert a decimal degree value into a DMS dictionary and a reference.
    
    Args:
        decimal_value (float): The decimal degrees.
        is_lat (bool): True for latitude (N/S) or False for longitude (E/W).
    
    Returns:
        tuple: (ref, dms_dict) where ref is a string and dms_dict contains 'degrees', 'minutes', 'seconds'.
    """
    if is_lat:
        ref = "N" if decimal_value >= 0 else "S"
    else:
        ref = "E" if decimal_value >= 0 else "W"
    
    abs_val = abs(decimal_value)
    degrees = int(abs_val)
    minutes_full = (abs_val - degrees) * 60
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60
    return ref, {"degrees": float(degrees), "minutes": float(minutes), "seconds": float(seconds)}

def load_metadata_files(metadata_dir):
    """
    Load all JSON metadata files from the specified directory (and subdirectories).
    """
    metadata_files = glob.glob(os.path.join(metadata_dir, "**", "*_metadata.json"), recursive=True)
    metadata_list = []
    for filepath in metadata_files:
        try:
            with open(filepath, "r") as f:
                metadata = json.load(f)
                metadata["metadata_file"] = filepath  # Record source file if needed.
                metadata_list.append(metadata)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return metadata_list

def update_json_files_with_computed_position(metadata_dir):
    """
    For each metadata JSON file, compute each object's destination using the source GPS,
    the object's absolute angle, and depth. Then, add the computed position (in DMS format)
    into the object, and write the updated JSON back.
    """
    metadata_files = glob.glob(os.path.join(metadata_dir, "**", "*_metadata.json"), recursive=True)
    for filepath in metadata_files:
        try:
            with open(filepath, "r") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        # Get the source GPS metadata.
        source_info = meta.get("source", {})
        gps_metadata = source_info.get("GPS_metadata", {})

        # Process each detected object.
        objs = meta.get("objects", [])
        for obj in objs:
            try:
                angle = float(obj.get("absolute_angle", 0))
                depth = float(obj.get("depth", 0))
                # Compute destination position in decimal degrees.
                dest_lat, dest_lon = compute_destination(gps_metadata, angle, depth)
                # Convert computed decimal coordinates into DMS format.
                lat_ref, lat_dms = convert_decimal_to_dms(dest_lat, is_lat=True)
                lon_ref, lon_dms = convert_decimal_to_dms(dest_lon, is_lat=False)
                # Add the computed position in the same structure as the source.
                obj["computed_location"] = {
                    "GPSLatitudeRef": lat_ref,
                    "GPSLatitude": lat_dms,
                    "GPSLongitudeRef": lon_ref,
                    "GPSLongitude": lon_dms
                }
            except Exception as e:
                print(f"Error processing object in file {filepath}: {e}")
                continue

        # Save the updated metadata file.
        try:
            with open(filepath, "w") as f:
                json.dump(meta, f, indent=4)
            print(f"Updated file: {filepath}")
        except Exception as e:
            print(f"Error writing {filepath}: {e}")

if __name__ == "__main__":
    # Set this directory to where your JSON metadata files are stored.
    metadata_dir = "/media/adrien/Space/Datasets/Overhead/processed/"
    update_json_files_with_computed_position(metadata_dir)
