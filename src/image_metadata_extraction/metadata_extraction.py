from PIL import Image, ExifTags
from PIL.TiffImagePlugin import IFDRational
def get_exif_data(image):
    exif_data = image._getexif() or {}
    exif = {}
    for tag, value in exif_data.items():
        decoded = ExifTags.TAGS.get(tag, tag)
        exif[decoded] = value
    return exif


def convert_tuples_to_dicts(name, tuple):
    #convert tuple of potential IFDRational to list of floats
    float_list = []

    for i in range(len(tuple)):
        if isinstance(tuple[i], IFDRational):
            float_list.append(float(tuple[i])) 
        else:
            float_list.append(tuple[i])

    if 'latitude' in name.lower():
        return {
            "degrees": float_list[0],
            "minutes": float_list[1],
            "seconds": float_list[2]
        }
    elif 'longitude' in name.lower():
        return {
            "degrees": float_list[0],
            "minutes": float_list[1],
            "seconds": float_list[2]
        }
    elif 'timestamp' in name.lower():
        return {
            "hour": float_list[0],
            "minute": float_list[1],
            "second": float_list[2]
        }
    else:
        raise ValueError(f"Unknown tuple name: {name}")
    

def get_gps_info_from_exif(exif):
    gps_info = exif.get("GPSInfo")
    if not gps_info:
        return None

    # Map the GPS tags to readable keys
    gps_data = {}
    for key in gps_info.keys():
        decoded_key = ExifTags.GPSTAGS.get(key, key)
        # if key is tuple, use convertion function
        if isinstance(gps_info[key], tuple):
            gps_data[decoded_key] = convert_tuples_to_dicts(
                decoded_key, 
                gps_info[key]
                )
        # if type is byte, convert it to string
        elif isinstance(gps_info[key], bytes):
            gps_data[decoded_key] = str(gps_info[key])
        # if type is IFDRational, convert it to float
        #verify if the key is json serializable
        elif isinstance(gps_info[key], IFDRational):
            gps_data[decoded_key] = float(gps_info[key])
        else:
            gps_data[decoded_key] = gps_info[key]
    return gps_data

def get_gps_info(image_path):
    image = Image.open(image_path)
    exif = get_exif_data(image)
    return get_gps_info_from_exif(exif)


verbose = False
if verbose:
    image_path = "data/images/test_images/GSAG2831.JPG"

    gps_data =get_gps_info(image_path)
    print("GPS Data:", gps_data)
    direction = gps_data['GPSImgDirection']
    GPS_latitude_ref = gps_data['GPSLatitudeRef']
    GPS_latitude_tuple = gps_data['GPSLatitude']
    GPS_longitude_ref = gps_data['GPSLongitudeRef']
    GPS_longitude_tuple = gps_data['GPSLongitude']
    GPS_date = gps_data['GPSDateStamp']
    GPS_speed = gps_data['GPSSpeed']

    print(direction)
    print(GPS_latitude_ref)
    print(GPS_latitude_tuple)
    print(GPS_longitude_ref)
    print(GPS_longitude_tuple)
    print(GPS_date)
    print(GPS_speed)

    print("direction is relatively to north. is full north, 90 is east, 180 is south, 270 is west")
