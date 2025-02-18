import math

def pixel_to_angle_perspective(image_width, pixel_position, fov=90):
    """
    Convert a horizontal pixel position to an angle relative to the left edge of the image,
    using a perspective (non-linear) projection model.
    
    Parameters:
        image_width (int or float): The width of the image in pixels.
        pixel_position (int or float): The horizontal pixel position (0-indexed) within the image.
        fov (float): The total horizontal field of view in degrees. Default is 90.
        
    Returns:
        float: The angle (in degrees) corresponding to the pixel position relative to the left edge.
               The left edge corresponds to 0°, and the right edge to fov°.
    """
    if image_width <= 0:
        raise ValueError("Image width must be a positive number.")
    if not (0 <= pixel_position <= image_width):
        raise ValueError("Pixel position must be within the image width.")
    
    # Compute half width and effective focal length in pixels.
    half_width = image_width / 2.0
    # f is derived from the relation: half_width = f * tan(fov/2)
    f = half_width / math.tan(math.radians(fov / 2))
    
    # Calculate the angle relative to the center (in radians)
    delta = pixel_position - half_width
    angle_from_center_rad = math.atan(delta / f)
    angle_from_center_deg = math.degrees(angle_from_center_rad)
    
    # Convert angle relative to center to angle relative to left edge
    # (center corresponds to fov/2, so add fov/2 to the result)
    angle_from_left = angle_from_center_deg + (fov / 2)
    
    return angle_from_left

# Example usage:
if __name__ == "__main__":
    image_width = 1080  # image width in pixels
    pixel = 1000       # center pixel (should yield ~45° for a 90° FOV image)
    angle = pixel_to_angle_perspective(image_width, pixel, fov=90)
    print(f"At pixel {pixel} in an image {image_width} pixels wide, the horizontal angle is {angle:.2f}° relative to the left edge.")
