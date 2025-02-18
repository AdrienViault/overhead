import numpy as np
import cv2
import argparse


def equirectangular_to_perspective(equi_img, out_width, out_height, fov_deg, theta_deg, phi_deg):
    """
    Re-project an equirectangular image into a perspective view.
    
    Parameters:
      equi_img : The input equirectangular image (assumed to cover 360° x 180°).
      out_width: Output perspective image width in pixels.
      out_height: Output perspective image height in pixels.
      fov_deg  : The horizontal field-of-view (in degrees) of the perspective view.
      theta_deg: Yaw angle (rotation about the vertical axis) in degrees.
      phi_deg  : Pitch angle (rotation about the horizontal axis) in degrees.
      
    Returns:
      A perspective-projected image (as a NumPy array).
    """
    # Convert degrees to radians.
    fov = np.deg2rad(fov_deg)
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    
    # Compute the focal length from the horizontal FOV.
    # Using the pinhole camera model: f = (out_width/2) / tan(fov/2)
    f = (out_width / 2.0) / np.tan(fov / 2.0)
    
    # Define the center of the output image.
    center_x = out_width / 2.0
    center_y = out_height / 2.0
    
    # Create a grid of pixel coordinates in the output image.
    # i: vertical coordinate (0 at top to out_height-1 at bottom)
    # j: horizontal coordinate (0 at left to out_width-1 at right)
    j, i = np.meshgrid(np.arange(out_width), np.arange(out_height))
    
    # Convert pixel coordinates to camera plane coordinates.
    # The x coordinate is computed as (j - center_x) / f.
    # The y coordinate is computed as (center_y - i) / f so that “up” in the scene 
    # corresponds to the top of the image.
    x = (j - center_x) / f
    y = (center_y - i) / f
    z = np.ones_like(x)
    
    # Stack into direction vectors (each pixel now has a 3D direction).
    dirs = np.stack((x, y, z), axis=2)  # shape: (out_height, out_width, 3)
    
    # (Optional) Normalize the direction vectors so that each has unit length.
    norm = np.linalg.norm(dirs, axis=2, keepdims=True)
    dirs_norm = dirs / norm

    # --- Build the rotation matrix ---
    # Yaw (theta) rotates about the y-axis.
    R_yaw = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    # Pitch (phi) rotates about the x-axis.
    R_pitch = np.array([
        [1, 0,           0          ],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi),  np.cos(phi)]
    ])
    # Combined rotation: first apply pitch then yaw
    R =  R_yaw @ R_pitch

    # Rotate every direction vector.
    # Reshape the (H x W x 3) array into a list of 3D vectors, apply R,
    # and then reshape back to (H x W x 3).
    dirs_rot = dirs_norm.reshape((-1, 3)) @ R.T
    dirs_rot = dirs_rot.reshape((out_height, out_width, 3))
    
    # --- Convert the rotated directions to spherical coordinates ---
    # Spherical coordinates:
    #   - Longitude (lon) is computed from the x and z components.
    #   - Latitude (lat) is computed from the y component.
    dx = dirs_rot[:, :, 0]
    dy = dirs_rot[:, :, 1]
    dz = dirs_rot[:, :, 2]
    lon = np.arctan2(dx, dz)
    lat = np.arcsin(dy)
    
    # --- Map the spherical coordinates to pixel coordinates in the equirectangular image ---
    equi_h, equi_w = equi_img.shape[:2]
    # The horizontal coordinate (u) is proportional to (lon + π) over a 2π range.
    u = (lon + np.pi) / (2 * np.pi) * equi_w
    # The vertical coordinate (v) is proportional to (π/2 - lat) over a π range.
    v = (np.pi/2 - lat) / np.pi * equi_h
    
    # cv2.remap requires map_x and map_y to be of type float32.
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    
    # Remap the pixels from the equirectangular image to produce the perspective view.
    perspective = cv2.remap(equi_img, map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
    return perspective

def crop_bottom(
        image,
        height_in,
        keep_top_crop_factor,
        ):
    """
    Crops the bottom third of the given image.
    
    Parameters:
      image: A NumPy array representing the image.
      
    Returns:
      A new image (NumPy array) with the bottom third removed.
    """
    # Compute the new height, which is the top two-thirds of the original.
    new_height = int(height_in * keep_top_crop_factor)
    
    # Return the cropped image.
    return image[:new_height]


def project_equirectangular_left_right(
        equi_img, 
        out_width, 
        horizontal_fov_deg,
        vertical_fov_deg,
        keep_top_crop_factor=1.0
        ):
    
    # Compute the focal length from the horizontal FOV.
    # f = (out_width/2) / tan(horizontal_fov/2)
    f = (out_width / 2.0) / np.tan(np.deg2rad(horizontal_fov_deg / 2.0))
    
    # Compute the output height from the desired vertical FOV.
    # out_height = 2 * f * tan(vertical_fov/2)

    out_height = int(2 * f * np.tan(np.deg2rad(vertical_fov_deg / 2.0)))
    print("Output perspective image dimensions: {}x{}".format(out_width, out_height))
    
    # angle_away_from_full_front = 25
    # positive_yaw = angle_away_from_full_front + horizontal_fov_deg/2
    centering_side_yaw = 90
    # -------------------------
    # 3. Define the yaw angles to cover 360° horizontally.
    # -------------------------
    # For a horizontal FOV of 90°, four images at yaw angles 0°, 90°, 180°, and 270°
    # will cover the full 360°.
    yaw_angles = [-centering_side_yaw, +centering_side_yaw]
    pitch_deg = 0  # slight vertical tilt (centered over the horizon). Adjust if desired.
    
    # -------------------------
    # 4. Loop over each yaw angle and generate the perspective image.
    # -------------------------
    persp_images = []
    for yaw in yaw_angles:
        persp_img = equirectangular_to_perspective(
            equi_img,
            out_width, 
            out_height,
            horizontal_fov_deg,
            theta_deg=yaw,
            phi_deg=pitch_deg
            )
        persp_img_cropped = crop_bottom(
            persp_img, 
            out_height,
            keep_top_crop_factor,
            )
        persp_images.append(persp_img_cropped)
    return persp_images


def main_equirectangular_to_perspective():
    # -------------------------
    # 1. Load the equirectangular image.
    # -------------------------
    equi_img_path = "data/images/test_images/"
    equi_img_name = 'GSAC0346'
    equi_img_extension = '.JPG'
    equi_img = cv2.imread(equi_img_path+equi_img_name+equi_img_extension)

    if equi_img is None:
        print("Error: Could not load the equirectangular image!")
        return

    # -------------------------
    # 2. Set parameters for the perspective views.
    # -------------------------
    out_width = 1080                  # Width of each perspective image in pixels.
    horizontal_fov_deg = 90          # Horizontal FOV for each perspective image.
    vertical_fov_deg = 140            # Vertical FOV (ignoring the very top and bottom).
    keep_top_crop_factor = 2/3

    persp_images = project_equirectangular_left_right(
        equi_img, 
        out_width, 
        horizontal_fov_deg,
        vertical_fov_deg,
        keep_top_crop_factor
        )
    pic_suffixes = ['left', 'right']
    pic_names = []
    for pic_suffix in pic_suffixes:
        pic_names.append(f"{equi_img_name}_perspective_{pic_suffix}.jpg")

    out_dir_path = "data/images/test_images/reprojected/"
    for persp_image, pic_name in zip(persp_images, pic_names):    
        filename = out_dir_path + pic_name
        cv2.imwrite(filename, persp_image)
        print(f"Saved perspective image for {pic_name} side as {filename}")


def main_equirectangular_to_perspective_square():
    # -------------------------
    # 1. Load the equirectangular image.
    # -------------------------
    equi_img_path = "data/images/test_images/"
    equi_img_name = 'GSAC0346'
    equi_img_extension = '.JPG'
    equi_img = cv2.imread(equi_img_path+equi_img_name+equi_img_extension)

    if equi_img is None:
        print("Error: Could not load the equirectangular image!")
        return

    # -------------------------
    # 2. Set parameters for the perspective views.
    # -------------------------
    out_width = 1080                  # Width of each perspective image in pixels.
    horizontal_fov_deg = 90          # Horizontal FOV for each perspective image.
    vertical_fov_deg = 90            # Vertical FOV (ignoring the very top and bottom).
    keep_top_crop_factor =1.0

    persp_images = project_equirectangular_left_right(
        equi_img, 
        out_width, 
        horizontal_fov_deg,
        vertical_fov_deg,
        keep_top_crop_factor
        )
    pic_suffixes = ['levelsquare_left', 'levelsquare_right']
    pic_names = []
    for pic_suffix in pic_suffixes:
        pic_names.append(f"{equi_img_name}_perspective_{pic_suffix}.jpg")

    out_dir_path = "data/images/test_images/reprojected/"
    for persp_image, pic_name in zip(persp_images, pic_names):    
        filename = out_dir_path + pic_name
        cv2.imwrite(filename, persp_image)
        print(f"Saved perspective image for {pic_name} side as {filename}")



def cylindrical_projection(equi_img, vertical_fov_deg=120):
    """
    Convert an equirectangular image (360x180) to a cylindrical projection image with the specified vertical field-of-view.
    The output image will span 360° horizontally and vertical_fov_deg vertically.
    
    This version flips the vertical mapping so that the output is oriented correctly.
    
    Parameters:
        equi_img (numpy.ndarray): Input equirectangular image.
        vertical_fov_deg (float): Desired vertical field-of-view in degrees (default is 120).

    Returns:
        numpy.ndarray: The cylindrical projection image.
    """
    h, w, _ = equi_img.shape

    # The full equirectangular image covers 180° vertically.
    # Thus, output height is scaled proportionally:
    out_height = int(h * (vertical_fov_deg / 180))
    out_width = w  # full 360° horizontal

    # Prepare mapping arrays for cv2.remap.
    map_x = np.zeros((out_height, out_width), dtype=np.float32)
    map_y = np.zeros((out_height, out_width), dtype=np.float32)

    # Convert vertical FOV to radians.
    half_fov_rad = np.deg2rad(vertical_fov_deg / 2)

    # For each output pixel, calculate the corresponding spherical coordinates.
    # In the equirectangular image:
    #   - u (x coordinate) corresponds to longitude: phi in [-π, π]
    #   - v (y coordinate) corresponds to latitude: theta in [π/2, -π/2]
    #
    # For the cylindrical output:
    #   - x spans phi from -π to π.
    #   - y is computed so that the center of the output (y = out_height/2) corresponds to theta = 0.
    #     The mapping has been inverted so the top of the image represents positive theta.
    for y in range(out_height):
        # Invert the y coordinate to get the correct vertical orientation.
        theta = ((out_height/2 - y) / (out_height/2)) * half_fov_rad
        for x in range(out_width):
            # Compute phi from x coordinate: from -π to π.
            phi = (x / out_width) * 2 * np.pi - np.pi

            # Map spherical coordinates back to equirectangular coordinates.
            # u coordinate: (phi + π) / (2π) * image width.
            # v coordinate: (π/2 - theta) / π * image height.
            u = (phi + np.pi) / (2 * np.pi) * w
            v = (np.pi/2 - theta) / np.pi * h

            map_x[y, x] = u
            map_y[y, x] = v

    # Use cv2.remap to interpolate the pixels from the source image.
    # BORDER_WRAP ensures horizontal continuity.
    cylindrical_img = cv2.remap(equi_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return cylindrical_img
def main_cylindrical_projection():
    image_path = "data/images/test_images/GSAC0346.JPG"
    # Load the input image.
    img = cv2.imread(image_path , cv2.IMREAD_COLOR)
    vertical_angle_deg = 140

    # Generate the cylindrical projection.
    cyl_img = cylindrical_projection(img, vertical_fov_deg=vertical_angle_deg)

    # Save the resulting image.
    output_path = "data/images/test_images/reprojected/test.jpg"
    cv2.imwrite(output_path, cyl_img)
    print("Cylindrical image saved to", output_path)

if __name__ == '__main__':
    main_equirectangular_to_perspective()
    main_equirectangular_to_perspective_square()