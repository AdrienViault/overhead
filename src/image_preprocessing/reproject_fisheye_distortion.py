import numpy as np
import cv2

def equirectangular_to_perspective(e_img, fov=90, theta=0, phi=0, out_hw=(600, 800)):
    """
    Convert an equirectangular (360°) image into a perspective (rectilinear) view.

    The function computes a mapping for each pixel in the output image by:
      - Creating a grid of normalized direction vectors.
      - Rotating those vectors based on the desired view (given by theta and phi).
      - Converting the rotated vectors into spherical coordinates.
      - Mapping these spherical coordinates to pixel coordinates in the input image.
      - Remapping using cv2.remap.
      - Flipping vertically to correct for a bottom-up orientation.

    Args:
        e_img (np.ndarray): Input equirectangular image.
        fov (float, optional): Horizontal field of view (in degrees) for the perspective view.
                               Defaults to 90.
        theta (float, optional): Yaw angle (in degrees) – rotates the view horizontally.
                                 Defaults to 0.
        phi (float, optional): Pitch angle (in degrees) – rotates the view vertically.
                               Defaults to 0.
        out_hw (tuple, optional): Output image dimensions as (height, width). Defaults to (600, 800).

    Returns:
        np.ndarray: The generated perspective view image.
    """
    h_out, w_out = out_hw
    h_equi, w_equi, _ = e_img.shape

    # Convert field of view to radians and compute focal length.
    fov_rad = np.deg2rad(fov)
    f = (w_out / 2.0) / np.tan(fov_rad / 2.0)

    # Create a grid of pixel coordinates in the output image.
    x, y = np.meshgrid(np.arange(w_out), np.arange(h_out))
    x_c = (w_out - 1) / 2.0
    y_c = (h_out - 1) / 2.0

    # Normalize coordinates relative to the focal length.
    x_norm = (x - x_c) / f
    y_norm = (y - y_c) / f
    z = np.ones_like(x_norm)

    # Stack to create 3D direction vectors.
    dirs = np.stack((x_norm, y_norm, z), axis=-1)  # shape: (h_out, w_out, 3)
    dirs = dirs / np.linalg.norm(dirs, axis=2, keepdims=True)

    # Convert the center angles to radians.
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)

    # Build rotation matrices.
    # Yaw rotation (around the y-axis)
    R_yaw = np.array([
        [ np.cos(theta_rad), 0, np.sin(theta_rad)],
        [ 0, 1, 0],
        [-np.sin(theta_rad), 0, np.cos(theta_rad)]
    ])
    # Pitch rotation (around the x-axis)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(phi_rad), -np.sin(phi_rad)],
        [0, np.sin(phi_rad),  np.cos(phi_rad)]
    ])
    # Combined rotation (no roll is applied)
    R = R_pitch @ R_yaw

    # Apply the rotation to each direction vector.
    dirs_rot = dirs.reshape(-1, 3) @ R.T
    dirs_rot = dirs_rot.reshape(h_out, w_out, 3)

    # Convert rotated vectors to spherical coordinates.
    x_r = dirs_rot[..., 0]
    y_r = dirs_rot[..., 1]
    z_r = dirs_rot[..., 2]
    lon = np.arctan2(x_r, z_r)  # range: [-pi, pi]
    lat = np.arcsin(y_r)        # range: [-pi/2, pi/2]

    # Map spherical coordinates to equirectangular pixel coordinates.
    u = (lon + np.pi) / (2 * np.pi) * w_equi
    v = (np.pi/2 - lat) / np.pi * h_equi

    # Create maps for remapping.
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    persp = cv2.remap(e_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # Flip vertically so that the image orientation is correct.
    persp = cv2.flip(persp, 0)
    return persp
