import cv2
import numpy as np

image_path = "data/images/test_images/reprojected/perspective_0deg.jpg"

# =============================================================================
# Function to load and preprocess the image.
# =============================================================================
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def crop_and_save_object(image, ann, output_path):
    """
    Given an image and a segmentation annotation (with key "segmentation" as a binary mask),
    apply the mask to keep only the object, crop the image to the smallest rectangle containing
    the object, and save the resulting image to a JPEG file.

    Parameters:
      image       : NumPy array representing the image (assumed to be in RGB format).
      ann         : Dictionary with at least a key 'segmentation' containing a binary mask.
      output_path : File path (including filename) where the JPEG output should be saved.
    """
    # Retrieve the mask from the annotation
    mask = ann['segmentation']
    
    # Ensure the mask is boolean
    mask_bool = mask.astype(bool) if mask.dtype != np.bool_ else mask
    
    # Find coordinates where the mask is True
    ys, xs = np.where(mask_bool)
    if len(ys) == 0 or len(xs) == 0:
        print("Warning: the mask does not contain any True values.")
        return
    
    # Compute the bounding box of the object
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    
    # Crop the image to the bounding box
    cropped_img = image[y1:y2+1, x1:x2+1].copy()
    
    # Also crop the mask to the bounding box
    cropped_mask = mask_bool[y1:y2+1, x1:x2+1]
    
    # If the image is colored (3 channels), replicate the mask for each channel
    if len(cropped_img.shape) == 3 and cropped_img.shape[2] == 3:
        cropped_mask_3 = np.stack([cropped_mask] * 3, axis=-1)
    else:
        cropped_mask_3 = cropped_mask
    
    # Apply the mask: set background pixels to black
    result = np.where(cropped_mask_3, cropped_img, 0)
    
    return result


# =============================================================================
# Example usage:
# =============================================================================
# Assume we have an image (in RGB) and a corresponding annotation "ann" from our segmentation.
# For example, let's say:
# image = preprocess_image(image_path)  # your image as RGB
# ann = masks_results['points_per_side'][32][0]  # taking the first object from a chosen parameter

# Define an output path:
# output_file = "cropped_object_0.jpg"
# crop_and_save_object(image, ann, output_file)

# Load the image
image = preprocess_image(image_path)
#todo : calculate image mask, apply it with function and show and save result
