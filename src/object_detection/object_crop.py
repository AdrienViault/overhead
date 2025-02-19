import numpy as np

def crop_object_image(image: np.ndarray, box) -> np.ndarray:
    """
    Crops the input image to the region defined by the bounding box.

    Parameters:
        image (np.ndarray): The input image (H x W x channels or H x W).
        box: The bounding box defining the object. It can be either:
             - A tensor with a 'tolist' method in the format (xmin, ymin, xmax, ymax), or
             - A tuple/list in the format (xmin, ymin, xmax, ymax).

    Returns:
        np.ndarray: The cropped image.
    """
    # If the box is a tensor, convert it to a list using the provided convention.
    if hasattr(box, 'tolist'):
        xmin, ymin, xmax, ymax = map(int, box.tolist())
    else:
        xmin, ymin, xmax, ymax = map(int, box)
    
    # Clip boundaries to ensure they lie within the image dimensions.
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, image.shape[1])
    ymax = min(ymax, image.shape[0])
    
    # Crop the image directly using slicing.
    return image[ymin:ymax, xmin:xmax]

# Example usage:
if __name__ == "__main__":
    import cv2
    import torch

    # Load an example image (ensure the path is correct)
    image = cv2.imread("data/images/test_images/reprojected/GSAC0346_perspective_left.jpg")
    
    # Example bounding box as a tensor in (xmin, ymin, xmax, ymax) format
    box_tensor = torch.tensor([50, 100, 250, 350])
    
    # Crop the image using the box tensor
    cropped_img = crop_object_image(image, box_tensor)
    
    # Save the cropped image
    cv2.imwrite("data/images/test_images/reprojected/GSAC0346_perspective_left_cropped.jpg", cropped_img)
