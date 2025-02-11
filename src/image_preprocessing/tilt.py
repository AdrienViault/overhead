import cv2
import numpy as np

def detect_tilt_angle(image, canny_thresh1=50, canny_thresh2=150, hough_thresh=100,
                      min_line_length=100, max_line_gap=10, angle_limit=45):
    """
    Detect the tilt angle of the image by computing the median angle of near-horizontal lines.
    
    Args:
        image (np.ndarray): Input image (BGR).
        canny_thresh1 (int): Lower threshold for Canny edge detection.
        canny_thresh2 (int): Upper threshold for Canny edge detection.
        hough_thresh (int): Threshold for the probabilistic Hough transform.
        min_line_length (int): Minimum length of a line to be detected.
        max_line_gap (int): Maximum allowed gap between line segments.
        angle_limit (float): Consider only lines with absolute angle less than this (in degrees).
        
    Returns:
        float: The median tilt angle (in degrees). Positive values mean a clockwise rotation is needed.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Optionally, blur the image to reduce noise.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using Canny.
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    
    # Use the Probabilistic Hough Transform to detect line segments.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_thresh,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    angles = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Compute the angle in degrees.
                angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
                # We consider only near-horizontal lines.
                if abs(angle) < angle_limit:
                    angles.append(angle)
    
    if len(angles) > 0:
        # Use the median angle to be robust to outliers.
        median_angle = np.median(angles)
    else:
        median_angle = 0.0
    
    return median_angle

def correct_tilt(image):
    """
    Detect and correct the tilt of the image.
    
    Returns:
        tuple: (rotated_image, detected_angle)
            - rotated_image: The tilt-corrected image.
            - detected_angle: The angle (in degrees) that was detected (and corrected).
    """
    angle = detect_tilt_angle(image)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # Create a rotation matrix that rotates the image by -angle to correct the tilt.
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

# Example usage:
if __name__ == '__main__':
    # Load an image (replace with your file path)
    img_path = 'data/images/test_images/reprojected/perspective_90_GSAC0346.JPG'
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    corrected_image, detected_angle = correct_tilt(image)
    print(f"Detected tilt angle: {detected_angle:.2f} degrees")

    # Save or display the corrected image.
    cv2.imwrite('data/images/test_images/reprojected/perspective_90_GSAC0346_tilt_corrected_image.jpg', corrected_image)
    cv2.imshow('Corrected Image', corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()