import torch
import matplotlib.pyplot as plt
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
import os

# Define the paths to the checkpoint and configuration file

sam2_checkpoint = "//home/adrien/Documents/Dev/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "//home/adrien/Documents/Dev/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" 

device = "cuda" if torch.cuda.is_available() else "cpu"

# Build the SAM2 model
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

# Initialize the automatic mask generator
mask_generator = SAM2AutomaticMaskGenerator(sam2)



def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



# Example usage
image_path = "data/images/test_images/reprojected/perspective_-80.0deg.jpg"  
image = preprocess_image(image_path)

# Generate masks
masks = mask_generator.generate(image)

# Print the number of masks and keys in the first mask
print(f"Number of masks generated: {len(masks)}")
print("Keys in the first mask:", masks[0].keys())


plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 


# Save the figure for this parameter group
output_folder = "/home/adrien/Documents/Dev/overhead/data/images/test_images/segmentation/"
output_filename = f"segmentation_autom.png"
output_path = os.path.join(output_folder, output_filename)
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
print(f"Saved: {output_path}")