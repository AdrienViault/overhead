import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Load processor and model
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Load your image (ensure it's in RGB format)
image_path = "data/images/test_images/reprojected/GSAC0346_perspective_right.jpg"
image = Image.open(image_path).convert("RGB")

# Define text labels to search for
text_labels = [[
    "a photo of a street lamp", 
    "a photo of an overhead utility power distribution line",
    "a photo of a n overhead tram power line",
    "a photo of a safety cone",
    "a photo of a Single-phase low-voltage pole",
    "a photo of a Three-phase low-voltage pole with neutral",
    "a photo of a Three-phase low-voltage pole without neutral",
    "a photo of a Three-phase medium-voltage pole",
    "a photo of a Three-phase medium-voltage pole with ground wire",
    "a photo of a Three-phase high-voltage transmission tower",
    "a photo of a Combined utility pole (power + telecom)",
    "a photo of a Pole-Mounted Transformers",
    "a photo of a Switchgear",
    "a photo of an underground Distribution Box",
    "a photo of a Remote Terminal Units",
    "a photo of a transformer",
    "a photo of a substation",
    "a photo of a secondary substation",
    "a photo of a busbar",
    "a photo of a transformer",
    "a photo of a surge arrester",
    "a photo of a grounding system",
    "a photo of a switchgear",
    ]]

# Prepare inputs and perform inference
inputs = processor(text=text_labels, images=image, return_tensors="pt")
outputs = model(**inputs)

# Set target image sizes (height, width) for post-processing
target_sizes = torch.tensor([(image.height, image.width)])

# Post-process outputs to obtain boxes, scores, and text labels in Pascal VOC format
results = processor.post_process_grounded_object_detection(
    outputs=outputs, 
    target_sizes=target_sizes, 
    threshold=0.1, 
    text_labels=text_labels
)

# Retrieve predictions for the first image
result = results[0]
boxes, scores, detected_labels = result["boxes"], result["scores"], result["text_labels"]

# Print detected objects with their confidence scores and bounding boxes
for box, score, label in zip(boxes, scores, detected_labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {label} with confidence {round(score.item(), 3)} at location {box}")

# Visualization: draw bounding boxes on the image
def visualize_detections(image, boxes, scores, labels):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Define a list of colors for different labels (or use a color map)
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # Unpack and scale box coordinates
        xmin, ymin, xmax, ymax = box.tolist()
        width, height = image.size
        
        # Create a rectangle patch (convert normalized coordinates if necessary)
        rect = patches.Rectangle(
            (xmin, ymin), 
            xmax - xmin, 
            ymax - ymin, 
            linewidth=2, 
            edgecolor=colors[i % len(colors)], 
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin, f"{label} ({score:.2f})", 
                bbox=dict(facecolor=colors[i % len(colors)], alpha=0.5), 
                fontsize=12, 
                color='white')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Call the visualization function
visualize_detections(image, boxes, scores, detected_labels)
