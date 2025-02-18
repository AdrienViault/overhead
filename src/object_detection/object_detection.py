import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_detections(image, boxes, scores, labels):
    """
    Draws bounding boxes and labels on the image.
    
    Args:
        image (PIL.Image): The input image.
        boxes (Tensor): Bounding boxes (in pixel coordinates).
        scores (Tensor): Confidence scores for each detection.
        labels (List[str]): Detected text labels.
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Colors for the boxes (cycled through if there are many detections)
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        xmin, ymin, xmax, ymax = box.tolist()
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor=colors[i % len(colors)],
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            xmin, ymin,
            f"{label} ({score:.2f})",
            bbox=dict(facecolor=colors[i % len(colors)], alpha=0.5),
            fontsize=12,
            color='white'
        )
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def detect_and_visualize(model, processor, image, text_labels, threshold=0.1, visualize=True):
    """
    Performs object detection using a preloaded model and processor,
    prints the detections, and optionally visualizes the results.
    
    Args:
        model: The object detection model (assumed to be on GPU).
        processor: The corresponding processor.
        image (PIL.Image): The input image (in RGB).
        text_labels (list): A list (or list of lists) of text queries.
        threshold (float): The detection threshold for post-processing.
        visualize (bool): Whether to draw the detections on the image.
        
    Returns:
        boxes (Tensor): Bounding boxes of detections.
        scores (Tensor): Confidence scores.
        detected_labels (List[str]): Detected labels.
    """
    # Prepare inputs for the model
    inputs = processor(text=text_labels, images=image, return_tensors="pt")
    
    # Move inputs to GPU if the model is on GPU
    if next(model.parameters()).is_cuda:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Perform inference
    outputs = model(**inputs)
    
    # Set target image sizes (height, width)
    target_sizes = torch.tensor([(image.height, image.width)])
    if next(model.parameters()).is_cuda:
        target_sizes = target_sizes.to(model.device)
    
    # Post-process outputs to obtain boxes, scores, and labels
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, 
        target_sizes=target_sizes, 
        threshold=threshold, 
        text_labels=text_labels
    )
    
    # Retrieve predictions for the first (and only) image
    result = results[0]
    boxes, scores, detected_labels = result["boxes"], result["scores"], result["text_labels"]
    
    # Print detected objects with their confidence scores and bounding boxes
    for box, score, label in zip(boxes, scores, detected_labels):
        box_coords = [round(i, 2) for i in box.tolist()]
        print(f"Detected {label} with confidence {round(score.item(), 3)} at location {box_coords}")
    
    # Optionally visualize the detections
    if visualize:
        visualize_detections(image, boxes, scores, detected_labels)
    
    return boxes, scores, detected_labels

# Example usage:
if __name__ == "__main__":
    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    # Load processor and model (assume the model is then moved to GPU)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    model.to("cuda")  # Move the model to GPU if available

    # Load an image (ensure it's in RGB format)
    image_path = "data/images/test_images/reprojected/GSAC0346_perspective_right.jpg"
    image = Image.open(image_path).convert("RGB")

    # Define text labels to search for
    text_labels = [[
        "a photo of a street lamp", 
        "a photo of an overhead line",
        "a photo of a safety cone",
        "a photo of an overhead distribution line",
        "a photo of an utility pole",
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

    # Call the function
    detect_and_visualize(model, processor, image, text_labels, threshold=0.1, visualize=True)
