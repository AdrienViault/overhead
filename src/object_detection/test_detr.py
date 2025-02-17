import torch
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the image from a local path (update the path as needed)
image_path = "data/images/test_images/reprojected/perspective_-70.0deg.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Load the processor and model
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd")
model.to(device)  # Move model to GPU if available

# Process the image
inputs = image_processor(images=image, return_tensors="pt")
# Move inputs (which is a dict of tensors) to the appropriate device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process the outputs (target_sizes remains on CPU)
target_sizes = torch.tensor([(image.height, image.width)])
results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)

# Print the detected objects
for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score_val, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score_val:.2f} {box}")
