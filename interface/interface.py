import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Load the pre-trained EfficientNet model
model_path = "runs/train/best.pt"
# Create model architecture first
model = efficientnet_b3(pretrained=False)

# Modify the classifier layer to match the expected number of classes (1481)
num_classes = 1481  # Based on the error message
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, num_classes)

# Load state dictionary
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()

# Define image transformations for EfficientNet
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load class names from folders in the data directory
class_names = []
data_dir = "data"
try:
    # Get all folders in the data directory
    class_names = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]
    # Sort to ensure consistent ordering
    class_names.sort()
    print(f"Loaded {len(class_names)} class names from data directory")
except FileNotFoundError:
    # If data directory doesn't exist, use dummy names
    class_names = [f"class_{i}" for i in range(num_classes)]
    print("Data directory not found, using dummy class names")


def predict_image(input_image):
    """Process the input image with EfficientNet model and return results"""
    # Convert numpy array to PIL Image
    if input_image is None:
        return None, "No image provided"

    img = Image.fromarray(input_image)

    # Preprocess the image
    img_tensor = transform(img).unsqueeze(0)

    # Run prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get top 5 predictions
    top_probs, top_indices = torch.topk(probabilities, 5)

    # Extract classification results
    classifications = []
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        class_idx = idx.item()
        class_name = (
            class_names[class_idx]
            if class_idx < len(class_names)
            else f"class_{class_idx}"
        )
        score = prob.item()
        classifications.append(f"{class_name}: {score:.2f}")

    # Join all classifications into a single text
    classification_text = "\n".join(classifications)

    # Create a copy of the image to draw on
    result_img = img.copy()
    draw = ImageDraw.Draw(result_img)

    # Calculate adaptive font size based on image dimensions - MUCH larger now
    width, height = img.size
    font_size = max(36, int(min(width, height) / 4))  # Greatly increased font size
    margin = int(font_size / 2)

    # Create a font object with the calculated size
    try:
        # Try to load a system font
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            # Try another common font
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            # Fall back to default font
            font = ImageFont.load_default()
            print(
                f"Warning: Using default font, may appear small. Calculated size was {font_size}"
            )

    # Only draw the top prediction (first item in classifications)
    if classifications:
        top_prediction = classifications[0]
        text_y = margin

        # Draw text outline for better visibility
        for offset_x, offset_y in [(-3, -3), (-3, 3), (3, -3), (3, 3)]:
            draw.text(
                (margin + offset_x, text_y + offset_y),
                top_prediction,
                fill=(0, 0, 0),
                font=font,
            )

        # Draw white text
        draw.text((margin, text_y), top_prediction, fill=(255, 255, 255), font=font)

    # Convert back to numpy array for Gradio
    result_img_array = np.array(result_img)

    return result_img_array, classification_text


# Create Gradio interface
with gr.Blocks(title="Character Detection") as interface:
    gr.Markdown("# Character Detection and Classification")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload Image")
            submit_btn = gr.Button("Detect")

        with gr.Column():
            output_image = gr.Image(label="Detection Results")
            output_text = gr.Textbox(label="Classifications")

    submit_btn.click(
        fn=predict_image, inputs=input_image, outputs=[output_image, output_text]
    )

# Launch the interface when script is run directly
if __name__ == "__main__":
    interface.launch(share=True)
