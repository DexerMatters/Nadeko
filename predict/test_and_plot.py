import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Turn off interactive plotting - only save figures
plt.ioff()


def load_model(model_path, device="cpu"):
    """Load trained PyTorch model"""
    # Check if the model is a YOLO model
    if model_path.endswith(".pt") and "/yolo_" in model_path:
        try:
            from ultralytics import YOLO

            model = YOLO(model_path)
            print("Loaded YOLO model using ultralytics API")
            return model, True  # Return model and flag indicating it's a YOLO model
        except ImportError:
            print(
                "Could not import ultralytics. Falling back to generic model loading."
            )

    # Traditional PyTorch model loading (for non-YOLO models)
    torch.serialization.add_safe_globals(["ultralytics.nn.tasks.ClassificationModel"])
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle model structure (dict with 'model' key)
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            model = checkpoint["model"]
        else:
            # Try to find the model in common keys
            for key in ["model", "net", "state_dict"]:
                if key in checkpoint:
                    model = checkpoint[key]
                    break
            else:
                raise ValueError(
                    f"Could not find model in checkpoint keys: {checkpoint.keys()}"
                )
    else:
        model = checkpoint

    model.eval()
    model = model.float()
    return model, False  # Return model and flag indicating it's not a YOLO model


def get_training_tags():
    """Get available tags from training data directory"""
    data_dir = "./data"
    if not os.path.exists(data_dir):
        return set()
    return set(os.listdir(data_dir))


def load_test_data(test_dir="./data_test"):
    """Load test data filtered by available training tags"""
    training_tags = get_training_tags()
    test_data = []

    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found")
        return test_data

    # Filter test folders by training tags
    test_folders = [f for f in os.listdir(test_dir) if f in training_tags]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for tag in test_folders:
        tag_path = os.path.join(test_dir, tag)
        if os.path.isdir(tag_path):
            for img_file in os.listdir(tag_path):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(tag_path, img_file)
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img_tensor = transform(img)
                        test_data.append((img_tensor, tag, img_path))
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")

    return test_data


def predict_and_plot(model_path, device="cpu"):
    """Run predictions and plot results"""
    # Load model and data
    model, is_yolo = load_model(model_path, device)
    test_data = load_test_data()

    if not test_data:
        print("No test data found")
        return

    # Get class names from training data
    training_tags = sorted(list(get_training_tags()))

    predictions = []
    actual_labels = []
    image_paths = []

    print(f"Testing {len(test_data)} images...")

    if is_yolo:
        # YOLO prediction approach
        for _, true_tag, img_path in test_data:
            # YOLO models can predict directly from image path
            results = model(img_path, verbose=False)

            # Get classification results
            if hasattr(results[0], "probs") and results[0].probs is not None:
                probs = results[0].probs
                predicted_idx = probs.top1
                confidence = probs.top1conf.item()

                if predicted_idx < len(training_tags):
                    predicted_tag = training_tags[predicted_idx]
                else:
                    predicted_tag = "unknown"

                predictions.append(predicted_tag)
                actual_labels.append(true_tag)
                image_paths.append(img_path)
                print(
                    f"Actual: {true_tag}\nPredicted: {predicted_tag}, Confidence: {confidence:.4f}"
                )
            else:
                print(f"Warning: No probability output for {img_path}")
    else:
        # Traditional PyTorch prediction approach
        with torch.no_grad():
            for img_tensor, true_tag, img_path in test_data:
                img_batch = img_tensor.unsqueeze(0).to(device).float()

                # Get prediction
                outputs = model(img_batch)

                # Handle tuple output from models
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take first output (usually logits)

                probabilities = F.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()

                if predicted_idx < len(training_tags):
                    predicted_tag = training_tags[predicted_idx]
                else:
                    predicted_tag = "unknown"

                predictions.append(predicted_tag)
                actual_labels.append(true_tag)
                image_paths.append(img_path)
                print(
                    f"Actual:\t{true_tag}\nPredicted:\t{predicted_tag}\nAccuracy:\t{probabilities[0][predicted_idx].item():.4f}"
                )

    # Calculate accuracy
    correct = sum(1 for p, a in zip(predictions, actual_labels) if p == a)
    accuracy = correct / len(predictions) if predictions else 0
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(predictions)})")

    # Plot confusion matrix
    plot_confusion_matrix(actual_labels, predictions, training_tags)

    # Plot sample predictions
    plot_sample_predictions(image_paths[:12], actual_labels[:12], predictions[:12])


def plot_confusion_matrix(actual, predicted, class_names):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(actual, predicted, labels=class_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Ensure directory exists
    os.makedirs("predict", exist_ok=True)
    plt.savefig("predict/confusion_matrix.png")
    plt.close()  # Close the figure to free memory


def plot_sample_predictions(image_paths, actual_labels, predictions):
    """Plot sample predictions with images"""
    n_samples = min(12, len(image_paths))
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(n_samples):
        try:
            img = Image.open(image_paths[i])
            axes[i].imshow(img)  # Add this line to show the image
            axes[i].set_title(
                f"Actual: {actual_labels[i]}\nPredicted: {predictions[i]}",
                color="green" if actual_labels[i] == predictions[i] else "red",
            )
            axes[i].axis("off")
        except Exception as e:
            axes[i].text(
                0.5,
                0.5,
                f"Error loading image",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
            )
            axes[i].axis("off")

    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs("predict", exist_ok=True)
    plt.savefig("predict/sample_predictions.png")
    plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    # Usage example
    model_path = "/home/dexer/Repos/Python/Nadeko/runs/train/yolo_train_20250608_200028/weights/best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    predict_and_plot(model_path, device)
