import matplotlib

# Use Agg backend (non-GUI) to avoid display server issues
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import sys
from sklearn.metrics import f1_score, confusion_matrix
from datetime import datetime

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
        # Check if it's a direct state dict (containing weights and biases directly)
        if any(
            k.endswith((".weight", ".bias", ".running_mean", ".running_var"))
            for k in checkpoint.keys()
        ):
            print("Detected direct state dict, creating model...")
            try:
                # If model path contains 'resnet', create a ResNet model
                if "resnet" in model_path.lower():
                    from torchvision.models import resnet50

                    num_classes = len(get_training_tags())
                    model = resnet50(pretrained=False)
                    # Modify the final layer to match our number of classes
                    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                    model.load_state_dict(checkpoint)
                    print(f"Created ResNet50 model with {num_classes} classes")
                else:
                    raise ValueError(
                        "Could not determine model architecture from model path"
                    )
            except Exception as e:
                print(f"Error creating model: {e}")
                raise
        elif "model" in checkpoint:
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
    # Explicitly move model to the specified device
    model = model.to(device)
    print(f"Model moved to device: {device}")
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
                # First send tensor to device, then add batch dimension
                img_tensor = img_tensor.to(device)
                img_batch = img_tensor.unsqueeze(0)

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

    # Calculate F1 score (macro average)
    try:
        f1 = f1_score(actual_labels, predictions, labels=training_tags, average="macro")
        print(f"F1 Score (macro): {f1:.4f}")

        # Calculate per-class F1 scores
        f1_per_class = f1_score(
            actual_labels, predictions, labels=training_tags, average=None
        )
        for i, tag in enumerate(training_tags):
            print(f"F1 Score for {tag}: {f1_per_class[i]:.4f}")
    except Exception as e:
        print(f"Error calculating F1 score: {e}", file=sys.stderr)

    # Plot confusion matrix
    plot_confusion_matrix(actual_labels, predictions, training_tags)


def plot_confusion_matrix(actual, predicted, class_names):
    """Plot confusion matrix alternatives for many classes"""
    try:
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, precision_score, recall_score
        from collections import Counter

        # Create timestamp-based directory to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("predict", f"results_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        # If we have too many classes (e.g., more than 50), use alternative visualizations
        if len(class_names) > 50:
            print(f"Using alternative visualizations for {len(class_names)} classes")

            # 1. Calculate per-class metrics
            precisions = precision_score(
                actual, predicted, labels=class_names, average=None, zero_division=0
            )
            recalls = recall_score(
                actual, predicted, labels=class_names, average=None, zero_division=0
            )

            # Count actual instances per class
            class_counts = Counter(actual)
            counts = [class_counts.get(cls, 0) for cls in class_names]

            # Create a DataFrame with metrics
            import pandas as pd

            metrics_df = pd.DataFrame(
                {
                    "Class": class_names,
                    "Count": counts,
                    "Precision": precisions,
                    "Recall": recalls,
                }
            )

            # Sort by count (descending)
            metrics_df = metrics_df.sort_values("Count", ascending=False)

            # 2. Plot top 30 classes by frequency
            plt.figure(figsize=(12, 10))
            top_classes = metrics_df.head(30)

            # Reformat class names to include count
            top_classes["ClassLabel"] = top_classes.apply(
                lambda x: f"{x['Class']} ({x['Count']})", axis=1
            )

            # Plot precision and recall
            x = np.arange(len(top_classes))
            width = 0.35

            fig, ax = plt.subplots(figsize=(15, 8))
            ax.bar(x - width / 2, top_classes["Precision"], width, label="Precision")
            ax.bar(x + width / 2, top_classes["Recall"], width, label="Recall")

            ax.set_xticks(x)
            ax.set_xticklabels(top_classes["ClassLabel"], rotation=45, ha="right")
            ax.legend()
            ax.set_ylim(0, 1.0)
            ax.set_title("Precision and Recall for Top 30 Classes")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "top_classes_metrics.png"))
            plt.close()

            # 3. Plot accuracy distribution
            # Calculate accuracy per class (when class appears in actual)
            class_correct = {}
            for cls in class_names:
                cls_indices = [i for i, label in enumerate(actual) if label == cls]
                if cls_indices:
                    correct = sum(1 for i in cls_indices if predicted[i] == actual[i])
                    class_correct[cls] = correct / len(cls_indices)
                else:
                    class_correct[cls] = 0

            # Plot histogram of class accuracies
            plt.figure(figsize=(10, 6))
            accuracies = list(class_correct.values())
            plt.hist(accuracies, bins=20, alpha=0.7)
            plt.axvline(
                x=np.mean(accuracies),
                color="red",
                linestyle="--",
                label=f"Mean Accuracy: {np.mean(accuracies):.3f}",
            )
            plt.xlabel("Accuracy")
            plt.ylabel("Number of Classes")
            plt.title("Distribution of Per-Class Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "accuracy_distribution.png"))
            plt.close()

            # 4. Plot most confused pairs
            # Create a simplified confusion matrix focusing on misclassifications
            cm = confusion_matrix(actual, predicted, labels=class_names)
            np.fill_diagonal(cm, 0)  # Remove diagonal (correct predictions)

            # Get the top confused pairs
            confused_pairs = []
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    if cm[i, j] > 0:
                        confused_pairs.append(
                            (class_names[i], class_names[j], cm[i, j])
                        )

            # Sort by confusion count (descending)
            confused_pairs.sort(key=lambda x: x[2], reverse=True)

            # Plot top confused pairs
            if confused_pairs:
                top_pairs = confused_pairs[:20]  # Get top 20 confused pairs
                plt.figure(figsize=(12, 8))
                pair_labels = [f"{actual} → {pred}" for actual, pred, _ in top_pairs]
                counts = [count for _, _, count in top_pairs]

                plt.barh(range(len(top_pairs)), counts, align="center")
                plt.yticks(range(len(top_pairs)), pair_labels)
                plt.xlabel("Count")
                plt.title("Top 20 Confused Class Pairs (Actual → Predicted)")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "top_confused_pairs.png"))
                plt.close()

            print(f"Created alternative visualizations in {save_dir}")

        else:
            # For fewer classes, plot the standard confusion matrix
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

            save_path = os.path.join(save_dir, "confusion_matrix.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved confusion matrix to {save_path}")

    except Exception as e:
        print(f"Error creating visualizations: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Usage example
    model_path = "/home/dexer/Repos/Python/Nadeko/runs/train/resnet_train_20250609_200651/best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    predict_and_plot(model_path, device)
