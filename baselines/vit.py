import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import logging
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class CustomViT:
    def __init__(self, model_path=None, metrics_tracker=None):
        """
        Vision Transformer model with YOLO-compatible interface

        Args:
            model_path (str): Path to pretrained model or model name
            metrics_tracker: Metrics tracking object
        """
        self.model_path = model_path or "vit_base_patch16_224"
        self.metrics_tracker = metrics_tracker
        self.model = None
        self.device = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()

    def _build_model(self, num_classes=1000, pretrained=True, img_size=224):
        """Build ViT model using timm"""
        try:
            # Use timm to create ViT model
            self.model = timm.create_model(
                self.model_path,
                pretrained=pretrained,
                num_classes=num_classes,
                img_size=img_size,
            )
            logger.info(
                f"Created ViT model: {self.model_path} with {num_classes} classes"
            )
        except Exception as e:
            logger.warning(f"Failed to create {self.model_path}, using default: {e}")
            self.model = timm.create_model(
                "vit_base_patch16_224",
                pretrained=pretrained,
                num_classes=num_classes,
                img_size=img_size,
            )

        return self.model

    def train(
        self,
        data,
        epochs=100,
        batch=16,
        imgsz=224,
        device="auto",
        lr0=0.001,
        weight_decay=0.0001,
        patience=10,
        optimizer="adamw",
        **kwargs,
    ):
        """
        Train the ViT model with YOLO-compatible interface

        Args:
            data: Data directory or dataloader
            epochs: Number of training epochs
            batch: Batch size
            imgsz: Image size
            device: Device to use
            lr0: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            optimizer: Optimizer type
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Training ViT on device: {self.device}")

        # Create dataloaders if data is a directory
        if isinstance(data, (str, Path)):
            from train.dataset import create_dataloaders

            train_loader, val_loader, _ = create_dataloaders(
                data_dir=data, batch_size=batch, img_size=imgsz, task="classification"
            )
        else:
            train_loader, val_loader = data, None

        if not train_loader:
            raise ValueError("No training data available")

        # Get number of classes from dataset
        num_classes = train_loader.dataset.num_classes

        # Build model
        self.model = self._build_model(
            num_classes=num_classes,
            pretrained=kwargs.get("pretrained", True),
            img_size=imgsz,
        )
        self.model.to(self.device)

        # Setup optimizer
        if optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr0, weight_decay=weight_decay
            )
        elif optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr0, weight_decay=weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr0,
                weight_decay=weight_decay,
                momentum=kwargs.get("momentum", 0.9),
            )

        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=patience // 2, factor=0.5
        )

        # Training loop
        best_acc = 0
        patience_counter = 0
        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, labels, _ in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                pbar.set_postfix(
                    {"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"}
                )

            epoch_loss = running_loss / len(train_loader)
            train_acc = 100.0 * correct / total
            train_losses.append(epoch_loss)

            # Validation phase
            val_acc = 0
            if val_loader:
                val_acc = self._validate(val_loader)
                val_accuracies.append(val_acc)
                self.scheduler.step(epoch_loss)

                # Early stopping and best model saving
                if val_acc > best_acc:
                    best_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    if "project" in kwargs and "name" in kwargs:
                        save_path = Path(kwargs["project"]) / kwargs["name"] / "best.pt"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1

                logger.info(
                    f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%"
                )

                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                logger.info(
                    f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Train Acc={train_acc:.2f}%"
                )

            # Track metrics
            if self.metrics_tracker:
                self.metrics_tracker.update("train_loss", epoch_loss)
                self.metrics_tracker.update("train_acc", train_acc)
                if val_acc > 0:
                    self.metrics_tracker.update("val_acc", val_acc)

        # Plot metrics
        if self.metrics_tracker:
            self.metrics_tracker.plot_metrics()

        results = {
            "best_accuracy": best_acc,
            "final_train_loss": train_losses[-1] if train_losses else 0,
            "epochs_trained": epoch + 1,
        }

        return results

    def _validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100.0 * correct / total

    def val(self, data=None, split="val", **kwargs):
        """Validate model with YOLO-compatible interface"""
        if isinstance(data, (str, Path)):
            from train.dataset import create_dataloaders

            _, val_loader, test_loader = create_dataloaders(
                data_dir=data,
                batch_size=kwargs.get("batch", 16),
                img_size=kwargs.get("imgsz", 224),
                task="classification",
            )

            if split == "test" and test_loader:
                val_loader = test_loader
        else:
            val_loader = data

        if not val_loader:
            logger.warning("No validation data available")
            return {}

        if not self.model:
            logger.error("Model not trained yet")
            return {}

        accuracy = self._validate(val_loader)
        logger.info(f"Validation Accuracy: {accuracy:.2f}%")

        return {"accuracy": accuracy}

    def predict(self, source, **kwargs):
        """Predict with YOLO-compatible interface"""
        if not self.model:
            logger.error("Model not trained yet")
            return None

        self.model.eval()
        # Implementation for prediction would go here
        # This is a placeholder for compatibility
        logger.info(f"Prediction on {source} - placeholder implementation")
        return None
