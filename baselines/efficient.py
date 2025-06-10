import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import logging
from tqdm import tqdm
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class CustomEfficientNet:
    """EfficientNet model with YOLO-compatible interface"""

    def __init__(self, model_path=None, metrics_tracker=None):
        """
        Initialize EfficientNet model

        Args:
            model_path (str): Model path or version (e.g. 'efficientnet_b0')
            metrics_tracker: Object to track metrics during training
        """
        self.model_path = model_path or "efficientnet_b0"
        self.metrics_tracker = metrics_tracker
        self.model = None
        self.device = None
        self.criterion = nn.CrossEntropyLoss()

    def _build_model(self, num_classes=1000, pretrained=True):
        """Build the EfficientNet model"""
        try:
            # Try to load the specified version
            if self.model_path in models.__dict__:
                # Use torchvision's implementation if available
                weights = "DEFAULT" if pretrained else None
                self.model = models.__dict__[self.model_path](weights=weights)

                # Modify the classifier for our number of classes
                if hasattr(self.model, "classifier"):
                    in_features = self.model.classifier[1].in_features
                    self.model.classifier[1] = nn.Linear(in_features, num_classes)
                elif hasattr(self.model, "fc"):
                    in_features = self.model.fc.in_features
                    self.model.fc = nn.Linear(in_features, num_classes)
            else:
                # Try loading from local path
                self.model = torch.load(self.model_path)

            logger.info(f"Successfully loaded model: {self.model_path}")
        except Exception as e:
            logger.warning(
                f"Failed to load specified model: {e}. Using efficientnet_b0"
            )
            # Fallback to default EfficientNet
            weights = "DEFAULT" if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)

            # Modify the classifier for our number of classes
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)

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
        optimizer="adam",
        project="./runs/train",
        name="exp",
        exist_ok=True,
        pretrained=True,
        workers=4,
        **kwargs,
    ):
        """
        Train the EfficientNet model

        Args:
            data: Path to data directory or DataLoader
            epochs: Number of training epochs
            batch: Batch size
            imgsz: Image size
            device: Device to run on ('cpu', 'cuda', or 'auto')
            lr0: Initial learning rate
            weight_decay: Weight decay factor
            patience: Early stopping patience
            optimizer: Optimizer type ('adam', 'sgd', 'adamw')
            project: Project directory
            name: Experiment name
            exist_ok: Allow overwriting existing experiment
            pretrained: Use pretrained weights
            workers: Number of workers for data loading
        """
        # Set up device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Training on {self.device}")

        # Create dataloaders if data is a directory
        if isinstance(data, (str, Path)):
            from train.dataset import create_dataloaders

            train_loader, val_loader, _ = create_dataloaders(
                data_dir=data,
                batch_size=batch,
                img_size=imgsz,
                num_workers=workers,
                task="classification",
            )
        else:
            train_loader, val_loader = data, None

        if not train_loader:
            raise ValueError("No training data available")

        # Get number of classes from dataset
        num_classes = train_loader.dataset.num_classes
        logger.info(f"Training with {num_classes} classes")

        # Build model
        self.model = self._build_model(num_classes=num_classes, pretrained=pretrained)
        self.model.to(self.device)

        # Set up optimizer
        if optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr0, weight_decay=weight_decay
            )
        elif optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr0, weight_decay=weight_decay
            )
        else:  # sgd
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr0,
                momentum=kwargs.get("momentum", 0.9),
                weight_decay=weight_decay,
            )

        # Set up learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=patience // 3,
            min_lr=lr0 / 100,
            verbose=True,
        )

        # Training variables
        start_time = time.time()
        best_val_loss = float("inf")
        best_accuracy = 0
        patience_counter = 0

        # Setup save directory
        save_dir = Path(project) / name
        save_dir.mkdir(parents=True, exist_ok=exist_ok)

        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (inputs, targets, _) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Update metrics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": train_loss / (batch_idx + 1),
                        "accuracy": 100.0 * correct / total,
                    }
                )

            train_loss = train_loss / len(train_loader)
            train_accuracy = 100.0 * correct / total

            # Validation phase
            if val_loader:
                val_loss, val_accuracy = self._validate(val_loader)

                # Update learning rate
                scheduler.step(val_loss)

                # Check for improvement
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    torch.save(self.model.state_dict(), save_dir / "best.pt")
                    logger.info(f"Saved best model with accuracy {best_accuracy:.2f}%")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.2f}% | "
                    f"Val loss: {val_loss:.4f}, accuracy: {val_accuracy:.2f}%"
                )

                # Update metrics tracker
                if self.metrics_tracker:
                    self.metrics_tracker.update("train_loss", train_loss)
                    self.metrics_tracker.update("train_acc", train_accuracy)
                    self.metrics_tracker.update("val_loss", val_loss)
                    self.metrics_tracker.update("val_acc", val_accuracy)
            else:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.2f}%"
                )

                # Update metrics tracker
                if self.metrics_tracker:
                    self.metrics_tracker.update("train_loss", train_loss)
                    self.metrics_tracker.update("train_acc", train_accuracy)

        # Save final model
        torch.save(self.model.state_dict(), save_dir / "last.pt")

        # Training complete
        total_time = time.time() - start_time
        logger.info(f"Training complete. Total time: {total_time/60:.2f} minutes")

        # Plot metrics
        if self.metrics_tracker:
            self.metrics_tracker.plot_metrics()

        return {
            "model": self.model,
            "best_accuracy": best_accuracy,
            "best_val_loss": best_val_loss,
            "total_time": total_time,
            "epochs_trained": epoch + 1,
        }

    def _validate(self, dataloader):
        """Run validation on the provided dataloader"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets, _ in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        return val_loss, accuracy

    def val(self, data=None, split="val", **kwargs):
        """
        Validate model on dataset

        Args:
            data: Path to data directory or DataLoader
            split: Which split to use ('val' or 'test')
        """
        if not self.model:
            logger.error("No model available for validation")
            return {}

        # Set device if not already set
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        # Create dataloaders if data is a directory
        if isinstance(data, (str, Path)):
            from train.dataset import create_dataloaders

            _, val_loader, test_loader = create_dataloaders(
                data_dir=data,
                batch_size=kwargs.get("batch", 16),
                img_size=kwargs.get("imgsz", 224),
                num_workers=kwargs.get("workers", 4),
                task="classification",
            )

            # Select the appropriate loader
            if split == "test" and test_loader:
                dataloader = test_loader
            else:
                dataloader = val_loader
        else:
            dataloader = data

        if not dataloader:
            logger.warning(f"No data available for {split} split")
            return {}

        # Run validation
        val_loss, accuracy = self._validate(dataloader)

        logger.info(
            f"Validation on {split} split - Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        return {"loss": val_loss, "accuracy": accuracy}

    def predict(self, source, **kwargs):
        """
        Run prediction on images

        Args:
            source: Path to image or directory
        """
        # This is a stub for compatibility with YOLO interface
        logger.info(f"Prediction on {source} - not fully implemented")

        if not self.model:
            logger.error("No model available for prediction")
            return None

        # Would implement actual prediction code here

        return {"source": source, "message": "Prediction not fully implemented"}
