import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LeNetModel(nn.Module):
    """Simple LeNet-5 architecture for classification"""

    def __init__(self, num_classes=1000, input_size=224):
        super().__init__()
        self.num_classes = num_classes

        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            dummy_output = self.features(dummy_input)
            flatten_size = dummy_output.numel()

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CustomLeNet:
    """LeNet wrapper that mimics YOLO interface"""

    def __init__(self, model_path=None, metrics_tracker=None):
        self.metrics_tracker = metrics_tracker
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_epoch = 0

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def _create_model(self, num_classes, img_size):
        """Create LeNet model"""
        self.model = LeNetModel(num_classes=num_classes, input_size=img_size)
        self.model.to(self.device)
        return self.model

    def train(
        self,
        data=None,
        epochs=100,
        batch=16,
        imgsz=224,
        device=None,
        lr0=0.001,
        weight_decay=0.0005,
        patience=10,
        project="runs/train",
        name="lenet_train",
        optimizer="adam",
        **kwargs,
    ):
        """Train the LeNet model"""

        if device:
            self.device = torch.device(device)

        # Import here to avoid circular imports
        from train.dataset import create_dataloaders

        # Create dataloaders for classification task
        train_loader, val_loader, _ = create_dataloaders(
            data_dir=data,
            batch_size=batch,
            img_size=imgsz,
            num_workers=kwargs.get("workers", 4),
            task="classification",
        )

        if not train_loader:
            raise ValueError("No training data available")

        num_classes = train_loader.dataset.num_classes

        # Create model
        if self.model is None:
            self._create_model(num_classes, imgsz)

        # Setup optimizer
        if optimizer.lower() == "adam":
            optim = torch.optim.Adam(
                self.model.parameters(), lr=lr0, weight_decay=weight_decay
            )
        else:
            optim = torch.optim.SGD(
                self.model.parameters(), lr=lr0, weight_decay=weight_decay
            )

        # Setup loss and scheduler
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        save_dir = Path(project) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            self.current_epoch = epoch + 1

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, batch_data in enumerate(
                tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
            ):
                # For classification, we expect (images, targets, paths)
                images, targets, _ = batch_data
                images, targets = images.to(self.device), targets.to(self.device)

                optim.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optim.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total

            # Validation phase
            val_loss = None
            val_acc = None
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_data in val_loader:
                        images, targets, _ = batch_data
                        images, targets = images.to(self.device), targets.to(
                            self.device
                        )
                        outputs = self.model(images)
                        loss = criterion(outputs, targets)

                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()

                val_loss /= len(val_loader)
                val_acc = 100.0 * val_correct / val_total

            # Update metrics
            if self.metrics_tracker:
                self.metrics_tracker.update(
                    self.current_epoch, train_loss, val_loss, train_acc
                )

            # Logging
            log_str = f"Epoch {self.current_epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%"
            if val_loss is not None:
                log_str += f", val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
            logger.info(log_str)

            # Early stopping and model saving
            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_dir / "best.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {self.current_epoch}")
                    break

            scheduler.step()

        return {"best_loss": best_val_loss}

    def val(self, data=None, split="val", **kwargs):
        """Validate the model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        from train.dataset import create_dataloaders

        _, val_loader, test_loader = create_dataloaders(
            data_dir=data,
            batch_size=kwargs.get("batch", 16),
            img_size=kwargs.get("imgsz", 224),
            num_workers=kwargs.get("workers", 4),
            task="classification",
        )

        loader = test_loader if split == "test" else val_loader
        if not loader:
            return {}

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_data in loader:
                images, targets, _ = batch_data
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def load(self, path):
        """Load model weights"""
        if self.model is None:
            # Need to know architecture to load - this is a limitation
            logger.warning("Model architecture not initialized. Cannot load weights.")
            return

        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")

    def save(self, path):
        """Save model weights"""
        if self.model is None:
            raise ValueError("No model to save")

        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
