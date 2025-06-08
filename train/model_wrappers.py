import torch
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOCallback:
    """Callback to track YOLO training progress and update metrics"""

    def __init__(self, metrics_tracker):
        self.metrics_tracker = metrics_tracker
        self.best_val_loss = float("inf")
        self.current_epoch = 0

    def on_train_epoch_end(self, trainer):
        """Called at the end of each training epoch"""
        self.current_epoch += 1
        train_loss = trainer.loss.cpu().item() if hasattr(trainer, "loss") else None

        # Check if trainer exposes metrics
        metrics = getattr(trainer, "metrics", {})
        val_loss = metrics.get("val/loss", None)

        if train_loss is not None:
            self.metrics_tracker.update(self.current_epoch, train_loss, val_loss)

            # Log the losses
            log_str = f"Epoch {self.current_epoch}: train_loss={train_loss:.4f}"
            if val_loss is not None:
                log_str += f", val_loss={val_loss:.4f}"
            logger.info(log_str)


class CustomYOLO(YOLO):
    """Custom YOLO class with enhanced callbacks for tracking metrics"""

    def __init__(self, model_path, metrics_tracker=None):
        super().__init__(model_path)
        self.metrics_tracker = metrics_tracker
        self.current_epoch = 0

    def train(self, *args, **kwargs):
        """Override train method to capture epoch results"""
        # Set up a callback to capture per-epoch metrics
        original_on_train_epoch_end = (
            getattr(self.trainer, "on_train_epoch_end", None)
            if hasattr(self, "trainer")
            else None
        )

        def patched_on_train_epoch_end(trainer):
            """Custom callback to update metrics after each epoch"""
            # First call the original method if it exists
            if original_on_train_epoch_end:
                original_on_train_epoch_end()

            # Now update our metrics
            if self.metrics_tracker:
                self.current_epoch += 1
                # Try to get the loss values - different YOLO versions might store them differently
                train_loss = None
                val_loss = None
                accuracy = None

                # Try different ways to access the loss values
                if hasattr(trainer, "loss") and trainer.loss is not None:
                    train_loss = (
                        trainer.loss.detach().cpu().item()
                        if torch.is_tensor(trainer.loss)
                        else trainer.loss
                    )

                if hasattr(trainer, "metrics") and isinstance(trainer.metrics, dict):
                    val_loss = trainer.metrics.get("val/loss")
                    # Try to get accuracy metrics
                    accuracy = trainer.metrics.get(
                        "metrics/accuracy_top1",
                        trainer.metrics.get(
                            "val/accuracy", trainer.metrics.get("accuracy", None)
                        ),
                    )

                # If we still don't have the metrics, look in the history
                if train_loss is None and hasattr(trainer, "history"):
                    history = trainer.history
                    if isinstance(history, list) and len(history) > 0:
                        last_entry = history[-1]
                        if isinstance(last_entry, dict):
                            train_loss = last_entry.get("train/loss")
                            if val_loss is None:
                                val_loss = last_entry.get("val/loss")
                            if accuracy is None:
                                accuracy = last_entry.get(
                                    "metrics/accuracy_top1",
                                    last_entry.get(
                                        "val/accuracy", last_entry.get("accuracy", None)
                                    ),
                                )

                # Update the metrics if we have at least the training loss
                if train_loss is not None:
                    log_str = f"Epoch {self.current_epoch}: train_loss={train_loss:.4f}"
                    if val_loss is not None:
                        log_str += f", val_loss={val_loss:.4f}"
                    if accuracy is not None:
                        log_str += f", accuracy={accuracy:.4f}"
                    logger.info(log_str)

                    self.metrics_tracker.update(
                        self.current_epoch, train_loss, val_loss, accuracy
                    )

        # Try to patch the trainer's callback
        if hasattr(self, "add_callback"):
            self.add_callback("on_train_epoch_end", patched_on_train_epoch_end)
        elif hasattr(self, "callbacks") and isinstance(self.callbacks, dict):
            self.callbacks["on_train_epoch_end"] = [patched_on_train_epoch_end]

        # Start training and get the results
        results = super().train(*args, **kwargs)

        # If callbacks didn't work, try to extract metrics from results
        if self.current_epoch == 0 and isinstance(results, dict):
            metrics = results.get("metrics", [])
            if isinstance(metrics, list):
                for i, epoch_metrics in enumerate(metrics, 1):
                    if isinstance(epoch_metrics, dict):
                        train_loss = epoch_metrics.get("train/loss")
                        val_loss = epoch_metrics.get("val/loss")
                        if train_loss is not None:
                            self.metrics_tracker.update(i, train_loss, val_loss)

        return results
