import matplotlib.pyplot as plt
import matplotlib
import logging
import csv
import os
from pathlib import Path
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend

logger = logging.getLogger(__name__)


class LossVisualizer:
    """Class for real-time visualization of training and validation losses"""

    def __init__(self, save_dir="./train_plot"):
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
        self.epochs = []
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

        # CSV file path
        self.csv_path = self.save_dir / "training_metrics.csv"

        # Initialize CSV with headers if it doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["epoch", "train_loss", "val_loss", "accuracy"])
            logger.info(f"Created metrics CSV file at {self.csv_path}")

        # Set plot appearance
        plt.style.use("seaborn-v0_8-darkgrid")
        self.ax.set_title("Training and Validation Loss", fontsize=15)
        self.ax.set_xlabel("Epochs", fontsize=12)
        self.ax.set_ylabel("Loss", fontsize=12)

        logger.info(f"Loss plots will be saved to {self.save_dir.absolute()}")

    def update(self, epoch, train_loss, val_loss=None, accuracy=None):
        """Update the plot with new loss values and save to CSV"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)

        if val_loss is not None:
            self.val_losses.append(val_loss)
        else:
            # Use NaN for missing values in the CSV
            val_loss = float("nan")

        if accuracy is not None:
            self.accuracies.append(accuracy)
        else:
            accuracy = float("nan")

        # Save to CSV
        self._save_to_csv(epoch, train_loss, val_loss, accuracy)

        # Update the plot
        self._redraw()

    def _save_to_csv(self, epoch, train_loss, val_loss, accuracy):
        """Save metrics to CSV file"""
        with open(self.csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, train_loss, val_loss, accuracy])
        logger.debug(f"Saved metrics for epoch {epoch} to CSV")

    def _redraw(self):
        """Redraw the plot and save it"""
        self.ax.clear()

        # Plot training loss
        self.ax.plot(
            self.epochs,
            self.train_losses,
            "b-",
            label="Training Loss",
            linewidth=2,
        )

        # Plot validation loss if available
        if self.val_losses:
            self.ax.plot(
                self.epochs,
                self.val_losses,
                "r-",
                label="Validation Loss",
                linewidth=2,
            )

        # Add grid and legend
        self.ax.grid(True)
        self.ax.legend(loc="upper right")

        # Set labels and title
        self.ax.set_title("Training and Validation Loss", fontsize=15)
        self.ax.set_xlabel("Epochs", fontsize=12)
        self.ax.set_ylabel("Loss", fontsize=12)

        # Set axis limits
        if len(self.epochs) > 1:
            self.ax.set_xlim(1, max(self.epochs))

        if self.train_losses:
            max_loss = max(
                max(self.train_losses),
                max(self.val_losses) if self.val_losses else 0,
            )
            min_loss = min(
                min(self.train_losses),
                min(self.val_losses) if self.val_losses else float("inf"),
            )
            # Add some padding to the y-axis
            y_padding = (max_loss - min_loss) * 0.1 if max_loss > min_loss else 0.1
            self.ax.set_ylim(max(0, min_loss - y_padding), max_loss + y_padding)

        # Save the figure with tight layout
        self.fig.tight_layout()
        self.fig.savefig(self.save_dir / "loss_plot.png", dpi=300, bbox_inches="tight")
        logger.info(f"Updated loss plot saved to {self.save_dir / 'loss_plot.png'}")


class MetricsTracker:
    """Class for tracking and saving training metrics to CSV"""

    def __init__(self, save_dir="./train_plot"):
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
        self.epochs = []
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # CSV file path
        self.csv_path = self.save_dir / "training_metrics.csv"

        # Initialize CSV with headers if it doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["epoch", "train_loss", "val_loss", "accuracy"])
            logger.info(f"Created metrics CSV file at {self.csv_path}")

        logger.info(f"Training metrics will be saved to {self.csv_path}")

    def update(self, epoch, train_loss, val_loss=None, accuracy=None):
        """Update metrics and save to CSV"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)

        if val_loss is not None:
            self.val_losses.append(val_loss)
        else:
            # Use NaN for missing values in the CSV
            val_loss = float("nan")

        if accuracy is not None:
            self.accuracies.append(accuracy)
        else:
            accuracy = float("nan")

        # Save to CSV
        self._save_to_csv(epoch, train_loss, val_loss, accuracy)

    def _save_to_csv(self, epoch, train_loss, val_loss, accuracy):
        """Save metrics to CSV file"""
        with open(self.csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, train_loss, val_loss, accuracy])
        logger.debug(f"Saved metrics for epoch {epoch} to CSV")
