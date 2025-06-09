import logging
from pathlib import Path
from datetime import datetime

import torch

from train.config import ConfigLoader
from train.dataset import create_dataloaders
from train.utils import seed_everything
from train.visualizers import MetricsTracker

logger = logging.getLogger(__name__)


def train(model: str):
    """Main training function"""
    # Load configuration
    if not model:
        logger.error("Model name must be provided.")
        return

    config = ConfigLoader(f"config_{model}.yml")
    logger.info(f"Loaded configuration: {config}")

    # Set seed for reproducibility
    seed_everything(config.get("seed", 42))

    # Prepare save directory
    save_dir = Path(config.get("save_dir", "./runs/train"))
    save_dir.mkdir(parents=True, exist_ok=True)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model}_train_{time_stamp}"
    save_dir = save_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize loss visualizer
    plots_dir = Path(config.get("plots_dir", "./train_plot"))
    metrics = MetricsTracker(save_dir=plots_dir)

    # Set device
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create dataloaders
    data_dir = config.get("data_dir", "./data")
    batch_size = config.get("batch_size", 16)
    img_size = config.get("img_size", 640)
    num_workers = config.get("num_workers", 4)

    logger.info(
        f"Creating dataloaders with batch_size={batch_size}, img_size={img_size}"
    )

    # Determine task type based on model
    task = "classification" if model.lower() in ["lenet", "resnet"] else "detection"

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        task=task,
    )

    if not train_loader:
        logger.error("No training data available. Check your data directory.")
        return

    # Get dataset info
    num_classes = train_loader.dataset.num_classes
    logger.info(f"Dataset has {num_classes} classes")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        logger.info(f"Validation samples: {len(val_loader.dataset)}")

    # Initialize the model based on the model type
    model_path = config.get("model_path")
    pretrained = config.get("pretrained", True)

    if model.lower() == "yolo":
        from baselines.yolo import CustomYOLO

        if not model_path:
            model_path = "yolo11s-cls.pt"
        logger.info(f"Initializing YOLO model from {model_path}")
        model_instance = CustomYOLO(model_path, metrics_tracker=metrics)
    elif model.lower() == "lenet":
        from baselines.lenet import CustomLeNet

        logger.info("Initializing LeNet model")
        model_instance = CustomLeNet(model_path, metrics_tracker=metrics)
    elif model.lower() == "resnet":
        from baselines.resnet import CustomResNet

        logger.info("Initializing ResNet model")
        model_instance = CustomResNet(model_path, metrics_tracker=metrics)
    else:
        logger.error(f"Unknown model type: {model}")
        return

    # Set up training parameters from config
    train_args = {
        "data": data_dir,
        "epochs": config.get("epochs", 100),
        "patience": config.get("patience", 10),
        "batch": batch_size,
        "imgsz": img_size,
        "device": device,
        "workers": num_workers,
        "project": str(save_dir.parent),
        "name": save_dir.name,
        "exist_ok": True,
        "pretrained": pretrained,
        "optimizer": config.get("optimizer", "adam").lower(),
        "lr0": config.get("learning_rate", 0.001),
        "weight_decay": config.get("weight_decay", 0.0005),
        "momentum": config.get("momentum", 0.937),
        "warmup_epochs": config.get("warmup_epochs", 3),
        "seed": config.get("seed", 42),
        "deterministic": config.get("deterministic", True),
    }

    # Add YOLO-specific parameters
    if model.lower() == "yolo":
        train_args["close_mosaic"] = 10

        # Add augmentation parameters if enabled
        if config.get("augment", True):
            train_args.update(
                {
                    "mosaic": config.get("mosaic", 0.5),
                    "mixup": config.get("mixup", 0.3),
                    "degrees": config.get("degrees", 0.0),
                    "translate": config.get("translate", 0.2),
                    "scale": config.get("scale", 0.5),
                    "shear": config.get("shear", 0.0),
                    "perspective": config.get("perspective", 0.0),
                    "flipud": config.get("flipud", 0.0),
                    "fliplr": config.get("fliplr", 0.5),
                    "hsv_h": config.get("hsv_h", 0.015),
                    "hsv_s": config.get("hsv_s", 0.7),
                    "hsv_v": config.get("hsv_v", 0.4),
                }
            )

    # Print training arguments
    logger.info("Training with the following parameters:")
    for k, v in train_args.items():
        logger.info(f"  {k}: {v}")

    # Start training
    logger.info("Starting training...")
    results = model_instance.train(**train_args)

    # Save final model
    final_model_path = save_dir / "best.pt"
    logger.info(f"Training completed. Best model saved to {final_model_path}")

    # Optionally run validation on test set
    if test_loader and config.get("run_test", True):
        logger.info("Running validation on test set...")
        results = model_instance.val(data=data_dir, split="test")
        logger.info(f"Test results: {results}")

    return model_instance, results, metrics
