import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path
import logging
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class NadekoDataset(Dataset):
    def __init__(self, data_dir="./data", img_size=640, transform=None, split="train"):
        """
        Custom dataset for YOLO models

        Args:
            data_dir (str): Directory containing dataset
            img_size (int): Input image size for YOLO
            transform: Custom transformations
            split (str): 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")

        self.img_size = img_size
        self.split = split
        self.transform = transform
        self.class_names = self._get_class_names()
        self.num_classes = len(self.class_names)
        logging.info(f"Found {self.num_classes} classes in {self.data_dir}")

        self.image_paths, self.label_paths = self._get_image_and_label_paths()
        logging.info(
            f"Found {len(self.image_paths)} valid image-label pairs for {split} split"
        )

        if len(self.image_paths) == 0:
            logging.warning(
                f"No images found for {split} split. Check your data directory structure."
            )

        # Default transforms if none provided
        if self.transform is None:
            self.transform = A.Compose(
                [
                    A.Resize(height=img_size, width=img_size),
                    A.Normalize(),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )

    def _get_class_names(self):
        """Get class names from directories"""
        class_dirs = [
            d
            for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ]
        class_names = sorted(class_dirs)
        return class_names

    def _get_image_and_label_paths(self):
        """Get paths to images and corresponding labels"""
        image_paths = []
        label_paths = []
        class_indices = []  # Store class index for each image

        # Determine which subset to use (train/val/test)
        if self.split == "train":
            subset_ratio = (0, 0.8)  # First 80%
        elif self.split == "val":
            subset_ratio = (0.8, 0.9)  # Next 10%
        else:  # test
            subset_ratio = (0.9, 1.0)  # Last 10%

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            img_paths = sorted(
                glob.glob(os.path.join(class_dir, "*.jpg"))
                + glob.glob(os.path.join(class_dir, "*.jpeg"))
                + glob.glob(os.path.join(class_dir, "*.png"))
            )

            logging.debug(f"Found {len(img_paths)} images in class {class_name}")

            if len(img_paths) == 0:
                continue

            # Split based on ratio, but ensure at least one image in each split
            if len(img_paths) < 3:
                # If too few images, use all for train
                if self.split == "train":
                    subset_img_paths = img_paths
                else:
                    subset_img_paths = []
            else:
                start_idx = int(len(img_paths) * subset_ratio[0])
                end_idx = int(len(img_paths) * subset_ratio[1])
                # Ensure at least one image in each split
                if start_idx == end_idx:
                    if self.split == "train":
                        end_idx = start_idx + 1
                    elif self.split == "val" and start_idx > 0:
                        start_idx = start_idx - 1

                subset_img_paths = img_paths[start_idx:end_idx]

            for img_path in subset_img_paths:
                # Check for label file first (for compatibility with YOLO format)
                label_path = os.path.splitext(img_path)[0] + ".txt"

                if os.path.exists(label_path):
                    # Use existing YOLO format label file
                    image_paths.append(img_path)
                    label_paths.append(label_path)
                else:
                    # Use folder name as the class label
                    # Store a special value to indicate we need to generate a label
                    image_paths.append(img_path)
                    label_paths.append("FOLDER_LABEL")  # Special marker
                    class_indices.append(class_idx)
                    logging.debug(f"Using folder name as label for {img_path}")

        # Store class indices for use in __getitem__
        self.class_indices = class_indices if len(class_indices) > 0 else None

        logging.info(f"Total images found: {len(image_paths)}")
        return image_paths, label_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get image and label by index"""
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Failed to load image: {img_path}")
            # Return a dummy image in case of error
            img = np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load labels
        boxes = []
        class_labels = []

        if label_path == "FOLDER_LABEL":
            # If using folder as label, create a full-image bounding box
            # The class index comes from the folder name
            class_idx = self.class_indices[idx] if self.class_indices else 0

            # Create a box that covers the whole image (YOLO format is normalized)
            boxes.append([0.5, 0.5, 1.0, 1.0])  # center_x, center_y, width, height
            class_labels.append(class_idx)
        elif os.path.exists(label_path):
            # Process standard YOLO format label file
            with open(label_path, "r") as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) >= 5:  # class, x, y, w, h
                        class_idx = int(data[0])
                        x_center, y_center, width, height = map(float, data[1:5])

                        # YOLO format is already normalized
                        boxes.append([x_center, y_center, width, height])
                        class_labels.append(class_idx)

        # Apply transformations
        if self.transform and len(boxes) > 0:
            transformed = self.transform(
                image=img, bboxes=boxes, class_labels=class_labels
            )
            img = transformed["image"]
            boxes = transformed["bboxes"]
            class_labels = transformed["class_labels"]
        elif self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]

        # Convert to tensor format for YOLO
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            class_labels = torch.tensor(class_labels, dtype=torch.int64)
            # Combine class and boxes for YOLO format
            labels = torch.zeros((len(boxes), 5))
            labels[:, 0] = class_labels
            labels[:, 1:] = boxes
        else:
            labels = torch.zeros((0, 5))

        return img, labels, img_path


# Helper function to create data loaders
def create_dataloaders(data_dir="./data", batch_size=16, img_size=640, num_workers=4):
    """Create DataLoaders for training, validation and testing"""
    # Verify data directory exists
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_path} does not exist")

    logging.info(f"Creating dataloaders from {data_path}")

    # Create datasets
    train_dataset = NadekoDataset(data_dir=data_dir, img_size=img_size, split="train")
    val_dataset = NadekoDataset(data_dir=data_dir, img_size=img_size, split="val")
    test_dataset = NadekoDataset(data_dir=data_dir, img_size=img_size, split="test")

    # Check if datasets have samples
    if len(train_dataset) == 0:
        logging.warning(
            "Training dataset is empty! Verify your data directory structure."
        )
        # Fallback to use all data for train if we can find any images
        all_dataset = NadekoDataset(data_dir=data_dir, img_size=img_size, split="train")
        all_dataset.split = "all"  # Set split to 'all' to try to find any images
        all_dataset.image_paths, all_dataset.label_paths = (
            all_dataset._get_all_image_and_label_paths()
        )
        if len(all_dataset) > 0:
            logging.info(f"Using all {len(all_dataset)} images for training")
            train_dataset = all_dataset

    # Create data loaders - use empty data loader if dataset is empty
    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        if len(train_dataset) > 0
        else None
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        if len(val_dataset) > 0
        else None
    )

    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        if len(test_dataset) > 0
        else None
    )

    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """Custom collate function for handling variable size data"""
    imgs, labels, paths = zip(*batch)
    # Remove empty boxes
    labels = [label for label in labels if label.shape[0] > 0]
    # Stack images
    imgs = torch.stack([img for img in imgs])
    return imgs, labels, paths


# Add helper method to get all images (fallback)
def _get_all_image_and_label_paths(self):
    """Get all image paths regardless of split"""
    image_paths = []
    label_paths = []

    for class_idx, class_name in enumerate(self.class_names):
        class_dir = os.path.join(self.data_dir, class_name)
        img_paths = sorted(
            glob.glob(os.path.join(class_dir, "*.jpg"))
            + glob.glob(os.path.join(class_dir, "*.jpeg"))
            + glob.glob(os.path.join(class_dir, "*.png"))
        )

        for img_path in img_paths:
            label_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(label_path):
                image_paths.append(img_path)
                label_paths.append(label_path)

    return image_paths, label_paths


# Add the method to the NadekoDataset class
NadekoDataset._get_all_image_and_label_paths = _get_all_image_and_label_paths


if __name__ == "__main__":
    # Example usage
    try:
        train_loader, val_loader, test_loader = create_dataloaders()
        print(f"Train batches: {len(train_loader) if train_loader else 0}")
        print(f"Val batches: {len(val_loader) if val_loader else 0}")
        print(f"Test batches: {len(test_loader) if test_loader else 0}")

        # Print number of classes
        if train_loader:
            dataset = train_loader.dataset
            print(f"Number of classes: {dataset.num_classes}")
            print(f"Class names: {dataset.class_names[:10]}... (showing first 10)")

            # Test loading a batch
            for imgs, labels, paths in train_loader:
                print(f"Batch shape: {imgs.shape}")
                print(f"Number of labels: {len(labels)}")
                break
        else:
            print("No training data found. Check your dataset directory structure.")
    except Exception as e:
        print(f"Error: {e}")
        # Print diagnostics about the data directory
        data_dir = Path("./data")
        if data_dir.exists():
            print(f"Data directory exists: {data_dir.absolute()}")
            print(f"Contents: {os.listdir(data_dir)}")
        else:
            print(f"Data directory does not exist: {data_dir.absolute()}")
