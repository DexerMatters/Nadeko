import logging
import torch
import sys
from pathlib import Path

# Ensure the train module can be imported
sys.path.append(str(Path(__file__).parent))

from train import train

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        model, results, metrics_tracker = train("lenet")
        logger.info(f"Training metrics saved to {metrics_tracker.csv_path}")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
