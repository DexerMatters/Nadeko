import yaml
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class ConfigLoader:
    def __init__(self, config_path: str):
        """
        Initialize the ConfigLoader by loading configurations from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config = {}
        self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
                if self.config is None:
                    self.config = {}
                    print(f"Warning: Empty configuration file at {config_path}")
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {config_path}: {e}")

    def get(self, key, default=None):
        """Get a configuration value by key"""
        return self.config.get(key, default)

    def create_optimizer(self, model_parameters):
        """
        Create an optimizer based on the configuration.

        Args:
            model_parameters: The parameters to optimize (e.g., model.parameters())

        Returns:
            A PyTorch optimizer instance
        """
        optimizer_name = self.get("optimizer", "adam").lower()
        lr = self.get("learning_rate", 0.001)

        if optimizer_name == "adam":
            return optim.Adam(model_parameters, lr=lr)
        elif optimizer_name == "sgd":
            return optim.SGD(model_parameters, lr=lr)
        elif optimizer_name == "rmsprop":
            return optim.RMSprop(model_parameters, lr=lr)
        elif optimizer_name == "adagrad":
            return optim.Adagrad(model_parameters, lr=lr)
        else:
            print(f"Warning: Unknown optimizer '{optimizer_name}', using Adam instead.")
            return optim.Adam(model_parameters, lr=lr)

    def create_lr_scheduler(self, optimizer):
        """
        Create a learning rate scheduler based on configuration.

        Args:
            optimizer: The optimizer to schedule

        Returns:
            A PyTorch learning rate scheduler
        """
        scheduler_type = self.get("lr_scheduler", "cosine").lower()
        epochs = self.get("epochs", 100)

        if scheduler_type == "cosine":
            return lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=self.get("lr_min", 1e-6),
            )
        elif scheduler_type == "step":
            return lr_scheduler.StepLR(
                optimizer,
                step_size=self.get("lr_step_size", 30),
                gamma=self.get("lr_gamma", 0.1),
            )
        elif scheduler_type == "exponential":
            return lr_scheduler.ExponentialLR(
                optimizer, gamma=self.get("lr_gamma", 0.9)
            )
        elif scheduler_type == "plateau":
            return lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.get("lr_factor", 0.1),
                patience=self.get("lr_patience", 10),
                min_lr=self.get("lr_min", 1e-6),
            )
        else:
            print(
                f"Warning: Unknown scheduler '{scheduler_type}', using CosineAnnealingLR instead."
            )
            return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def get_training_params(self):
        """
        Get a dictionary of training parameters.

        Returns:
            dict: Training parameters including batch_size, epochs, etc.
        """
        return {
            "batch_size": self.get("batch_size", 32),
            "epochs": self.get("epochs", 10),
            "learning_rate": self.get("learning_rate", 0.001),
        }

    def get_augmentation_params(self):
        """
        Get a dictionary of augmentation parameters.

        Returns:
            dict: Augmentation parameters for training
        """
        return {
            "mosaic": self.get("mosaic", 0.5),
            "mixup": self.get("mixup", 0.3),
            "degrees": self.get("degrees", 0.0),
            "translate": self.get("translate", 0.2),
            "scale": self.get("scale", 0.5),
            "shear": self.get("shear", 0.0),
            "perspective": self.get("perspective", 0.0),
            "flipud": self.get("flipud", 0.0),
            "fliplr": self.get("fliplr", 0.5),
            "hsv_h": self.get("hsv_h", 0.015),
            "hsv_s": self.get("hsv_s", 0.7),
            "hsv_v": self.get("hsv_v", 0.4),
        }

    def to_dict(self):
        """
        Convert the configuration to a dictionary.

        Returns:
            dict: The configuration as a dictionary
        """
        return self.config.copy()

    def __str__(self):
        """String representation of the configuration"""
        return str(self.config)
