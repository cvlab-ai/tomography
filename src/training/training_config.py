from models.unet import UNet
import torch

class TrainingConfig:
    def __init__(self):
        # Batch size for training
        self.batch_size: int = 8

        # Number of epochs to train for
        self.epochs: int = 50

        # Learning rate
        self.learning_rate: float = 0.01
        self.learning_rate_patience: int = 3

        # Input shape
        self.input_h = 512  # Always 128
        self.input_w = 512  # Corresponds to the track's length; 512 is around 6 seconds
        self.channels = 1
        self.classess = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Mode layers definition
        self.net = UNet(n_channels=self.channels, n_classes=self.classess, bilinear=False).to(se)


