from enum import Enum
from typing import Union

import torch.nn as nn

from src.models.window_layer_adaptive_sigmoid import WindowLayerAdaptiveSigmoid
from src.models.window_layer_adaptive_tanh import WindowLayerAdaptiveTanh
from src.models.window_layer_hard_tanh import WindowLayerHardTanH
from src.testing.testing_config import TestingConfig
from src.training.training_config import TrainingConfig


class ConfigMode(Enum):
    """
    Enum for config mode
    """

    TRAIN = 1
    TEST = 2


def get_layer(
    special_layer: str, window_width: int, window_center: int
) -> Union[nn.Module, None]:
    """
    Returns layer based on special_layer string
    :param special_layer: name of the layer
    :return: layer object
    """
    if special_layer == "adaptive_sigmoid":
        return WindowLayerAdaptiveSigmoid(window_center, window_width)
    elif special_layer == "adaptive_tanh":
        return WindowLayerAdaptiveTanh(window_center, window_width)
    elif special_layer == "hard_tanh":
        return WindowLayerHardTanH(window_center, window_width)
    return None


def config_factory(
    mode: ConfigMode,
    special_layer: str,
    batch_size: int,
    epochs: int = 50,
    learning_rate: float = 0.0001,
    window_learning_rate: float = 0.0001,
    use_batch_norm: bool = True,
    window_width: int = 0,
    window_center: int = 0,
) -> Union[TrainingConfig, TestingConfig]:
    """
    Returns config based on parameters
    :param mode: mode of the config
    :param special_layer: name of the special layer
    :param batch_size: batch size for training
    :param epochs: number of epochs
    :param learning_rate: learning rate
    :param window_learning_rate: learning rate for window layer
    :param use_batch_norm: use batch norm
    :return:
    """
    layer = get_layer(special_layer, window_width, window_center)
    if mode == ConfigMode.TRAIN:
        return TrainingConfig(
            layer,
            batch_size,
            epochs,
            learning_rate,
            window_learning_rate,
            use_batch_norm,
        )
    elif mode == ConfigMode.TEST:
        return TestingConfig(layer, batch_size, use_batch_norm)
    raise ValueError("Invalid mode")
