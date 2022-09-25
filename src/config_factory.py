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


def get_layer(special_layer: str) -> Union[nn.Module, None]:
    """
    Returns layer based on special_layer string
    :param special_layer: name of the layer
    :return: layer object
    """
    if special_layer == "adaptive_sigmoid":
        return WindowLayerAdaptiveSigmoid()
    elif special_layer == "adaptive_tanh":
        return WindowLayerAdaptiveTanh()
    elif special_layer == "hard_tanh":
        return WindowLayerHardTanH()
    return None


def config_factory(
    mode: ConfigMode,
    special_layer: str,
    overwrite: bool,
    batch_size: int,
    epochs: int = 50,
    learning_rate: float = 0.0001,
) -> Union[TrainingConfig, TestingConfig]:
    """
    Returns config based on parameters
    :param mode: mode of the config
    :param special_layer: name of the special layer
    :param overwrite: if true old training will be renamed
    :param batch_size: batch size for training
    :param epochs: number of epochs
    :param learning_rate: learning rate
    :return:
    """
    layer = get_layer(special_layer)
    if mode == ConfigMode.TRAIN:
        return TrainingConfig(layer, overwrite, batch_size, epochs, learning_rate)
    elif mode == ConfigMode.TEST:
        return TestingConfig(layer, overwrite, batch_size)
    raise ValueError("Invalid mode")
