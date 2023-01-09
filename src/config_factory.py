from enum import Enum
from typing import Union, List

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
    special_layer: str,
    n_channels: int,
    window_width: List[float],
    window_center: List[float],
) -> Union[nn.Module, None]:
    """
    Returns layer based on special_layer string
    :param special_layer: name of the layer
    :param n_channels: number of channels
    :param window_width: list of window widths
    :param window_center: list of window centers
    :return: layer object
    """
    if special_layer == "adaptive_sigmoid":
        # return WindowLayerAdaptiveSigmoid(window_center, window_width)
        raise NotImplementedError(
            "Adaptive sigmoid not implemented for multiple channels"
        )
    elif special_layer == "adaptive_tanh":
        return WindowLayerAdaptiveTanh(n_channels, window_center, window_width)
    elif special_layer == "hard_tanh":
        # return WindowLayerHardTanH(window_center, window_width)
        raise NotImplementedError("Hard tanh not implemented for multiple channels")
    return None


def config_factory(
    mode: ConfigMode,
    special_layer: str,
    batch_size: int,
    n_windows: int,
    window_widths: List[float],
    window_centers: List[float],
    epochs: int = 50,
    learning_rate: float = 0.0001,
    window_learning_rate: float = 0.0001,
    use_batch_norm: bool = True,
    multiclass: bool = False,
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
    layer = get_layer(special_layer, n_windows, window_widths, window_centers)
    n_channels = 1 if layer is None else n_windows
    if mode == ConfigMode.TRAIN:
        return TrainingConfig(
            custom_layer=layer,
            n_channels=n_channels,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            window_learning_rate=window_learning_rate,
            use_batch_norm=use_batch_norm,
            multiclass=multiclass,
        )
    elif mode == ConfigMode.TEST:
        return TestingConfig(
            custom_layer=layer,
            n_channels=n_channels,
            batch_size=batch_size,
            use_batchnorm=use_batch_norm,
            multiclass=multiclass,
        )
    raise ValueError("Invalid mode")
