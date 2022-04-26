import os
import shutil
import math
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import gc
from typing import Tuple
from sklearn import preprocessing
from sklearn.utils import shuffle

from src.data_process.config_paths import DataPathsManager
from src.training.training_config import TrainingConfig

def run_training(
    training_name: str,
    training_metadata: pd.DataFrame,
    training_path: str,
    validation_metadata: pd.DataFrame,
    validation_path: str,
    data_paths: DataPathsManager,
    augment: bool,
    overwrite_previous: bool = False,
) -> None:
    training_config = TrainingConfig

    # Protect previous models from overwriting
    if os.path.exists(f"{data_paths.model_path}{training_name}"):
        if overwrite_previous:
            print("WARNING: Model with the same name already exists. Overwriting it...")
            shutil.rmtree(f"{data_paths.model_path}{training_name}")
        else:
            print("ERROR: Model with the same name already exists. Skipping...")
            print("INFO: To overwrite the models, use the overwrite_previous flag.")

    if os.path.exists(f"{data_paths.training_log_path}{training_name}"):
        if overwrite_previous:
            print(
                "WARNING: Logs with the same name already exists. Overwriting them..."
            )
            shutil.rmtree(f"{data_paths.training_log_path}{training_name}")
        else:
            print("ERROR: Logs with the same name already exists. Skipping...")
            print("INFO: To overwrite the logs, use the overwrite_previous flag.")

