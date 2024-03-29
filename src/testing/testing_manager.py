import os

from torch.utils.data import DataLoader

from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

import src.utils as utils
from collections import defaultdict
from tqdm import tqdm
import gc

from src.testing.testing_config import TestingConfig


def run_test(
    weights_filename: str,
    testing_config: TestingConfig,
    device: torch.device,
    data_loader: DataLoader,
    log_dir: str,
    existing_tensorboard: SummaryWriter = None,
) -> None:
    """
    Runs the training loop.
    :param device:
    :param testing_config:
    :param existing_tensorboard:
    :param log_dir: log dir
    :param weights_filename: name of the weights file
    :param data_loader: DataLoader object
    """
    test_name = weights_filename + "_test"
    print(f"Testing {weights_filename} on device: {device}")
    if existing_tensorboard is None:
        existing_tensorboard = utils.create_tensorboard(log_dir, test_name)
    testing_config.tb = existing_tensorboard
    testing_config.net.load_state_dict(torch.load(os.path.join(log_dir, weights_filename, weights_filename)))

    if torch.cuda.is_available():
        testing_config.net.cuda(device)

    with torch.no_grad():
        testing_config.net.eval()  # Set model to evaluate mode
        metrics: dict = defaultdict(float)
        test_samples = 0
        if testing_config.classes > 1:
            loss = utils.DiceLoss(ignore_index=0).to(device)
        else:
            loss = utils.DiceLoss().to(device)
        with tqdm(
            total=len(data_loader) * testing_config.batch_size,
            desc=f"Testing {weights_filename}",
            unit="img",
        ) as pbar:
            for (inputs, labels) in data_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                outputs = testing_config.net(inputs)
                # if it is first sample, save first image to tensorboard

                loss_value = loss(outputs, labels)
                metrics["loss"] += loss_value.item() * inputs.size(0)
                if test_samples == 0:
                    testing_config.tb.add_image(
                        "inputs", inputs[0], 0, dataformats="CHW"
                    )

                    testing_config.tb.add_image(
                        "labels", labels[0], 0, dataformats="CHW"
                    )
                    # If output value less than 0.5, set to 0, else set to 1
                    output_tres = torch.where(
                        outputs[0] < 0.5,
                        torch.zeros_like(outputs[0]),
                        torch.ones_like(outputs[0]),
                    )
                    testing_config.tb.add_image(
                        "outputs", output_tres, 0, dataformats="CHW"
                    )
                utils.calc_metrics(
                    outputs, labels, metrics, device, testing_config.classes
                )

                test_samples += inputs.size(0)
                pbar.update(inputs.size(0))
                pbar.set_postfix(**{"loss (batch)": loss_value.item()})

        utils.print_metrics(testing_config.tb, metrics, test_samples, "test", 0)
