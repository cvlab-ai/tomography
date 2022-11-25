from datetime import datetime

from torch import optim
from torch.utils.data import DataLoader

import src.utils as utils
from src.training.training_config import TrainingConfig
import copy
import torch
import time
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
import gc


class RerunException(Exception):
    pass


def run_training(
    training_name: str,
    training_config: TrainingConfig,
    device: torch.device,
    data_loaders: dict,
) -> None:
    """
    Runs the training loop.
    :param device:
    :param training_config:
    :param training_name: name of the training
    :param data_loaders: dictionary of dataloaders
    """
    print(f"Training {training_name} on device: {device}")
    now = datetime.now()
    date = now.strftime("%d_%m_%Y")
    timestamp = datetime.now().strftime("%H_%M_%S")
    training_config.tb = utils.create_tensorboard(date, f"{training_name}_{timestamp}")
    training_config.net.train()
    best_model_wts = copy.deepcopy(training_config.net.state_dict())
    best_loss = 1e10
    global_step = 0

    if torch.cuda.is_available():
        training_config.net.cuda(device)

    try:
        for epoch in range(training_config.epochs):
            print("Epoch {}/{}".format(epoch, training_config.epochs - 1))
            print("-" * 10)

            since = time.time()
            if training_config.classes > 1:
                loss = utils.DiceLoss(ignore_index=0).to(device)
            else:
                loss = utils.DiceLoss().to(device)
            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    for param_group in training_config.optimizer.param_groups:
                        print("LR", param_group["lr"])

                    training_config.net.train()  # Set model to training mode
                else:
                    training_config.net.eval()  # Set model to evaluate mode

                metrics: dict = defaultdict(float)
                epoch_samples = 0
                with tqdm(
                    total=len(data_loaders[phase]) * training_config.batch_size,
                    desc=f"{phase} Epoch {epoch}/{training_config.epochs}",
                    unit="img",
                ) as pbar:
                    for (inputs, labels) in data_loaders[phase]:
                        inputs = inputs.to(device, dtype=torch.float32)
                        labels = labels.to(device, dtype=torch.long)
                        training_config.optimizer.zero_grad()
                        # forward
                        # track history if only in train
                        torch.set_grad_enabled(phase == "train")

                        outputs = training_config.net(inputs)
                        loss_value = loss(outputs, labels)

                        metrics["loss"] += loss_value.item() * inputs.size(0)
                        utils.calc_metrics(
                            outputs, labels, metrics, device, training_config.classes
                        )

                        # backward + optimize only if in training phase
                        # log center and width only in training phase
                        if phase == "train":
                            loss_value.backward()
                            training_config.optimizer.step()

                            if training_config.net.window_layer is not None:
                                training_config.tb.add_scalar(
                                    "center",
                                    utils.denorm_point(
                                        training_config.net.window_layer.center.item()
                                    ),
                                    global_step,
                                )
                                training_config.tb.add_scalar(
                                    "width",
                                    utils.denorm_point(
                                        training_config.net.window_layer.width.item()
                                    ),
                                    global_step,
                                )
                        # statistics
                        epoch_samples += inputs.size(0)
                        pbar.update(inputs.size(0))
                        global_step += 1
                        pbar.set_postfix(**{"loss (batch)": loss_value.item()})

                utils.print_metrics(
                    training_config.tb, metrics, epoch_samples, phase, epoch
                )
                epoch_loss = metrics["loss"] / epoch_samples

                try:
                    if phase == "val":
                        training_config.scheduler.step(epoch_loss)

                        if epoch_loss < best_loss:
                            print("saving best model")
                            best_loss = epoch_loss
                            best_model_wts = copy.deepcopy(
                                training_config.net.state_dict()
                            )

                        epoch_dice = metrics["dice"] / epoch_samples
                        if epoch >= 24 and epoch_dice < 50:
                            raise RerunException(f"Dice under 50 in epoch {epoch}")
                finally:
                    time_elapsed = time.time() - since
                    print(
                        "{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
    finally:
        print(f"Best val loss: {best_loss:4f}")
        # load best model weights
        training_config.net.load_state_dict(best_model_wts)
        # save model
        utils.save_model(training_config.net, training_name)
