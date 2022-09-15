from torch import optim

from src.training.training_config import TrainingConfig
import copy
import torch
import time
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
import gc


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
    training_config.create_tensorboard(training_name)
    training_config.net.train()
    best_model_wts = copy.deepcopy(training_config.net.state_dict())
    best_loss = 1e10

    global_step = 0

    if torch.cuda.is_available():
        training_config.net.cuda(device)

    for epoch in range(training_config.epochs):
        print("Epoch {}/{}".format(epoch, training_config.epochs - 1))
        print("-" * 10)

        since = time.time()

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
                desc=f"Epoch {epoch}/{training_config.epochs}",
                unit="img",
            ) as pbar:
                for (inputs, labels) in data_loaders[phase]:
                    inputs = inputs.to(device, dtype=torch.float32)
                    labels = labels.to(device, dtype=torch.long)
                    training_config.optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = training_config.net(inputs)
                        loss = training_config.dice_loss(outputs, labels)

                        metrics["loss"] += loss.item() * inputs.size(0)
                        training_config.calc_metrics(outputs, labels, metrics)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            training_config.grad_scaler.scale(loss).backward()
                            training_config.grad_scaler.step(training_config.optimizer)
                            training_config.grad_scaler.update()

                    # statistics
                    epoch_samples += inputs.size(0)
                    pbar.update(inputs.size(0))
                    global_step += 1
                    pbar.set_postfix(**{"loss (batch)": loss.item()})

            training_config.print_metrics(metrics, epoch_samples, phase, epoch)
            epoch_loss = metrics["loss"] / epoch_samples

            if phase == "val":
                training_config.scheduler.step(epoch_loss)
            if phase == "val" and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(training_config.net.state_dict())

        time_elapsed = time.time() - since
        print("{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Best val loss: {best_loss:4f}")

    # load best model weights
    training_config.net.load_state_dict(best_model_wts)
    training_config.close_tensorboard()
