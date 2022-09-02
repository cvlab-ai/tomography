from src.training.training_config import TrainingConfig
import copy
import torch
import time
from collections import defaultdict


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

    if torch.cuda.is_available():
        training_config.net.cuda()

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

            for batch_index, (inputs, labels) in enumerate(data_loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                training_config.optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = training_config.net(inputs.float())
                    loss = training_config.loss(outputs, labels)
                    metrics["loss"] += loss.item()
                    training_config.calc_metrics(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        training_config.optimizer.step()
                        training_config.scheduler.step()

                # statistics
                epoch_samples += inputs.size(0)

            training_config.print_metrics(metrics, epoch_samples, phase, epoch)
            epoch_loss = metrics["loss"] / epoch_samples

            # deep copy the model
            if phase == "val" and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(training_config.net.state_dict())

        time_elapsed = time.time() - since
        print("{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    print(f"Best val loss: {best_loss:4f}")

    # load best model weights
    training_config.net.load_state_dict(best_model_wts)
    training_config.close_tensorboard()
