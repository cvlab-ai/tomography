import argparse

import numpy as np

from src import utils
from src.data_loader import TomographyDataset
from src.prepare_dataset import load_metadata
from src.testing.testing_manager import run_test
from training.training_manager import run_training, RerunException
import torch
import torch.multiprocessing as mp
from src.config_factory import config_factory, ConfigMode
import torchmetrics


def training_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", required=True)
    parser.add_argument("--epochs", type=int, help="Number of epochs", required=True)
    parser.add_argument("--gpu", type=int, help="GPU no", required=True)
    parser.add_argument("--fold", type=int, help="Fold number", required=True)
    parser.add_argument("--learning_rate", type=float, help="network learning rate", required=True)
    parser.add_argument(
        "--window_learning_rate", type=float, help="learning rate of window layer", required=True
    )
    parser.add_argument("--n_windows", type=int, help="number of windows", required=True)
    parser.add_argument("--window_widths", type=str, help="space delimited widths of window layers", required=True)
    parser.add_argument("--window_centers", type=str, help="space delimited centers of window layers", required=True)
    parser.add_argument(
        "--img_size",
        type=int,
        help="Size of image, it should be lower than image width/height",
        required=True,
    )
    parser.add_argument("--metadata", type=str, help="Metadata path", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset path", required=True)
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experimenal layer",
        choices=["normal_unet", "hard_tanh", "adaptive_sigmoid", "adaptive_tanh"],
        required=True,
    )
    parser.add_argument("--name", type=str, help="name of training", required=True)

    parser.add_argument("--use_batch_norm", action="store_true", help="Use batch norm")
    parser.add_argument("--tumor", action="store_true", help="Use tumor labels")
    parser.add_argument("--normalize", action="store_true", help="Normalize images")
    parser.add_argument(
        "--val_test_switch", action="store_true", help="Normalize images"
    )
    parser.add_argument(
        "--discard", action="store_true", help="Discard images with 100% background"
    )
    parser.add_argument("--multiclass", action="store_true", help="Use multiclass")

    args = parser.parse_args()
    vars(args)["window_widths"] = [int(x) for x in args.window_widths.split()]
    vars(args)["window_centers"] = [int(x) for x in args.window_centers.split()]

    return args


def main():
    mp.set_start_method("spawn", force=True)
    args = training_arg_parser()

    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"Running {args.experiment}")

    metadata = load_metadata(args.metadata)

    metadata.drop("series_id", axis=1, inplace=True)
    metadata = metadata.to_numpy()
    dataset = TomographyDataset(
        args.dataset,
        metadata,
        target_size=args.img_size,
        tumor=args.tumor,
        normalize=args.normalize,
        discard=args.discard,
        multiclass=args.multiclass,
    )

    folds, test = dataset.train_val_test_k_fold(0.2)
    print(folds)

    if args.val_test_switch:
        tmp = folds[0]["val"]
        folds[0]["val"] = test
        test = tmp

    folds_data_loaders = dataset.create_k_fold_data_loaders(folds, args.batch_size)
    test_dataset = dataset.create_data_loader(test, args.batch_size)

    finished = False
    rerun = 0
    config = None
    while not finished:
        name = args.name if rerun == 0 else f"{args.name}_rerun-{rerun}"
        try:
            config = config_factory(
                mode=ConfigMode.TRAIN,
                special_layer=args.experiment,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                window_learning_rate=args.window_learning_rate,
                n_windows=args.n_windows,
                window_widths=args.window_widths,
                window_centers=args.window_centers,
                use_batch_norm=args.use_batch_norm,
                multiclass=args.multiclass,
            )

            run_training(name, config, device, folds_data_loaders[args.fold])
        except RerunException as e:
            rerun += 1
            print(e)
            print("Rerunning after test")
        except Exception as e:
            raise e
        else:
            finished = True
            test_config = config_factory(
                mode=ConfigMode.TEST,
                batch_size=args.batch_size,
                special_layer=args.experiment,
                n_windows=args.n_windows,
                window_widths=args.window_widths,
                window_centers=args.window_centers,
                use_batch_norm=args.use_batch_norm,
                multiclass=args.multiclass,
            )
            run_test(
                name, test_config, device, test_dataset, existing_tensorboard=config.tb
            )
        finally:
            if config is not None:
                utils.close_tensorboard(config.tb)


if __name__ == "__main__":
    main()
