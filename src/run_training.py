import argparse
import os
from datetime import datetime
from typing import IO

import sys
from src import utils
from src.data_loader import TomographyDataset
from src.prepare_dataset import load_metadata
from src.testing.testing_manager import run_test
from training.training_manager import run_training, RerunException
import torch
import torch.multiprocessing as mp
from src.config_factory import config_factory, ConfigMode


class FileConsoleOut(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def training_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", required=True)
    parser.add_argument("--epochs", type=int, help="Number of epochs", required=True)
    parser.add_argument("--gpu", type=int, help="GPU no", required=True)
    parser.add_argument("--fold", type=int, help="Fold number", required=True)
    parser.add_argument(
        "--learning_rate", type=float, help="network learning rate", required=True
    )
    parser.add_argument(
        "--window_learning_rate",
        type=float,
        help="learning rate of window layer",
        required=True,
    )
    parser.add_argument(
        "--n_windows", type=int, help="number of windows", required=True
    )
    parser.add_argument(
        "--window_widths",
        type=str,
        help="space delimited widths of window layers",
        required=True,
    )
    parser.add_argument(
        "--window_centers",
        type=str,
        help="space delimited centers of window layers",
        required=True,
    )
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
    parser.add_argument("--name", type=str, help="name of training")

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
    vars(args)["window_widths"] = [float(x) for x in args.window_widths.split()]
    vars(args)["window_centers"] = [float(x) for x in args.window_centers.split()]

    if args.n_windows != len(args.window_widths) or args.n_windows != len(
        args.window_centers
    ):
        raise ValueError(
            "Number of windows should be equal to number of window widths and centers"
        )

    return args


def main():
    root_path = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/../")
    now = datetime.now()
    timestamp = now.strftime("%d_%m_%Y__%H_%M_%S")
    log_dir = os.path.join(root_path, "runs", timestamp)
    if os.path.exists(log_dir):
        print("Log dir already exists")
        exit()
    os.makedirs(log_dir)
    log_file: IO = open(f"{log_dir}/log.log", "w")
    original_out = sys.stdout
    sys.stdout = FileConsoleOut(original_out, log_file)

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
    name_params = [
        f"size-{args.img_size}",
        f'{"multi-window" if args.n_windows > 1 else "single-window"}',
        f'{"multiclass" if args.multiclass else ""}',
        f'{"tumor" if args.tumor else ""}',
        f'{"normalize" if args.normalize else ""}',
        f'{"discard" if args.discard else ""}',
        f'{"val_test_switch" if args.val_test_switch else ""}',
        f"fold-{args.fold}",
        f"lr-{args.learning_rate}",
        f"wlr-{args.window_learning_rate}",
        f"nw-{args.n_windows}",
        f"ww-{list(map(int, args.window_widths))}",
        f"wl-{list(map(int, args.window_centers))}",
    ]
    name_params = filter(None, name_params)
    base_name = "_".join(name_params)
    if args.name:
        base_name = f"{args.name}_{base_name}"

    while not finished:
        name = base_name if rerun == 0 else f"{base_name}_rerun-{rerun}"
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

            run_training(
                training_name=name,
                log_dir=log_dir,
                training_config=config,
                device=device,
                data_loaders=folds_data_loaders[args.fold],
            )
        except RerunException as e:
            rerun += 1
            print(e)
            print("Rerunning after test")
        except Exception as e:
            print(e)
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
                weights_filename=name,
                log_dir=log_dir,
                testing_config=test_config,
                device=device,
                data_loader=test_dataset,
                existing_tensorboard=config.tb,
            )
        finally:
            if config is not None and config.tb is not None:
                utils.close_tensorboard(config.tb)
            sys.stdout = original_out
            log_file.close()


if __name__ == "__main__":
    main()
