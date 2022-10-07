import argparse

from src.data_loader import TomographyDataset
from src.prepare_dataset import load_metadata
from src.testing.testing_manager import run_test
from training.training_manager import run_training
import torch
import torch.multiprocessing as mp
from src.config_factory import config_factory, ConfigMode


def training_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("batch_size", type=int, help="Batch size")
    parser.add_argument("epochs", type=int, help="Number of epochs")
    parser.add_argument("gpu", type=int, help="GPU no")
    parser.add_argument("fold", type=int, help="Fold number")
    parser.add_argument("learning_rate", type=float, help="network learning rate")
    parser.add_argument(
        "window_learning_rate", type=float, help="learning rate of window layer"
    )
    parser.add_argument("window_width", type=int, help="width of window layer")
    parser.add_argument("window_center", type=int, help="center of window layer")
    parser.add_argument(
        "img_size",
        type=int,
        help="Size of image, it should be lower than image width/height",
    )
    parser.add_argument("metadata", type=str, help="Metadata path")
    parser.add_argument("dataset", type=str, help="Dataset path")
    parser.add_argument(
        "experiment",
        type=str,
        help="Experimenal layer",
        choices=["normal_unet", "hard_tanh", "adaptive_sigmoid", "adaptive_tanh"],
    )
    parser.add_argument("name", type=str, help="name of training")
    parser.add_argument("--use_batch_norm", action="store_true", help="Use batch norm")
    parser.add_argument("--tumor", action="store_true", help="Use tumor labels")
    return parser.parse_args()


def main():
    mp.set_start_method("spawn", force=True)
    args = training_arg_parser()

    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # U-net
    name = args.name
    config = config_factory(
        ConfigMode.TRAIN,
        args.experiment,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.window_learning_rate,
        args.use_batch_norm,
        args.window_width,
        args.window_center,
    )
    print(f"Running {args.experiment}")

    metadata = load_metadata(args.metadata)

    metadata.drop("series_id", axis=1, inplace=True)
    metadata = metadata.to_numpy()
    dataset = TomographyDataset(
        args.dataset, metadata, target_size=args.img_size, tumor=args.tumor
    )

    train, test = dataset.train_test_split(0.2)
    folds = dataset.k_fold_split(train, k=config.k_folds)
    print(folds)

    folds_data_loaders = dataset.create_k_fold_data_loaders(
        folds, batch_size=config.batch_size
    )

    for i, fold_data_loaders in enumerate(folds_data_loaders):
        if i == args.fold:
            run_training(
                name,
                config,
                device,
                fold_data_loaders,
            )
    test_config = config_factory(
        ConfigMode.TEST,
        args.experiment,
        args.batch_size,
        use_batch_norm=args.use_batch_norm,
    )
    test_dataset = dataset.create_data_loader(test, config.batch_size)
    run_test(name, test_config, device, test_dataset)


if __name__ == "__main__":
    main()
