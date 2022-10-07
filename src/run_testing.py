import argparse

from src.data_loader import TomographyDataset
from src.prepare_dataset import load_metadata
from testing.testing_manager import run_test
import torch
import torch.multiprocessing as mp
from src.config_factory import config_factory, ConfigMode


def test_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("batch_size", type=int, help="Batch size")
    parser.add_argument("gpu", type=int, help="GPU no")
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
    parser.add_argument("test", type=str, help="Test model")
    parser.add_argument("--use_batch_norm", action="store_true", help="Use batch norm")
    return parser.parse_args()


def main():
    mp.set_start_method("spawn", force=True)
    args = test_arg_parser()

    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    name = args.experiment
    # U-net
    config = config_factory(
        ConfigMode.TEST, name, args.batch_size, use_batch_norm=args.use_batch_norm
    )
    metadata = load_metadata(args.metadata)
    print(metadata)
    metadata.drop("series_id", axis=1, inplace=True)
    metadata = metadata.to_numpy()
    dataset = TomographyDataset(args.dataset, metadata, target_size=args.img_size)

    _, test = dataset.train_test_split(0.2)
    print(test)

    test_dataset = dataset.create_data_loader(test, config.batch_size)
    weights_filename = args.test
    run_test(
        weights_filename,
        config,
        device,
        test_dataset,
    )


if __name__ == "__main__":
    main()
