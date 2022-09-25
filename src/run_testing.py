import argparse

from src.data_loader import TomographyDataset
from src.models.window_layer_adaptive_sigmoid import WindowLayerAdaptiveSigmoid
from src.models.window_layer_adaptive_tanh import WindowLayerAdaptiveTanh
from src.prepare_dataset import load_metadata
from src.testing.testing_config import TestingConfig
from testing.testing_manager import run_test
import torch
import torch.multiprocessing as mp
import multiprocessing
from src.models.window_layer_hard_tanh import WindowLayerHardTanH

overwrite_previous_trainings = False

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("batch_size", type=int, help="Batch size")
    parser.add_argument("gpu", type=int, help="GPU no")
    parser.add_argument("metadata", type=str, help="Metadata path")
    parser.add_argument("dataset", type=str, help="Dataset path")
    parser.add_argument(
        "experiment",
        type=str,
        help="Experimenal layer",
        choices=["normal_unet", "hard_tanh", "adaptive_sigmoid", "adaptive_tanh"],
    )
    parser.add_argument("test", type=str, help="Test model")
    parser.add_argument(
        "-o", action="store_true", help="Overwrite previous trainings dirs"
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # U-net
    testing_config_normalunet = TestingConfig()
    testing_config_window_hard_tanh_unet = TestingConfig(WindowLayerHardTanH())
    testing_config_window_adaptive_sigmoid_unet = TestingConfig(
        WindowLayerAdaptiveSigmoid()
    )
    testing_config_window_adaptive_tanh_unet = TestingConfig(WindowLayerAdaptiveTanh())

    experiments = {
        "normal_unet": (testing_config_normalunet, "unet"),
        "hard_tanh": (testing_config_window_hard_tanh_unet, "unet-hard-tanh-window"),
        "adaptive_sigmoid": (
            testing_config_window_adaptive_sigmoid_unet,
            "unet-adaptive-sigmoid-window",
        ),
        "adaptive_tanh": (
            testing_config_window_adaptive_tanh_unet,
            "unet-adaptive-tanh-window",
        ),
    }

    config, name = experiments[args.experiment]
    print(f"Running {args.experiment}")

    if args.o:
        config.overwrite_previous_testing = True
        print("Overwriting previous trainings enabled")

    config.batch_size = args.batch_size
    metadata = load_metadata(args.metadata)
    print(metadata)
    metadata.drop("series_id", axis=1, inplace=True)
    metadata = metadata.to_numpy()
    dataset = TomographyDataset(args.dataset, metadata)

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
