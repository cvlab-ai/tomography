from src.data_loader import TomographyDataset
from src.models.window_layer_adaptive_sigmoid import WindowLayerAdaptiveSigmoid
from src.models.window_layer_adaptive_tanh import WindowLayerAdaptiveTanh
from src.prepare_dataset import load_metadata
from src.training.training_config import TrainingConfig
from training.training_manager import run_training
import torch
import multiprocessing
from src.models.window_layer_hard_tanh import WindowLayerHardTanH

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # U-net

    training_config_normalunet = TrainingConfig()
    training_config_window_hard_tanh_unet = TrainingConfig(WindowLayerHardTanH())
    training_config_window_adaptive_sigmoid_unet = TrainingConfig(
        WindowLayerAdaptiveSigmoid()
    )
    training_config_window_adaptive_tanh_unet = TrainingConfig(
        WindowLayerAdaptiveTanh()
    )

    for config, name in [
        (training_config_normalunet, "unet"),
        (training_config_window_hard_tanh_unet, "unet-hard-tanh-window"),
        (training_config_window_adaptive_sigmoid_unet, "unet-adaptive-sigmoid-window"),
        (training_config_window_adaptive_tanh_unet, "unet-adaptive-tanh-window"),
    ]:
        metadata = load_metadata(
            "D:\\domik\\Documents\\tomography\\data\\lits-prepared\\metadata.csv"
        )
        print(metadata)
        dataset = TomographyDataset(
            "D:\\domik\\Documents\\tomography\\data\\lits-prepared", metadata
        )

        train, test = dataset.train_test_split(0.2)
        print(test)
        folds = dataset.k_fold_split(train, k=training_config_normalunet.k_folds)
        print(folds)

        test_data_loader = dataset.create_data_loader(
            test, batch_size=training_config_normalunet.batch_size
        )
        folds_data_loaders = dataset.create_k_fold_data_loaders(
            folds, batch_size=training_config_normalunet.batch_size
        )

        for i, fold_data_loaders in enumerate(folds_data_loaders):
            run_training(
                f"{name}-fold-{i}",
                training_config_normalunet,
                device,
                fold_data_loaders,
            )
