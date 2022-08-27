from src.data_loader import TomographyDataset
from src.prepare_dataset import load_metadata
from src.training.training_config import TrainingConfig
from training.training_manager import run_training
import torch
import multiprocessing

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_config = TrainingConfig()

    metadata = load_metadata(
        "D:\\domik\\Documents\\tomography\\data\\lits-prepared\\metadata.csv"
    )[:2000]
    print(metadata)
    dataset = TomographyDataset(
        "D:\\domik\\Documents\\tomography\\data\\lits-prepared", metadata
    )

    train, test = dataset.train_test_split(0.2)
    print(test)
    folds = dataset.k_fold_split(train, k=training_config.k_folds)
    print(folds)

    test_data_loader = dataset.create_data_loader(
        test, batch_size=training_config.batch_size
    )
    folds_data_loaders = dataset.create_k_fold_data_loaders(
        folds, batch_size=training_config.batch_size
    )

    for i, fold_data_loaders in enumerate(folds_data_loaders):
        run_training(
            f"u-net-window-fold-{i}", training_config, device, fold_data_loaders
        )
