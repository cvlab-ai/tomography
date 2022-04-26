from training.training_manager import run_training
import torch
import multiprocessing

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
