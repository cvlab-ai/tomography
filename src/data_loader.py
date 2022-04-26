import os.path

import numpy as np
import torch
from torch.utils.data import Dataset


class LitsDataset(Dataset):
    def __init__(self, dataset_dir, metadata, transform=None, target_transform=None):
        self.metadata = metadata
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.labels_dir = os.path.join(dataset_dir, "labels")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        item = self.metadata.iloc[idx]
        filename = f'{item["patient_id"]}\\slice_{item["slice_id"]}.npz'
        image = np.load(os.path.join(self.images_dir, filename))
        label = np.load(os.path.join(self.labels_dir, filename))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
