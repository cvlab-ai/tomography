import os.path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from sklearn.model_selection import KFold


class TomographyDataset(Dataset):
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
        image = np.load(os.path.join(self.images_dir, filename))['arr_0']
        label = np.load(os.path.join(self.labels_dir, filename))['arr_0']

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def train_test_split(self, test_ratio: float, seed: int = 42) -> (list, list):
        """
        Split the dataset into train and test set by patient ids.
        :param test_ratio: float, ratio of test set size to whole dataset size
        :param seed: int, seed for random state
        :return: train - list of train slice indexes, test - list of test slice indexes
        """
        np.random.seed(seed)

        metadata_grouped = self.metadata.groupby("patient_id")
        patients = [*metadata_grouped.groups]
        np.random.shuffle(patients)

        test_size = int(len(patients) * test_ratio)
        test = patients[:test_size]
        train = patients[test_size:]

        return train, test

    def k_fold_split(self, ids: [int], k: int, seed=42) -> [dict[str, [int]]]:
        """
        Split input patient_id list into and k train-val folds.
        :param ids:
        :param test_ratio:
        :param k:
        :param seed:
        :return:
        """
        np.random.seed(seed)

        folds = []
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        for _, (train_idx, val_patient_idx) in enumerate(kf.split(ids)):
            fold_dict = {"train": list(train_idx), "val": list(val_patient_idx)}
            folds.append(fold_dict)

        return folds

    def create_k_fold_data_loaders(self, folds: [dict[str, [int]]], batch_size: int) -> [dict[str, DataLoader]]:
        folds_data_loaders = []
        for fold in folds:
            train_loader = self.create_data_loader(fold["train"], batch_size)
            val_loader = self.create_data_loader(fold["val"], batch_size)
            data_loaders_dict = {"train": train_loader, "val": val_loader}
            folds_data_loaders.append(data_loaders_dict)
        return folds_data_loaders

    def create_data_loader(self, patient_ids, batch_size, shuffle=True, seed=42):
        slice_ids = self.patients_to_slice_ids(patient_ids)

        if shuffle:
            sampler = SeededSubsetRandomSampler(slice_ids, seed)
        else:
            sampler = SubsetSequentialSampler(slice_ids)

        data_loader = DataLoader(self, batch_size=batch_size, sampler=sampler)
        return data_loader

    def patients_to_slice_ids(self, patients):
        slice_ids = list(self.metadata[self.metadata["patient_id"].isin(patients)].index.values)
        return slice_ids


class SeededSubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, seed):
        self.indices = indices
        self.g = torch.Generator()
        self.g.manual_seed(seed)

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices), generator=self.g))

    def __len__(self):
        return len(self.indices)


class SubsetSequentialSampler(Sampler):
    r"""Samples subset elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
