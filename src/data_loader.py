import os.path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from sklearn.model_selection import KFold
from typing import Dict, Tuple
from scipy.ndimage import convolve

from src.utils import norm_point


class TomographyDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        metadata,
        target_size=512,
        transform=None,
        target_transform=None,
        tumor=False,
        normalize=False
    ):
        # Store metadata, 2 and 3 column change type to string_
        self.metadata = metadata
        self.metadata[:, 2] = self.metadata[:, 2].astype(np.string_)
        self.metadata[:, 3] = self.metadata[:, 3].astype(np.string_)

        self.target_size = target_size
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform

        self.label_map_value = 2.0 if tumor is True else 1.0
        self.normalize = normalize

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image = np.load(
            os.path.join(self.dataset_dir, str(self.metadata[idx][2], encoding="utf-8"))
        )["arr_0"]
        image = np.reshape(image, (1, image.shape[0], image.shape[1]))
        label = np.load(
            os.path.join(self.dataset_dir, str(self.metadata[idx][3], encoding="utf-8"))
        )["arr_0"]
        label = np.reshape(label, (1, label.shape[0], label.shape[1]))

        label_map = np.vectorize(lambda x: 1.0 if x >= self.label_map_value else 0.0)
        label[0] = label_map(label[0])

        if self.target_size != image.shape[1]:
            factor = int(image.shape[1] / self.target_size)
            filter = np.ones((1, factor, factor)) / (factor ** 2)
            # reshape all images and labels to target size using downscaling
            image = convolve(image, filter)[:, 0::factor, 0::factor]
            label_sampled = convolve(label, filter)[:, 0::factor, 0::factor]
            # check if label valuesare binary, if not, make them binary
            label = np.where(label_sampled > 0.5, 1, 0)

        if self.normalize:
            # Assuming that values are in range [-1024,4096], normalize to [-1,1]
            norm_point(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def train_test_split(self, test_ratio: float, seed: int = 42) -> Tuple[list, list]:
        """
        Split the dataset into train and test set by patient ids.
        :param test_ratio: float, ratio of test set size to whole dataset size
        :param seed: int, seed for random state
        :return: train - list of train slice indexes, test - list of test slice indexes
        """
        np.random.seed(seed)

        # Get all patient ids, patient id has 1 index in np array
        patients = np.unique(self.metadata[:, 1])
        np.random.shuffle(patients)

        test_size = int(len(patients) * test_ratio)
        test = patients[:test_size]
        train = patients[test_size:]

        return train, test

    def k_fold_split(self, ids: list, k: int, seed: int = 42) -> list:
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

    def create_k_fold_data_loaders(self, folds, batch_size):
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

        data_loader = DataLoader(
            self, batch_size=batch_size, sampler=sampler, num_workers=16
        )
        return data_loader

    def patients_to_slice_ids(self, patients):
        # Get all slice ids for given patients
        slice_ids = []
        for patient in patients:
            patient_slice_ids = np.where(self.metadata[:, 1] == patient)[0]
            slice_ids.extend(patient_slice_ids)
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
        return (
            self.indices[i] for i in torch.randperm(len(self.indices), generator=self.g)
        )

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
