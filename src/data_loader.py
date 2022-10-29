import os.path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
from scipy.ndimage import convolve
from collections import Counter
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
        normalize=False,
        discard=False,
        multiclass=False,
    ):
        # Store metadata, 2 and 3 column change type to string_
        self.metadata = metadata

        # Remove slices with only background
        if discard:
            self.metadata = np.delete(
                self.metadata, np.where(self.metadata[:, 4] == 1.0), axis=0
            )

        self.metadata[:, 2] = self.metadata[:, 2].astype(np.string_)
        self.metadata[:, 3] = self.metadata[:, 3].astype(np.string_)

        self.target_size = target_size
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform

        self.label_map_value = 2.0 if tumor is True else 1.0
        self.multiclass = multiclass
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

        #
        if self.multiclass:
            # Duplicate label map to create 3 channel label map
            label = np.repeat(label, 3, axis=0)
            # In first channel, sets 0 to 1, 1 to 0, 2 to 0
            label[0, :, :] = np.where(label[0, :, :] == 0, 1, 0)
            # In second channel, sets 0 to 0, 1 to 1, 2 to 0
            label[1, :, :] = np.where(label[1, :, :] == 1, 1, 0)
            # In third channel, sets 0 to 0, 1 to 0, 2 to 1
            label[2, :, :] = np.where(label[2, :, :] == 2, 1, 0)

        else:
            label = np.where(label >= self.label_map_value, 1, 0)

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
            image = norm_point(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def train_val_test_k_fold(
        self, test_ratio: float, k: int = 5, seed: int = 42
    ) -> Tuple[list, list]:
        """
        Split the dataset into train and test set by patient ids.
        :param test_ratio: float, ratio of test set size to whole dataset size
        :param k: folds number
        :param seed: int, seed for random state
        :return: train - list of train slice indexes, test - list of test slice indexes
        """
        np.random.seed(seed)

        # Get all patient ids, patient id has 1 index in np array
        # lambda r: 0 if r['tumor_percent'] == 0 else np.rint(-np.log10(r['tumor_percent']
        metadata = pd.DataFrame(self.metadata)
        patients = pd.DataFrame()
        patients["patient_id"] = metadata[1]
        patients["tumor_percent"] = metadata[6]
        patients = patients.groupby("patient_id").mean()
        print(patients)
        patients["tumor_magnitude"] = patients.apply(
            lambda r: 0
            if r["tumor_percent"] == 0
            else int(np.rint(-np.log10(r["tumor_percent"]))),
            axis=1,
        )
        patient_ids = patients.index.values.tolist()
        tumor_classes = patients["tumor_magnitude"].tolist()
        # Count number of each tumor class
        tumor_classes_count: Dict[int, int] = Counter(tumor_classes)
        # if there is <5 elements in any class, then decrease k
        for key, value in tumor_classes_count.items():
            if value < 5:
                tumor_classes[:] = [x - 1 if x == key else x for x in tumor_classes]
        np.random.seed(seed)

        # Get all patient ids, patient id has 1 index in np array
        old_patients = np.unique(self.metadata[:, 1])
        np.random.shuffle(old_patients)

        train, test = train_test_split(
            patient_ids, stratify=tumor_classes, random_state=seed, test_size=test_ratio
        )

        train_patients = patients.iloc[train]
        patient_ids = train_patients.index.values.tolist()
        tumor_classes = train_patients["tumor_magnitude"].tolist()

        folds = []
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for (train_idx, val_patient_idx) in kf.split(patient_ids, tumor_classes):
            val_patients = [patient_ids[i] for i in val_patient_idx]
            train_patients = [patient_ids[i] for i in train_idx]
            fold_dict = {"train": list(train_patients), "val": list(val_patients)}
            folds.append(fold_dict)

        # print(f'train:\t{len(patients.iloc[folds[0]["train"]])}\t{patients.iloc[folds[0]["train"]]["tumor_percent"].mean()}')
        # print(f'val:\t{len(patients.iloc[folds[0]["val"]])}\t{patients.iloc[folds[0]["val"]]["tumor_percent"].mean()}')
        # print(f'test:\t{len(patients.iloc[test])}\t{patients.iloc[test]["tumor_percent"].mean()}')

        return folds, test

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
