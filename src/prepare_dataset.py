import nibabel as nib
import numpy as np
import pandas as pd
import re
import os
from data_loader import TomographyDataset
import pydicom


def get_all_files(dir_path, ext=""):
    """
    Get list of all files in directory recursively
    """
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(ext):
                file_list.append(os.path.join(root, file))
    return file_list


def process_nifti(file: str, output_path: str, dir_name: str, pg: bool = False) -> None:
    filename = (
        os.path.basename(file)
        .replace(".nii.gz", "")
        .replace("liver_", "")
        .replace("nii", "")
    )
    save_dir = os.path.join(output_path, dir_name, filename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image = nib.load(file).get_fdata()

    for i in range(image.shape[2]):
        image_slice = image[:, :, i]
        index = i + 1 if pg else i
        npy_filename = f"slice_{index}.npz"
        np.savez_compressed(os.path.join(save_dir, npy_filename), image_slice)


def prepare_lits_dataset(dataset_path, output_path):
    images_path = os.path.join(dataset_path, "imagesTr_gz")
    labels_path = os.path.join(dataset_path, "labelsTr_gz")

    images_files = get_all_files(images_path, ".gz")[:2]
    labels_files = get_all_files(labels_path, ".gz")[:2]

    for files, dir_name in ((images_files, "images"), (labels_files, "labels")):
        for file in files:
            process_nifti(file, output_path, dir_name)


def prepare_pg_dataset(dataset_path, output_path):
    images_path = os.path.join(dataset_path, "Liver3D_originals")
    labels_path = os.path.join(dataset_path, "Liver3D_labels")
    # Get files
    images_files = get_all_files(images_path)
    labels_files = get_all_files(labels_path)

    # Images are in DICom format
    for file in images_files:
        filename = os.path.basename(file)
        if filename == "DICOMDIR":
            continue
        save_dir = os.path.join(output_path, "images")
        # \001\DICOMS\STU00001\SER00001 - we want to extract the patient ID, study ID and series ID
        dir_parts = file.split("\\")
        patient_id = dir_parts[-5]
        series_id = dir_parts[-2][-2:]

        save_dir = os.path.join(save_dir, f"{patient_id}_{series_id}")
        try:
            slice_number = int(filename[3:])
        except ValueError:
            continue
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ds = pydicom.read_file(file)
        img = ds.pixel_array.astype(float)
        npy_filename = f"slice_{slice_number}.npz"
        np.savez_compressed(os.path.join(save_dir, npy_filename), img)

    # Labels are in NIFTI format
    for file in labels_files:
        process_nifti(file, output_path, "labels", True)


def generate_csv(dir_path, csv_path):
    files = get_all_files(os.path.join(dir_path, "images"), ".npz")

    d = []
    for file in files:
        filename = os.path.basename(file)
        directory = os.path.basename(os.path.dirname(file))
        # if not os.path.exists(os.path.join(dir_path, "labels", directory, filename)):
        #   raise Exception("Corresponding label file not found")

        patient_id = int(directory)
        slice_id = int(re.search(r"([0-9]+)", filename).group(1))
        d.append({"filename": filename, "patient_id": patient_id, "slice_id": slice_id})
        # print(f"{filename} - {patient_id} - {slice_id}")

    df = pd.DataFrame(d)
    df = df.sort_values(["patient_id", "slice_id"], ascending=[True, True])
    save_path = os.path.join(csv_path, "metadata.csv")
    df.to_csv(save_path, index_label="id")


def load_metadata(csv_path):
    df = pd.read_csv(csv_path, index_col="id")
    return df


if __name__ == "__main__":
    # prepare_lits_dataset("D:\\domik\\Documents\\tomography\\data\\lits",
    #                      "D:\\domik\\Documents\\tomography\\data\\lits_prepared")
    # generate_csv("D:\\domik\\Documents\\tomography\\data\\lits_prepared",
    #              "D:\\domik\\Documents\\tomography\\data\\lits_prepared")
    prepare_pg_dataset(
        "D:\\domik\\Documents\\tomography\\data\\pg",
        "D:\\domik\\Documents\\tomography\\data\\pg_prepared",
    )
