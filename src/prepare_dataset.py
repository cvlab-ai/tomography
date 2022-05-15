import nibabel as nib
import numpy as np
import pandas as pd
import re
import os
from data_loader import TomographyDataset

def get_all_files(dir_path, ext = ""):
    """
    Get list of all files in directory recursively
    """
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(ext):
                file_list.append(os.path.join(root, file))
    return file_list

def prepare_lits_dataset(dataset_path, output_path):
    images_path = os.path.join(dataset_path, "imagesTr_gz")
    labels_path = os.path.join(dataset_path, "labelsTr_gz")

    images_files = get_all_files(images_path, ".gz")
    labels_files = get_all_files(labels_path, ".gz")

    for files, dir_name in ((images_files, "images"), (labels_files, "labels")):
        for file in files:

            filename = os.path.basename(file).replace(".nii.gz", "").replace("liver_", "")
            save_dir = os.path.join(output_path, dir_name, filename)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            image = nib.load(file).get_fdata()

            for i in range(image.shape[2]):
                image_slice = image[:, :, i]
                npy_filename = f"slice_{i}.npz"
                np.savez_compressed(os.path.join(save_dir, npy_filename), image_slice)

def generate_csv(dir_path, csv_path):
    files = get_all_files(os.path.join(dir_path, "images"), ".npz")
    dir_name = "images"

    d = []
    for file in files:
        filename = os.path.basename(file)
        directory = os.path.basename(os.path.dirname(file))
        # if not os.path.exists(os.path.join(dir_path, "labels", directory, filename)):
        #   raise Exception("Corresponding label file not found")

        patient_id = int(directory)
        slice_id = int(re.search(r"([0-9]+)", filename).group(1))
        d.append({"filename": filename, "patient_id": patient_id, "slice_id": slice_id})
        #print(f"{filename} - {patient_id} - {slice_id}")

    df = pd.DataFrame(d)
    df = df.sort_values(["patient_id", "slice_id"], ascending=[True, True])
    save_path = os.path.join(csv_path, "metadata.csv")
    df.to_csv(save_path, index_label="id")

def load_metadata(csv_path):
    df = pd.read_csv(csv_path, index_col="id")
    return df

if __name__ == "__main__":
    prepare_lits_dataset("C:\\Pg\\lits", "C:\\Pg\\lits_prepared")
    generate_csv("C:\\Pg\\lits_prepared", "C:\\Pg\\lits_prepared")