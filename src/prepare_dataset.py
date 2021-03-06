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


def prepare_lits_labels(dataset_path, output_path):
    labels_path = os.path.join(dataset_path, "labelsTr_gz")
    labels_files = get_all_files(labels_path, ".gz")[:10]

    for file in labels_files:
        process_nifti(file, output_path, "labels")


def lits_metadata(processed_dataset_path):
    labels_files = get_all_files(os.path.join(processed_dataset_path, "labels"), ".npz")
    labels_files = list(
        map(lambda p: os.path.relpath(p, processed_dataset_path), labels_files)
    )

    df = pd.DataFrame(labels_files, columns=["label_path"])
    df["slice_id"] = df.apply(
        lambda row: int(
            re.search(r"(\d+)", os.path.basename(row["label_path"])).group(1)
        ),
        axis=1,
    )
    df["image_path"] = df.apply(
        lambda row: row["label_path"].replace("labels", "images"), axis=1
    )
    df["patient_id"] = df.apply(
        lambda row: int(os.path.basename(os.path.dirname(row["label_path"]))), axis=1
    )
    df["series_id"] = df.apply(lambda row: int(1), axis=1)
    df = df[["slice_id", "patient_id", "series_id", "image_path", "label_path"]]
    return df


def prepare_lits_images(dataset_path, output_path, labeled_patients: list[int]):
    images_path = os.path.join(dataset_path, "imagesTr_gz")
    images_files = get_all_files(images_path, ".gz")[:3]

    id_pattern = re.compile(r"(\d)")

    for file in images_files:
        res = id_pattern.search(os.path.basename(file))
        if res is None:
            raise Exception("No patient ID in filename")
        patient_id = int(res.group(1))
        if patient_id in labeled_patients:
            process_nifti(file, output_path, "images")


def prepare_lits_dataset(dataset_path, output_path):
    prepare_lits_labels(dataset_path, output_path)
    df = lits_metadata(output_path)
    labeled_patients = df["patient_id"].unique()
    prepare_lits_images(dataset_path, output_path, labeled_patients)
    df.to_csv(os.path.join(output_path, "metadata.csv"), index_label="id")


def pg_metadata(processed_dataset_path):
    labels_files = get_all_files(os.path.join(processed_dataset_path, "labels"), ".npz")

    labels_files = list(
        map(lambda p: os.path.relpath(p, processed_dataset_path), labels_files)
    )

    id_pattern = re.compile(r"(\d+)_(\d+).*")
    slice_id_pattern = re.compile(r"(\d+)")

    # labels_files = list(filter(lambda k: LabelType.Vesicle.value not in k, labels_files))
    df = pd.DataFrame(labels_files, columns=["label_path"])

    df["directory"] = df.apply(
        lambda row: os.path.split((row["label_path"]))[0], axis=1
    )
    df["filename"] = df.apply(lambda row: os.path.basename((row["label_path"])), axis=1)
    df["slice_id"] = df.apply(
        lambda row: slice_id_pattern.search(row["filename"]).group(1), axis=1
    )
    # Drop rows not containing proper patient ID and study ID format
    invalid_ids = df[df["directory"].map(lambda x: id_pattern.search(x) is None)]
    print(f"######\nDropping invalid rows:\n{invalid_ids}\n######\n\n")
    df.drop(invalid_ids.index, inplace=True)

    # Attach row with patient ID
    df["patient_id"] = df.apply(
        lambda row: int(id_pattern.search(row["label_path"]).group(1)), axis=1
    )

    # Attach row with series ID
    df["series_id"] = df.apply(
        lambda row: int(id_pattern.search(row["label_path"]).group(2)), axis=1
    )

    # Attach row with V qualifier
    v_pattern = re.compile(r"V(?!\w)")
    df["v"] = df.apply(
        lambda row: v_pattern.search(row["directory"], re.IGNORECASE) is not None,
        axis=1,
    )

    # Attach row with P qualifier
    p_pattern = re.compile(r"P(?!\w)")
    df["p"] = df.apply(
        lambda row: p_pattern.search(row["directory"], re.IGNORECASE) is not None,
        axis=1,
    )

    # Attach row with M qualifier
    m_pattern = re.compile(r"M(?!\w)")
    df["m"] = df.apply(
        lambda row: m_pattern.search(row["directory"], re.IGNORECASE) is not None,
        axis=1,
    )

    # Attach row with Vesicle qualifier
    vesicle_pattern = re.compile(r"vesicle")
    df["vesicle"] = df.apply(
        lambda row: vesicle_pattern.search(row["directory"], re.IGNORECASE) is not None,
        axis=1,
    )

    df["unqualified"] = df.apply(
        lambda row: row["p"] is False
        and row["v"] is False
        and row["m"] is False
        and row["vesicle"] is False,
        axis=1,
    )

    def get_image_path(row):
        image_dir = os.path.join(
            "images", f"{row['patient_id']:03d}_{row['series_id']:02d}"
        )
        image_path = os.path.join(image_dir, row["filename"])
        return image_path

    df["image_path"] = df.apply(get_image_path, axis=1)

    df = df.drop(["directory", "filename"], axis=1)
    df = df[
        [
            "slice_id",
            "patient_id",
            "series_id",
            "image_path",
            "label_path",
            "v",
            "p",
            "m",
            "unqualified",
        ]
    ]
    print(df.to_string())
    return df


def prepare_pg_labels(dataset_path, output_path):
    labels_path = os.path.join(dataset_path, "Liver3D_labels")
    labels_files = get_all_files(labels_path)

    id_pattern = re.compile(r"(\d+)_(\d+).*")
    # Labels are in NIFTI format
    for file in labels_files:
        if int(id_pattern.search(file).group(1)) >= 3:
            break
        process_nifti(file, output_path, "labels", True)


def prepare_pg_images(dataset_path, output_path, labeled_series: list[tuple[int, int]]):
    images_path = os.path.join(dataset_path, "Liver3D_originals")
    images_files = get_all_files(images_path)

    # Images are in DICom format
    for file in images_files:
        filename = os.path.basename(file)
        if filename == "DICOMDIR":
            continue
        save_dir = os.path.join(output_path, "images")
        # \001\DICOMS\STU00001\SER00001 - we want to extract the patient ID, study ID and series ID
        dir_parts = file.split("\\")
        patient_id = dir_parts[-5]
        try:
            int(patient_id)
        except ValueError:
            continue
        series_id = dir_parts[-2][-2:]

        # Skip images without labels
        if (int(patient_id), int(series_id)) not in labeled_series:
            continue

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


def prepare_pg_dataset(dataset_path, output_path):
    # prepare_pg_labels(dataset_path, output_path)
    df = pg_metadata(output_path)

    labeled_series = df[["patient_id", "series_id"]].drop_duplicates()
    labeled_series = list(labeled_series.itertuples(index=False, name=None))
    # prepare_pg_images(dataset_path, output_path, labeled_series)

    df.to_csv(os.path.join(output_path, "metadata.csv"), index_label="id")


def load_metadata(csv_path):
    df = pd.read_csv(csv_path, index_col="id")
    return df


if __name__ == "__main__":
    prepare_lits_dataset(
        "C:\\PG\\tomografia_pg\\liver", "C:\\PG\\tomografia_pg\\lits_prepared"
    )

    # prepare_pg_dataset(
    #     "C:\\PG\\tomografia_pg\\pg",
    #     "C:\\PG\\tomografia_pg\\pg_prepared"
    # )
