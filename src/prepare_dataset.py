from pickle import UnpicklingError

from tqdm import tqdm
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
import re
import os

from typing import List, Tuple

from data_loader import TomographyDataset
import pydicom


def get_all_files(dir_path: str, ext: str = "") -> List[str]:
    """
    Get list of all files in directory recursively
    """
    file_list: List[str] = []
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

    if skip_existing and os.path.exists(save_dir):
        return

    if not os.path.exists(file):
        raise FileNotFoundError

    image = nib.load(file).get_fdata()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(image.shape[2]):
        image_slice = image[:, :, i]
        index = i + 1 if pg else i
        npy_filename = f"slice_{index}.npz"
        np.savez_compressed(os.path.join(save_dir, npy_filename), image_slice)


def prepare_lits_labels(dataset_path, output_path):
    labels_path = os.path.join(dataset_path, "labelsTr_gz")
    labels_files = get_all_files(labels_path, ".gz")

    print("Processing labels")
    for file in tqdm(labels_files):
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


def prepare_lits_images(
    dataset_path: str, output_path: str, labeled_patients: List[int]
) -> None:
    images_path = os.path.join(dataset_path, "imagesTr_gz")
    images_files = get_all_files(images_path, ".gz")

    id_pattern = re.compile(r"(\d+)")

    print("Processing images")
    for file in tqdm(images_files):
        res = id_pattern.search(os.path.basename(file))
        if res is None:
            raise Exception("No patient ID in filename")
        patient_id = int(res.group(1))
        if patient_id in labeled_patients:
            try:
                process_nifti(file, output_path, "images")
            except EOFError:
                print(f"Error processing patient {patient_id} image")
                # patient 43 WA
                if patient_id == 43:
                    wa43_path = os.path.join(
                        os.getcwd(), "image_fix", "liver_43.nii.gz"
                    )
                    print(f"Trying to load patient 43 image from {wa43_path}")
                    try:
                        process_nifti(wa43_path, output_path, "images")
                    except FileNotFoundError:
                        print("Fixed image file not found")


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

    df["directory"] = df.apply(lambda row: os.path.split(row["label_path"])[0], axis=1)
    df["filename"] = df.apply(lambda row: os.path.basename(row["label_path"]), axis=1)
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
    print("Processing labels")
    for file in tqdm(labels_files):
        if int(id_pattern.search(file).group(1)) >= 3:
            break
        process_nifti(file, output_path, "labels", True)


def prepare_pg_images(
    dataset_path: str, output_path: str, labeled_series: List[Tuple[int, int]]
) -> None:
    images_path = os.path.join(dataset_path, "Liver3D_originals")
    images_files = get_all_files(images_path)

    # Images are in DICom format
    print("Processing images")
    for file in tqdm(images_files):
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

        if skip_existing and os.path.exists(save_dir):
            continue

        try:
            slice_number = int(filename[3:])
        except ValueError:
            continue
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ds = pydicom.read_file(file)  # type: ignore
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


def check_processed_dataset(processed_dataset_path):
    files = get_all_files(processed_dataset_path, ".npz")
    print("Verifying processed slices")
    for file in tqdm(files):
        try:
            np.load(file)
        except (OSError, UnpicklingError, ValueError):
            print(f"Error loading {file}")


skip_existing = False
check_files = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dataset", type=str, choices=["lits", "pg"], help="Dataset to process"
    )
    parser.add_argument("input", type=str, help="Dataset input directory")
    parser.add_argument("output", type=str, help="Prepared dataset output directory")
    parser.add_argument(
        "-s",
        action="store_true",
        help="Skip processing existing images and "
        "labels (checks only if dir exists)",
    )
    parser.add_argument("-c", action="store_true", help="Test processed files loading")
    args = parser.parse_args()

    if args.s:
        skip_existing = True
        print("Skipping files already having a directory in output directory")

    if args.c:
        check_files = True
        print("Processed file checking enabled")

    print(skip_existing)

    if args.dataset == "lits":
        prepare_lits_dataset(args.input, args.output)
    elif args.dataset == "pg":
        prepare_pg_dataset(args.input, args.output)

    if check_files:
        check_processed_dataset(args.output)

    # prepare_lits_dataset(
    #     "C:\\PG\\tomografia_pg\\liver_test",
    #     "C:\\PG\\tomografia_pg\\liver_test_proc"
    # )

    # prepare_pg_dataset(
    #     "C:\\PG\\tomografia_pg\\pg",
    #     "C:\\PG\\tomografia_pg\\pg_prepared"
    # )
