import argparse
import os
from tqdm import tqdm

import numpy as np

from src.prepare_dataset import load_metadata

size = 512 * 512


def calc_percentages(row):
    image = np.load(os.path.join(args.dataset, row["label_path"]))["arr_0"]
    s = np.unique(image, return_counts=True)
    counts = s[1]
    if len(counts) == 1:
        return counts[0] / size, 0, 0
    if len(counts) == 2:
        return counts[0] / size, counts[1] / size, 0
    if len(counts) == 3:
        return (
            counts[0] / size,
            counts[1] / size,
            counts[2] / size,
        )


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("metadata", type=str, help="Metadata path")
parser.add_argument("dataset", type=str, help="Dataset path")
args = parser.parse_args()

metadata = load_metadata(args.metadata)

tqdm.pandas()

metadata["percentages"] = metadata.progress_apply(lambda r: calc_percentages(r), axis=1)
metadata["background_percent"] = metadata.progress_apply(
    lambda r: r["percentages"][0], axis=1
)
metadata["liver_percent"] = metadata.progress_apply(
    lambda r: r["percentages"][1], axis=1
)
metadata["tumor_percent"] = metadata.progress_apply(
    lambda r: r["percentages"][2], axis=1
)
metadata.drop("percentages", axis=1, inplace=True)

metadata.to_csv(
    os.path.join(os.path.dirname(args.metadata), "metadata_percentages.csv"),
    index_label="id",
)
