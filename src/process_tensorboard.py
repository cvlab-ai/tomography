from packaging import version
import pandas as pd
from scipy import stats
import tensorboard as tb
import re

pd.set_option('display.max_colwidth', None)

major_ver, minor_ver, _ = version.parse(tb.__version__).release
assert major_ver >= 2 and minor_ver >= 3, \
    "This notebook requires TensorBoard 2.3 or later."
print("TensorBoard version: ", tb.__version__)

experiment_id = "B8ldWQ8hRXevTVkULSR8ZQ"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()

last_step = df['step'].max()

df = df[df["run"].str.contains("_single-window") & (df["step"].eq(last_step) | df["tag"].str.contains("test"))]
# df = df[df["run"].str.contains("_multi-window") & (df["step"].eq(last_step) | df["tag"].str.contains("test"))]

df = df.loc[
     df["tag"].str.contains("test-dice")
     | df["tag"].str.contains("window")
]

ww_regex = re.compile(r"ww-\[(-?(\d+)?,?)\]")
wl_regex = re.compile(r"wl-\[(-?(\d+)?,?)\]")
#ww_regex = re.compile(r"ww-\[((-?\d*,? ?)*)\]")
#wl_regex = re.compile(r"wl-\[((-?\d*,? ?)*)\]")

fold_regex = re.compile(r"fold-((\d+)?,?)")

df = df.pivot(index="run", columns="tag", values="value")

df["window_width_init"] = df.apply(lambda row: ww_regex.search(row.name).group(1), axis=1)
df["window_center_init"] = df.apply(lambda row: wl_regex.search(row.name).group(1), axis=1)
df["fold"] = df.apply(lambda row: fold_regex.search(row.name).group(1), axis=1)

df = df[["fold", "window_width_init", "window_center_init", "test-dice", "test-dice_liver", "test-dice_tumor", "window-0-width", "window-0-center"]]
# df = df[["fold", "window_width_init", "window_center_init", "test-dice", "test-dice_liver", "test-dice_tumor", "window-0-width", "window-0-center", "window-1-width", "window-1-center", "window-2-width", "window-2-center", "window-3-width", "window-3-center", "window-4-width", "window-4-center"]]

df.to_excel("out.xlsx")
