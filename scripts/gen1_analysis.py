import numpy as np

from prophesee_toolbox.io.dat_events_tools import load_td_data

from pathlib import Path


path = "./data/test_a/detection_dataset_duration_60s_ratio_1.0/test"
txt_file_path = Path("./data/test_a/detection_dataset_duration_60s_ratio_1.0/test/dataset_information.txt")

txt_file_path.unlink(missing_ok=True)

nb_events = 50000

files = [p for p in Path(path).rglob("*.npy") if p.is_file()]

for file in files:

    print(f"File : {file.name}")

    bboxes = np.load(file)
    unique_ts = np.unique(bboxes['ts'])

    dat_file = file.with_name(file.name.replace("bbox.npy", "td.dat"))

    events = load_td_data(dat_file)

    high = np.searchsorted(events['t'], unique_ts, side = "right")-1

    for i in range(len(unique_ts)):

        with open(txt_file_path, 'a') as f:
            f.write(f"{file.name} {unique_ts[i]} {high[i]} \n")

    print(f"Finish")