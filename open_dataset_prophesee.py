import numpy as np
from prophesee_automotive_dataset_toolbox.src.io.psee_loader import PSEELoader

#Load a sample from .dat file
path = "trainfilelist14/train/moorea_2019-02-19_004_td_244500000_304500000_td.dat"
print(f'path : {path}')

label = np.load("trainfilelist14/train/moorea_2019-02-19_004_td_244500000_304500000_bbox.npy")

video = PSEELoader(path)

print(video)
video.event_count()
video.total_time()

print(label)