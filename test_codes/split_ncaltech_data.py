from pathlib import Path
import shutil
import numpy as np
import time

np.random.seed(42)

raw_image_files_folder = Path('Caltech101_raw/Caltech101/')
# raw_annotation_files_folder = Path('Caltech101_raw/Caltech101_annotations/')

dst_dataset_folder = Path('ncaltech101')

classes = [f.name for f in raw_image_files_folder.iterdir() if f.is_dir()]

train_ratio = 0.80
val_ratio = 0.10

# classes = classes[:1]

for classe in classes:

    training_path = dst_dataset_folder / "training" / classe
    val_path = dst_dataset_folder / "validation" / classe
    test_path = dst_dataset_folder / "test" / classe

    training_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    datas = [f for f in (raw_image_files_folder / classe).iterdir() if f.is_file()]

    end_train = int(train_ratio * len(datas))
    end_val = int((train_ratio + val_ratio) * len(datas))

    # print(datas[:end_train])

    datas_train = datas[:end_train]
    datas_val = datas[end_train:end_val]
    datas_test = datas[end_val:]

    for k in datas_train:
        shutil.copy(k, training_path / k.name)
        # annot = Path(str(k).replace('image','annotation'))
        # annot_file = raw_annotation_files_folder / classe / annot.name
        # shutil.copy(annot_file, training_path / annot_file.name)

    for k in datas_val:
        shutil.copy(k, val_path / k.name)
        # annot = Path(str(k).replace('image','annotation'))
        # annot_file = raw_annotation_files_folder / classe / annot.name
        # shutil.copy(annot_file, val_path / annot_file.name)
    
    for k in datas_test:
        shutil.copy(k, test_path / k.name)
        # annot = Path(str(k).replace('image','annotation'))
        # annot_file = raw_annotation_files_folder / classe / annot.name
        # shutil.copy(annot_file, test_path / annot_file.name)

    print(f"Classe {classe}: finish")

