import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p


def make_out_dirs(dataset_id: int, task_name="FLARE2023TumorCrop"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    out_train_dir = out_dir / "imagesTr"
    out_labels_dir = out_dir / "labelsTr"
    out_test_dir = out_dir / "imagesTs"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    return out_dir, out_train_dir, out_labels_dir, out_test_dir


def copy_files(src_data_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path):
    """Copy files from the ACDC dataset to the nnUNet dataset folder. Returns the number of training cases."""
    partial_type_dict = load_json("/data/liupeng/nnUNet-master/DATASET/nnUNet_preprocessed/Dataset002_FLARE2023/different_partial_type_backup.json")
    num_training_cases = 0
    for pattern_type in partial_type_dict.keys():
        if '14' not in pattern_type:
            continue
        print(f"proccesing pattern type:{pattern_type}")
        full_label_cases = partial_type_dict[pattern_type]
        # Copy training files and corresponding labels.
        for case in tqdm(sorted(full_label_cases)):
            image = src_data_folder / "organ_cropped_imgs" / f"{case}_0000.nii.gz"
            label = src_data_folder / "organ_cropped_labels" / f"{case}.nii.gz"
            # replace label 14 to 1
            # if os.path.exists(labels_dir / f"{case}.nii.gz"):
            #     print(f'label already exists: {case}')
            #     continue
            label_nii = sitk.ReadImage(label)
            data = sitk.GetArrayFromImage(label_nii)
            data[data != 14] = 0
            data[data == 14] = 1
            if data.sum() < 50:
                print(f"{case} only has pixels less than 50 : {data.sum()}")
                continue
            print(f"total pixels:{data.size}  tumor pixels:{data.sum()}, tumor ratio:{data.sum() / data.size}")
            print("label unique: ",np.unique(data))
            new_nii = sitk.GetImageFromArray(data)
            new_nii.CopyInformation(label_nii)
            sitk.WriteImage(new_nii, labels_dir / f"{case}.nii.gz")
            #shutil.copy(label, labels_dir / f"{case}.nii.gz")
            print("copy image",image)
            if os.path.exists(train_dir / f"{case}_0000.nii.gz"):
                print(f'image already exists: {case}')
            else:
                shutil.copy(image, train_dir / f"{case}_0000.nii.gz")
            print("copy label", label)
            num_training_cases += 1

    # Copy test files.
    
    # print(f"patients_test:{patients_test}")
    # for file in patients_test.iterdir():
    #     if file.suffix == ".gz" :
    #         shutil.copy(file, test_dir / file.name)
    #         print(f"test file:{file}")

    return num_training_cases


def convert_flare(src_data_folder: str, dataset_id=7):
    out_dir, train_dir, labels_dir, test_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_folder), train_dir, labels_dir, test_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "CT",
        },
        labels={
            "background": 0,
            "Tumor": 1,  
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="The downloaded FLARE2023 dataset dir. Should contain extracted 'training' and 'testing' folders.",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=7, help="nnU-Net Dataset ID, default: 2"
    )
    args = parser.parse_args()
    print("Converting...")
    convert_flare(args.input_folder, args.dataset_id)
    print("Done!")
