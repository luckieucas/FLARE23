import os
import shutil
from pathlib import Path
from glob import glob

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def make_out_dirs(dataset_id: int, task_name="FLARE2023"):
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
    patients_train = src_data_folder
    patients_test = src_data_folder

    num_training_cases = 0
    # Copy training files and corresponding labels.
    for file in glob(str(patients_train)+"/*/*gz"):
        _, file_name = os.path.split(file)
        if "0000" in file_name:
            # The stem is 'patient.nii', and the suffix is '.gz'.
            # We split the stem and append _0000 to the patient part.
            save_name = train_dir / file_name
            print(f"saving image to {save_name}")
            shutil.copy(file, save_name)
            print(f"train file:{file}")
            num_training_cases += 1
        else:
            print(f"mask file:{file}")
            save_name = labels_dir / file_name
            print(f"saving mask to {save_name}")
            shutil.copy(file, save_name)

    # Copy test files.
    
    # print(f"patients_test:{patients_test}")
    # for file in patients_test.iterdir():
    #     if file.suffix == ".gz" :
    #         shutil.copy(file, test_dir / file.name)
    #         print(f"test file:{file}")

    return num_training_cases


def convert_flare(src_data_folder: str, dataset_id=27):
    out_dir, train_dir, labels_dir, test_dir = make_out_dirs(dataset_id=dataset_id,
                                                             task_name="AbdomenCT")
    num_training_cases = copy_files(Path(src_data_folder), train_dir, labels_dir, test_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "CT",
        },
        labels={
            "background": 0,
            "Liver": 4,
            "Right Kidney": 2,
            "Left Kidney": 3,
            "Spleen": 1,
            "Pancreas": 5
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
        help="The downloaded  dataset dir. Should contain extracted 'training' and 'testing' folders.",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=2, help="nnU-Net Dataset ID, default: 2"
    )
    args = parser.parse_args()
    print("Converting...")
    convert_flare(args.input_folder, args.dataset_id)
    print("Done!")
