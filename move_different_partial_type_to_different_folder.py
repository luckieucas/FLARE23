import os
import shutil
from tqdm import tqdm
from nnunetv2.paths import nnUNet_raw,nnUNet_preprocessed
from batchgenerators.utilities.file_and_folder_operations import load_json,maybe_mkdir_p,join,subfiles,isfile



def move_different_partial_type_to_different_folder(partial_type='14'):
    partial_dict = load_json(join(nnUNet_preprocessed,'Dataset002_FLARE2023','different_partial_type.json'))
    partial_type_cases = partial_dict[partial_type]
    for case in tqdm(partial_type_cases):
        src_path = join(nnUNet_raw,'Dataset002_FLARE2023','imagesTr',case+"_0000.nii.gz")
        assert isfile(src_path), f"file not exists: {src_path}"
        dst_path = join(nnUNet_raw,'Dataset002_FLARE2023',f'imagesTr_{partial_type}',case+"_0000.nii.gz")
        maybe_mkdir_p(os.path.split(dst_path)[0])
        shutil.copy(src_path,dst_path)
        print(case)


if __name__ == "__main__":
    move_different_partial_type_to_different_folder(partial_type='14')