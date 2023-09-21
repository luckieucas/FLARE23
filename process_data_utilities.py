import os 
import shutil
from tqdm import tqdm
from nnunetv2.paths import nnUNet_raw,nnUNet_preprocessed,nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import (load_json,
                                                    join, isfile)

def remove_npy_npz_data_from_preprocessed_folder(plans="nnUNetPlans_3d_midres"):
    """ remove npy and npz data from preprocessed folder in order to 
        reprocess them
    """
    partial_type = "14"
    partial_dict = load_json(join(nnUNet_preprocessed,'Dataset002_FLARE2023',
                                  'different_partial_type_backup.json'))
    partial_type_cases = partial_dict[partial_type]
    print(f"Total cases: {len(partial_type_cases)}")
    for case in tqdm(partial_type_cases):
        print(f"removing {case}")
        npy_data = join(nnUNet_preprocessed,'Dataset002_FLARE2023',
                        plans,case+".npy")
        npy_seg_data = join(nnUNet_preprocessed,'Dataset002_FLARE2023',
                        plans,case+"_seg.npy")
        npz_data = join(nnUNet_preprocessed,'Dataset002_FLARE2023',
                        plans,case+".npz")
        pkl_data = join(nnUNet_preprocessed,'Dataset002_FLARE2023',
                        plans,case+".pkl")
        try:
            os.remove(npy_data)
            os.remove(npy_seg_data)
            os.remove(npz_data)
            os.remove(pkl_data)
        except FileNotFoundError:
            print(f"file does not exist:{case}")


if __name__ == '__main__':
    plans = "nnUNetPlansSmall_3d_verylowres"
    remove_npy_npz_data_from_preprocessed_folder(plans=plans)