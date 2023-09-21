import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join, isfile, maybe_mkdir_p, subfiles



def merge_organ_tumor(organ_pred_folder, tumor_pred_folder, merged_save_folder):
    files_organ = subfiles(organ_pred_folder, suffix=".nii.gz", join=False)
    files_tumor = subfiles(tumor_pred_folder, suffix=".nii.gz", join=False)
    for organ, tumor in tqdm(zip(files_organ,files_tumor), total=len(files_tumor)):
        print(f"processing organ: {organ}, tumor: {tumor}")
        merged_save_path = join(merged_save_folder, organ)
        if isfile(merged_save_path):
            print(f"already merged:{organ}")
            continue
        organ_file = join(organ_pred_folder,organ)
        tumor_file = join(tumor_pred_folder,tumor)
        organ_itk = sitk.ReadImage(organ_file)
        tumor_itk = sitk.ReadImage(tumor_file)
        organ_np = sitk.GetArrayFromImage(organ_itk)
        tumor_np = sitk.GetArrayFromImage(tumor_itk)
        # remove tumor pred in organ model
        organ_np[organ_np==14] = 0
        
        # use tumor pred to fill organ pred
        organ_np[tumor_np==1] = 14
        
        merged_itk = sitk.GetImageFromArray(organ_np)
        merged_itk.CopyInformation(organ_itk)
        sitk.WriteImage(merged_itk, merged_save_path)


if __name__ == "__main__":
    organ_pred_folder = "/data/xining.zq/flare_nnunet/ret_v2/nnUNet_results_semi_lowres_mergemax/Dataset002_FLARE2023/nnUNetTrainerFlareSemiTumor__nnUNetPlans__3d_lowres/fold_all/pred_latest/ret"
    tumor_pred_folder = "/data/liupeng/nnUNet-master/DATASET/nnUNet_raw/Dataset002_FLARE2023/imagesTs_pred_tumor"
    merged_save_folder = "/data/liupeng/nnUNet-master/DATASET/nnUNet_raw/Dataset002_FLARE2023/imagesTs_pred_organ_tumor_merged"
    maybe_mkdir_p(merged_save_folder)
    merge_organ_tumor(organ_pred_folder=organ_pred_folder,
                      tumor_pred_folder=tumor_pred_folder,
                      merged_save_folder=merged_save_folder)