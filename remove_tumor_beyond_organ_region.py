import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join, isfile, maybe_mkdir_p, subfiles

def remove_tumor_beyond_region(pred_folder):
    pred_list = glob(pred_folder+"/*.nii.gz")
    print(f"total pred file:{len(pred_list)}")
    for pred in tqdm(pred_list):
        pred_itk = sitk.ReadImage(pred)
        pred_np = sitk.GetArrayFromImage(pred_itk)
        final_pred_np = np.zeros_like(pred_np)
        organ_indices = np.where((pred_np>0) & (pred_np<14))
        xmin = np.min(organ_indices[0])
        xmax = np.max(organ_indices[0])
        ymin = np.min(organ_indices[1])
        ymax = np.max(organ_indices[1])
        zmin = np.min(organ_indices[2])
        zmax = np.max(organ_indices[2])
        final_pred_np[xmin:xmax, ymin:ymax, zmin:zmax] = pred_np[xmin:xmax, ymin:ymax, zmin:zmax]
        final_pred_itk = sitk.GetImageFromArray(final_pred_np)
        final_pred_itk.CopyInformation(pred_itk)
        sitk.WriteImage(final_pred_itk, pred.replace("_best","_best_tumor"))
        
if __name__ == '__main__':
    pred_folder = "/data/liupeng/nnUNet-master/DATASET/nnUNet_raw/Dataset002_FLARE2023/imagesTs_pred/mergeprob_fold_1_less14train_ep1450_best"
    remove_tumor_beyond_region(pred_folder=pred_folder)