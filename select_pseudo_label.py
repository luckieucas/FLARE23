import numpy as np 
import SimpleITK as sitk
from tqdm import tqdm
from nnunetv2.paths import nnUNet_results,nnUNet_raw,nnUNet_preprocessed
from batchgenerators.utilities.file_and_folder_operations import join,load_json,save_json,maybe_mkdir_p,isfile


def select_pseudo_label():
    """
        compute uncertainty for pseudo label
    """
    partial_type = '1_2_3_4_5_6_7_8_9_10_11_12_13'
    partial_dict = load_json(join(nnUNet_preprocessed,'Dataset002_FLARE2023','different_partial_type_with_unsup.json'))
    partial_type_cases = partial_dict[partial_type]
    pseudo_label_path1 = join(nnUNet_results,"Dataset002_FLARE2023/nnUNetTrainerFlareMergeProb__nnUNetPlansSmall__3d_verylowres/fold_all/validation")
    pseudo_label_path2 = join(nnUNet_results,"Dataset002_FLARE2023/nnUNetTrainerFlareMergeProb__nnUNetPlans__3d_midres/fold_all/validation_fold1")
    pseudo_label_path3 = join(nnUNet_results,"Dataset002_FLARE2023/nnUNetTrainerFlareMergeProb__nnUNetPlans__3d_midres/fold_all/unlabeled_data_pred_by_fold_all")
    u_dict = {}
    for file_name in tqdm(partial_type_cases):
        print(f"compute uncertainty for file: {file_name}")
        itk_image_ite1 = sitk.ReadImage(join(pseudo_label_path1,file_name+".nii.gz"))
        data_npy_ite1 = sitk.GetArrayFromImage(itk_image_ite1)
        #data_npy_ite1[data_npy_ite1<14] = 0 #compute uncertatainty for tumor
        itk_image_ite2 = sitk.ReadImage(join(pseudo_label_path2,file_name+".nii.gz"))
        data_npy_ite2 = sitk.GetArrayFromImage(itk_image_ite2)
        #data_npy_ite2[data_npy_ite2<14] = 0 #compute uncertatainty for tumor
        itk_image_ite3 = sitk.ReadImage(join(pseudo_label_path3,file_name+".nii.gz"))
        data_npy_ite3 = sitk.GetArrayFromImage(itk_image_ite3)
        #data_npy_ite3[data_npy_ite3<14] = 0 #compute uncertatainty for tumor
        uncertainty2 = np.sum(data_npy_ite1 != data_npy_ite2)/np.sum(data_npy_ite2>0)
        uncertainty3 = np.sum(data_npy_ite2 != data_npy_ite3)/np.sum(data_npy_ite3>0)
        u = (uncertainty2+uncertainty3)/2
        print(file_name)
        print(u)
        u_dict[file_name] = u
    u_order=dict(sorted(u_dict.items(),key=lambda x:x[1],reverse=True))
    print(u_order)
    save_json(u_order,join(nnUNet_preprocessed,"Dataset002_FLARE2023",'tumor_pseudo_uncertainty_for_1_2_3_4_5_6_7_8_9_10_11_12_13.json'),sort_keys=False)
    
def merge_pseudo_label():
    partial_type = '0'
    partial_dict = load_json(join(nnUNet_preprocessed,'Dataset002_FLARE2023','different_partial_type_with_unsup.json'))
    partial_type_cases = partial_dict[partial_type]
    pseudo_label_path1 = join(nnUNet_results,"Dataset002_FLARE2023/nnUNetTrainerFlareMergeProb__nnUNetPlansSmall__3d_verylowres/fold_all/validation")
    pseudo_label_path2 = join(nnUNet_raw,"Dataset002_FLARE2023/aladdin5-pseudo-labels-FLARE23")
    pseudo_label_path3 = join(nnUNet_raw,"Dataset002_FLARE2023/blackbean-pseudo-labels-FLARE23")
    pseudo_label_midres_fold_all = join(nnUNet_results,"Dataset002_FLARE2023/nnUNetTrainerFlareMergeProb__nnUNetPlans__3d_midres/fold_all/unlabeled_data_pred_by_fold_all")
    pseudo_label_midres = join(nnUNet_results,"Dataset002_FLARE2023/nnUNetTrainerFlareMergeProb__nnUNetPlans__3d_midres/fold_all/validation_fold1")
    #gt_label_path = join(nnUNet_raw,"Dataset002_FLARE2023","labelsTr")
    for file_name in tqdm(partial_type_cases):
        save_path = join(nnUNet_raw,"labelsTr_unsup_final_pseudo_label",
                         file_name+".nii.gz")
        if isfile(save_path):
            continue
        midres_pred = sitk.ReadImage(join(pseudo_label_midres,file_name+".nii.gz"))
        midres_pred_np = sitk.GetArrayFromImage(midres_pred)
        organ_indices = np.where((midres_pred_np>0) & (midres_pred_np<14))
        final_label = np.zeros_like(midres_pred_np)
        
        #gt_label = sitk.GetArrayFromImage(sitk.ReadImage(join(gt_label_path,file_name+".nii.gz")))
        #final_label = np.zeros_like(gt_label)
        verylowres_pred = sitk.ReadImage(join(pseudo_label_path1,file_name+".nii.gz"))
        verylowres_pred_np = sitk.GetArrayFromImage(verylowres_pred)
        midres_fold_all_pred = sitk.ReadImage(join(pseudo_label_midres_fold_all,file_name+".nii.gz"))
        midres_fold_all_pred_np = sitk.GetArrayFromImage(midres_fold_all_pred)
        itk_image_ite2 = sitk.ReadImage(join(pseudo_label_path2,file_name+".nii.gz"))
        data_npy_ite2 = sitk.GetArrayFromImage(itk_image_ite2)
        itk_image_ite3 = sitk.ReadImage(join(pseudo_label_path3,file_name+".nii.gz"))
        data_npy_ite3 = sitk.GetArrayFromImage(itk_image_ite3)
        try:
            xmin = np.min(organ_indices[0])
            xmax = np.max(organ_indices[0])
            ymin = np.min(organ_indices[1])
            ymax = np.max(organ_indices[1])
            zmin = np.min(organ_indices[2])
            zmax = np.max(organ_indices[2])
            mask = (midres_pred_np==14)&(data_npy_ite2>0)
            midres_pred_np[mask] = data_npy_ite2[mask]
            mask = (midres_pred_np==14)&(data_npy_ite3>0)
            midres_pred_np[mask] = data_npy_ite3[mask]
            mask = (verylowres_pred_np == midres_fold_all_pred_np)
            midres_pred_np[mask] = midres_fold_all_pred_np[mask]
            #midres_pred_np[midres_pred_np==14] = 0
            #midres_pred_np[gt_label==14] = 14
            # mask = (midres_pred_np==14)&(data_npy_ite3==0)&(data_npy_ite2==0)
            # midres_pred_np[mask] = 0
            final_label[xmin:xmax,ymin:ymax,zmin:zmax] = midres_pred_np[xmin:xmax,ymin:ymax,zmin:zmax]
            final_label_itk = sitk.GetImageFromArray(final_label)
            final_label_itk.CopyInformation(itk_image_ite2)
            maybe_mkdir_p(join(nnUNet_raw,"labelsTr_unsup_final_pseudo_label"))
            sitk.WriteImage(final_label_itk,save_path)
        except:
            print(f"error get pseudo label {file_name}")

if __name__ == '__main__':
    #merge_pseudo_label()
    select_pseudo_label()