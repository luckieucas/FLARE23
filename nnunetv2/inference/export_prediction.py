import os
from copy import deepcopy
from typing import Union, List
import time

import numpy as np
import torch
import torch.nn.functional as F

from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle

from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

def slice_argmax(class_probabilities,step=5):
    C,Z,X,Y = class_probabilities.shape
    result = np.zeros((Z,X,Y))
    z = class_probabilities.shape[1]
    stride = int(z/step)
    step1 = [i*stride for i in range(step)]+[z]
    for i in range(step):
        slicer = class_probabilities[:,step1[i]:step1[i+1]].half()
        result[step1[i]:step1[i+1]] = torch.argmax(slicer.cuda(),0).cpu().numpy()
        del slicer
        torch.cuda.empty_cache()
    return result

def resize_and_argmax(class_probabilities,size_after_cropping,step=3): #切片resize
    c = class_probabilities.shape[0]
    reshaped = torch.zeros(15,*size_after_cropping)
    for c in range(class_probabilities.shape[0]):
        slicer = class_probabilities[c].half()
        reshaped[c] = torch.nn.functional.interpolate(slicer[None,None,:].cuda(),
                                                      mode='trilinear',
                                                      size=size_after_cropping)[0].cpu()
        del slicer
        torch.cuda.empty_cache()
    return reshaped

def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: torch.Tensor,
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False):
    #predicted_logits = predicted_logits.astype(np.float32)
    #print(f"prediction: shape:{predicted_logits.shape}, prediction type:{predicted_logits.dtype}, device:{predicted_logits.device}")
    # resample to original shape
    # current_spacing = configuration_manager.spacing if \
    #     len(configuration_manager.spacing) == \
    #     len(properties_dict['shape_after_cropping_and_before_resampling']) else \
    #     [properties_dict['spacing'][0], *configuration_manager.spacing]
    # predicted_logits_resampled1 = configuration_manager.resampling_fn_probabilities(predicted_logits,
    #                                         properties_dict['shape_after_cropping_and_before_resampling'],
    #                                         current_spacing,
    #                                         properties_dict['spacing'])
    #dtype_data = predicted_logits.dtype
    new_shape = np.array(properties_dict['shape_after_cropping_and_before_resampling'])
    predicted_logits_resampled = resize_and_argmax(predicted_logits, list(new_shape))
    #print(f"predicted_logits_resampled1 shape: {predicted_logits_resampled1.shape}")
    # if np.any(shape != new_shape):
    #     #predicted_logits = predicted_logits.astype(np.float32)
    #     if new_shape[0] < 250:
    #         # import time 
    #         # start_time = time.time()
    #         # reshaped_final_data1 = nn.functional.interpolate(torch.from_numpy(data).unsqueeze(0).type(torch.float32),size=list(new_shape),scale_factor= None , mode = 'trilinear').numpy()[0]
    #         # print(f"use cpu time:{time.time()-start_time}")
    #         #start_time = time.time()
    #         predicted_logits_resampled = F.interpolate(predicted_logits.unsqueeze(0).type(torch.float32),size=list(new_shape),scale_factor= None , mode = 'trilinear')[0]
    #         #print(f"use gpu time:{time.time()-start_time}")
    #         #np.testing.assert_almost_equal(reshaped_final_data1, reshaped_final_data, decimal=7, err_msg='', verbose=True) 
    #     else:
    #         predicted_logits_resampled = F.interpolate(predicted_logits.unsqueeze(0).type(torch.float32),size=list(new_shape))[0]
    #     #         # for c in range(data.shape[0]):
    #         #     print(f"resample data shape:{data.shape}")
    #         #     reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None])
    #         #     print(f"resample data or seg 2:{time.time() - start_time}")
    #         # reshaped_final_data = np.vstack(reshaped)
    #     #predicted_logits_resampled = predicted_logits_resampled.astype(dtype_data)
    # else:
    #     predicted_logits_resampled = predicted_logits
        
        
    
    #segmentation = torch.argmax(predicted_logits_resampled,0).numpy()
    segmentation = slice_argmax(predicted_logits_resampled)
    #segmentation1 = torch.argmax(predicted_logits_resampled1,0).numpy()
    #print(f"is equal gpu and cpu:{(segmentation!=segmentation1).sum()}")
    #print(f"predicted_logits_resampled: shape:{predicted_logits_resampled.shape}, prediction type:{predicted_logits_resampled.dtype}, device:{predicted_logits_resampled.device}")
    del predicted_logits
    del predicted_logits_resampled
    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)

    return segmentation_reverted_cropping


def export_prediction_from_logits(predicted_array_or_file: torch.Tensor, properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False):

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities
    )
    del predicted_array_or_file

    # save
    segmentation_final = ret
    
    # do post-processing
    do_postprocessing = False
    if do_postprocessing:
        print(f"do post-processing")
        from nnunetv2.inference.PostProcessing import RemoveSmallStructures,CloseStructures, \
            MaskWithVessels, RemoveConnectedComponents, FixGallblader
        import time
        start_time = time.time()
        min_volumes = {
        1: 1037420.7294452335,
        2: 107952.71351374676,
        3: 51560.09455919893,
        4: 41522.03935737655,
        5: 40533.60780851278,
        6: 41437.472288146426,
        7: 1385.5876138624267,
        8: 2445.168267657047,
        9: 4276.847870821996,
        10: 6978.699080946171,
        11: 140339.32771712207,
        12: 43614.98921844564,
        13: 102572.24069653488,
        14: 20
        }
        tic = time.time()
        # Exclude small structures
        label_spacing = properties_dict['spacing']
        voxel_volume = label_spacing[0] * label_spacing[1] * label_spacing[2]
        excluded_organs = [1, 5, 6, 10, 12, 11]
        label_remove_small = RemoveSmallStructures(
            segmentation_final, excluded_organs, 0.5, voxel_volume, min_volumes
        )
        toc = time.time()
        print(f"Excluding small structures took {toc-tic}")

        tic = time.time()
        # Close aorta, IVC, RAG, LAG
        structures_to_close = [7, 8]
        closed_label_np = CloseStructures(label_remove_small, structures_to_close)
        toc = time.time()
        print(f"Closing structures took {toc-tic}")

        tic = time.time()
        # Masking with vessels
        #masked_label_np = MaskWithVessels(closed_label_np, label_spacing[2])
        masked_label_np = closed_label_np
        toc = time.time()
        print(f"Masking with vessels took {toc-tic}")

        tic = time.time()
        # Remove connected components that are less than 5% total volume
        exclude_list_cca = [9]
        # removed_components_label_np = RemoveConnectedComponents(
        #     masked_label_np, 0.1, voxel_volume, exclude_list_cca
        # )
        removed_components_label_np = masked_label_np
        toc = time.time()
        print(f"Removing connected components took {toc-tic}")

        tic = time.time()
        # Fix gallbladder
        segmentation_final = FixGallblader(removed_components_label_np)
        toc = time.time()
        print(f"Fixing gallbladder took {toc-tic}")
        print(f"total post processing time:{time.time() - start_time}")
    
    do_postprocessing_for_tumor = False
    if do_postprocessing_for_tumor:
        import cc3d
        import fastremap
        from scipy import ndimage
        def keep_topk_largest_connected_object(npy_mask, k, area_least, 
                                               out_mask, out_label):
            labels_out = cc3d.connected_components(npy_mask, connectivity=26)
            areas = {}
            for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
                areas[label] = fastremap.foreground(extracted)
            candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)

            for i in range(min(k, len(candidates))):
                if candidates[i][1] > area_least:
                    out_mask[labels_out == int(candidates[i][0])] = out_label
        
        def extract_topk_largest_candidates(npy_mask, organ_num, area_least=0):
            ## npy_mask: w, h, d
            ## organ_num: the maximum number of connected component
            out_mask = np.zeros(npy_mask.shape, np.uint8)
            t_mask = npy_mask.copy()
            keep_topk_largest_connected_object(t_mask, organ_num, area_least, out_mask, 1)

            return out_mask
        
        def merge_and_top_organ(pred_mask, organ_list):
            ## merge 
            out_mask = np.zeros(pred_mask, np.uint8)
            for organ in organ_list:
                out_mask = np.logical_or(out_mask, pred_mask==organ)
            ## select the top k, for righr left case
            out_mask = extract_topk_largest_candidates(out_mask, len(organ_list))

            return out_mask
        
        def organ_region_filter_out(tumor_mask, organ_mask):
            ## dialtion
            organ_mask = ndimage.binary_closing(organ_mask, structure=np.ones((5,5,5)))
            organ_mask = ndimage.binary_dilation(organ_mask, structure=np.ones((5,5,5)))
            ## filter out
            tumor_mask = organ_mask * tumor_mask

            return tumor_mask

        organ_list = [1,2,4,13]
        organ_mask = merge_and_top_organ(segmentation_final, 
                                         organ_list=organ_list)
        post_pred_mask = organ_region_filter_out(segmentation_final, organ_mask)
    
    del ret

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)


def save_segmentation_nifti(predicted_array_or_file: np.ndarray, properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False):
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    
    
    predicted_logits = predicted_array_or_file.astype(np.float32)
    predicted_segmentation = predicted_logits.argmax(axis=0)

    # resample to original shape
    start_time = time.time()
    print(f"predicted_logits shape before: {predicted_logits.shape}")
    new_shape = properties_dict['shape_after_cropping_and_before_resampling']
    # predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
    #                                         properties_dict['shape_after_cropping_and_before_resampling'],
    #                                         current_spacing,
    #                                         properties_dict['spacing'])
    reshaped_final_data = F.interpolate(torch.from_numpy(predicted_segmentation).unsqueeze(0).unsqueeze(0).to(torch.float32), 
                                        size=new_shape, mode='nearest-exact',
                                        antialias=False).squeeze().numpy()
    print(f"segmenation shape: {reshaped_final_data.shape}")
    print(f"label manager time: {time.time() - start_time}")
    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = reshaped_final_data
    
    print(f"cropping time: {time.time() - start_time}")

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    
    del predicted_array_or_file


    segmentation_final = segmentation_reverted_cropping

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)


def resample_and_save(predicted: Union[str, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str]) -> None:
    # needed for cascade
    if isinstance(predicted, str):
        assert isfile(predicted), "If isinstance(segmentation_softmax, str) then " \
                                  "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(predicted)
        predicted = np.load(predicted)
        os.remove(del_file)

    predicted = predicted.astype(np.float32)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)

    np.savez_compressed(output_file, seg=segmentation.astype(np.uint8))
