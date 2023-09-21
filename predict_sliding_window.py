'''
Descripttion: test model
version: 
Author: Luckie
Date: 2021-04-19 13:09:02
LastEditors: Luckie
LastEditTime: 2021-06-09 15:00:23
'''
from email.policy import default
import os 
import time
import shutil
import torch
from torch.nn import functional as F
import math
import numpy as np
import SimpleITK as sitk
from medpy import metric
from tqdm import tqdm
from random import shuffle
from typing import OrderedDict
import yaml
import argparse
import nnunetv2
from glob import glob
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Compose,LoadImage,ToTensor
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from batchgenerators.utilities.file_and_folder_operations import save_json
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2.utilities.binary as binary
from skimage.transform import resize
from batchgenerators.augmentations.utils import resize_segmentation

task_name_id_dict={"full":0,"spleen":1,"kidney":2,"liver":4,"pancreas":5,
                   "heart":0}
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='test_config_new.yaml', 
                    help='training configuration'
)
parser.add_argument('--gpu', type=str, default='3',help='gpu id for testing')
parser.add_argument(
    '--model_path', type=str,
    default='/data/liupeng/semi-supervised_segmentation/SSL4MIS-master/model/BCV_500_CVCL_partial_test_vnet_cvcl_SGD_SGD/vnet/model_iter_18000_dice_0.8605.pth',
    help='model path for testing'
)


class_id_name_dict = {
    'MMWHS':['MYO', 'LA', 'LV', 'RA', 'AA', 'PA', 'RV'],
    'BCV':['Spleen', 'Right Kidney', 'Left Kidney','Liver','Pancreas'],
    'LA':['LA']
}



def process_fn(seg_prob_tuple, window_data, importance_map_):
    """seg_prob_tuple, importance_map = 
    process_fn(seg_prob_tuple, window_data, importance_map_)
    """
    if len(seg_prob_tuple)>0 and isinstance(seg_prob_tuple, (tuple, list)):
        seg_prob = torch.softmax(seg_prob_tuple[0],dim=1)
        return tuple(seg_prob.unsqueeze(0))+seg_prob_tuple[1:],importance_map_
    else:
        seg_prob = torch.softmax(seg_prob_tuple,dim=1)
        return seg_prob,importance_map_

def compute_new_shape(old_shape, old_spacing, new_spacing) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape

def resize3d(image, shape, mode='trilinear', data_type='numpy'):
    """
        resize 3d image
    """
    if data_type == 'numpy':
        image = torch.tensor(image)[None,None,:,:,:]
    image = F.interpolate(torch.as_tensor(image), size=shape, mode=mode)
    image = image[0,0,:,:,:].numpy()
    return image

def load_trained_model_for_inference(model_training_output_dir, use_folds, checkpoint_name):
    # we could also load plans and dataset_json from the init arguments in the checkpoint. Not quite sure what is the
    # best method so we leave things as they are for the moment.
    dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
    plans = load_json(join(model_training_output_dir, 'plans.json'))
    plans_manager = PlansManager(plans)

    if isinstance(use_folds, str):
        use_folds = [use_folds]

    parameters = []
    for i, f in enumerate(use_folds):
        f = int(f) if f != 'all' else f
        checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                map_location=torch.device('cpu'))
        if i == 0:
            trainer_name = checkpoint['trainer_name']
            configuration_name = checkpoint['init_args']['configuration']
            inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        parameters.append(checkpoint['network_weights'])

    configuration_manager = plans_manager.get_configuration(configuration_name)
    # restore network
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    network = trainer_class.build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                                       num_input_channels, enable_deep_supervision=False)
    return parameters, configuration_manager, inference_allowed_mirroring_axes, plans_manager, dataset_json, network, trainer_name

def get_model(model_training_output_dir, use_folds, checkpoint_name):
    parameters, configuration_manager, inference_allowed_mirroring_axes, \
    plans_manager, dataset_json, network, trainer_name = \
        load_trained_model_for_inference(model_training_output_dir, use_folds, checkpoint_name)
    network.load_state_dict(parameters[0])
    return network,plans_manager,configuration_manager

def normalize_data(data,cut_lower, cut_upper, mean, std):
    data = np.clip(data, cut_lower, cut_upper)
    data = (data - mean) / max(std, 1e-8)
    return data

def test_all_case_monai(net, configuration_manager, test_list="full_test.list", num_classes=4, 
                      patch_size=(48, 160, 160),overlap=0.5,
                      condition=-1,method="regular",cal_hd95=False,
                      cal_asd=False,intensityproperties=None, 
                      save_prediction=False,prediction_save_path='./',
                      class_name_list=[],
                      has_gt=False):
    if isinstance(test_list, str):
        with open(test_list, 'r') as f:
            image_list = [img.replace('\n','') for img in f.readlines()]
    else:
        image_list = test_list
    cut_lower = intensityproperties['percentile_00_5']
    cut_upper = intensityproperties['percentile_99_5']
    target_spacing = configuration_manager.spacing
    mean = intensityproperties['mean']
    std = intensityproperties['std']
    print("***************************validation begin************************")
    with torch.no_grad():
        for i, image_path in enumerate(tqdm(image_list)):
            if i < 4:
                continue
            _,img_name = os.path.split(image_path)
            print(f"=============>processing {img_name}")
            
            image_sitk = sitk.ReadImage(image_path)
            spacing = image_sitk.GetSpacing()[::-1]
            image = sitk.GetArrayFromImage(image_sitk)
            image_shape = image.shape
            image_new_shape = compute_new_shape(image_shape, 
                                                spacing, target_spacing)
            print(f"original shape:{image_shape}, new shape: {image_new_shape}")
            image = normalize_data(image, cut_lower, cut_upper, mean, std)
            print(f"image max: {image.max()}, image min:{image.min()}")
            new_image = resize(image, image_new_shape)
            print(f"new image max: {new_image.max()}, new image min:{new_image.min()},new image mean:{new_image.mean()}")
            #new_image = resize3d(image, tuple(image_new_shape))
            new_image = torch.from_numpy(new_image).unsqueeze(0).unsqueeze(0).cuda()
            prediction = sliding_window_inference(
                new_image.float(),patch_size,1,net,overlap=overlap)
            
            # prediction = F.interpolate(prediction, 
            #                            size=image_shape, mode='trilinear')
            print(f"prediction max:{prediciton.max()}, min:{prediction.min()}, mean:{prediction.mean()}")
            prediction = torch.argmax(prediction,dim=1).squeeze().type(torch.FloatTensor)
            prediction = F.interpolate(prediction[None,None,:,:,:], 
                                        size=image_shape, mode='nearest').squeeze().numpy()
            if save_prediction: 
                pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
                pred_itk.CopyInformation(image_sitk)
                sitk.WriteImage(
                    pred_itk, 
                    prediction_save_path+img_name.replace("_0000.nii.gz",".nii.gz")
                )
    
    print("***************************validation end**************************")
    return 


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    class_name_list = ['background', 'Liver', 'Right Kidney', 'Spleen', 
                       'Pancreas', 'Aorta', 'Inferior vena cava', 
                       'Right adrenal gland', 'Left adrenal gland', 
                       'Gallbladder', 'Esophagus', 'Stomach', 'Duodenum', 
                       'Left Kidney', 'Tumor']
    use_folds = (1,)
    checkpoint_name = "checkpoint_ep1900.pth"
    model_training_output_dir = "/data/liupeng/nnUNet-master/DATASET/nnUNet_results/Dataset002_FLARE2023/nnUNetTrainerFlare__nnUNetPlans__3d_midres/"
    input_folder = "/data/liupeng/nnUNet-master/DATASET/nnUNet_raw/Dataset002_FLARE2023/imagesTs"
    output_folder = "/data/liupeng/nnUNet-master/DATASET/nnUNet_raw/Dataset002_FLARE2023/imagesTs_pred_debug/"
    shutil.copy(join(model_training_output_dir, 'dataset.json'), join(output_folder, 'dataset.json'))
    shutil.copy(join(model_training_output_dir, 'plans.json'), join(output_folder, 'plans.json'))
    model,plans_manager,configuration_manager = get_model(model_training_output_dir, use_folds, checkpoint_name)
    model = model.cuda()
    model.eval()
    intensityproperties = plans_manager.foreground_intensity_properties_per_channel['0']
    test_list = glob(input_folder+"/*.nii.gz")
    
    test_all_case_monai(
                        model,
                        configuration_manager,
                        test_list=sorted(test_list),
                        num_classes=15, 
                        patch_size=[96,128,160],
                        overlap=0.2,
                        cal_hd95=False,
                        cal_asd=True,
                        intensityproperties=intensityproperties,
                        save_prediction=True,
                        prediction_save_path=output_folder,
                        class_name_list=class_name_list,
                        has_gt=False
                    )
    # print(avg_metric)
    # print(avg_metric[:, 0].mean(),avg_metric[:,1].mean(), avg_metric[:,2].mean())
