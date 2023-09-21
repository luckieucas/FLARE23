#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import argparse
import torch

from nnunetv2.inference.predict import predict_from_folder
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_busy
from batchgenerators.utilities.file_and_folder_operations import join, isdir
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                     " order (same as training). Files must be named "
                                                     "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                     "identifier (0000, 0001, etc)", required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')

    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')

    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')

    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')

    parser.add_argument('-f', '--folds', nargs='+', default='None',
                        help="folds to use for prediction. Default is None which means that folds will be detected "
                             "automatically in the model output folder")

    parser.add_argument('-z', '--save_npz', required=False, action='store_true',
                        help="use this if you want to ensemble these predictions with those of other models. Softmax "
                             "probabilities will be saved as compressed numpy arrays in output_folder and can be "
                             "merged between output_folders with nnUNet_ensemble_predictions")

    parser.add_argument('-l', '--lowres_segmentations', required=False, default='None',
                        help="if model is the highres stage of the cascade then you can use this folder to provide "
                             "predictions from the low resolution 3D U-Net. If this is left at default, the "
                             "predictions will be generated automatically (provided that the 3D low resolution U-Net "
                             "network weights are present")

    parser.add_argument("--part_id", type=int, required=False, default=0, help="Used to parallelize the prediction of "
                                                                               "the folder over several GPUs. If you "
                                                                               "want to use n GPUs to predict this "
                                                                               "folder you need to run this command "
                                                                               "n times with --part_id=0, ... n-1 and "
                                                                               "--num_parts=n (each with a different "
                                                                               "GPU (for example via "
                                                                               "CUDA_VISIBLE_DEVICES=X)")

    parser.add_argument("--num_parts", type=int, required=False, default=1,
                        help="Used to parallelize the prediction of "
                             "the folder over several GPUs. If you "
                             "want to use n GPUs to predict this "
                             "folder you need to run this command "
                             "n times with --part_id=0, ... n-1 and "
                             "--num_parts=n (each with a different "
                             "GPU (via "
                             "CUDA_VISIBLE_DEVICES=X)")

    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, help=
    "Determines many background processes will be used for data preprocessing. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 6")

    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
    "Determines many background processes will be used for segmentation export. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 2")

    parser.add_argument("--disable_tta", required=False, default=False, action="store_true",
                        help="set this flag to disable test time data augmentation via mirroring. Speeds up inference "
                             "by roughly factor 4 (2D) or 8 (3D)")

    parser.add_argument("--overwrite_existing", required=False, default=False, action="store_true",
                        help="Set this flag if the target folder contains predictions that you would like to overwrite")

    parser.add_argument("--mode", type=str, default="normal", required=False, help="Hands off!")
    parser.add_argument("--all_in_gpu", type=str, default="None", required=False, help="can be None, False or True. "
                                                                                       "Do not touch.")
    parser.add_argument("--step_size", type=float, default=0.5, required=False, help="don't touch")
    # parser.add_argument("--interp_order", required=False, default=3, type=int,
    #                     help="order of interpolation for segmentations, has no effect if mode=fastest. Do not touch this.")
    # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
    #                     help="order of interpolation along z is z is done differently. Do not touch this.")
    # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
    #                     help="force_separate_z resampling. Can be None, True or False, has no effect if mode=fastest. "
    #                          "Do not touch this.")
    parser.add_argument('-chk',
                        help='checkpoint name, default: model_final_checkpoint',
                        required=False,
                        default='model_final_checkpoint')
    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False,
                        help='Predictions are done with mixed precision by default. This improves speed and reduces '
                             'the required vram. If you want to disable mixed precision you can set this flag. Note '
                             'that this is not recommended (mixed precision is ~2x faster!)')

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    part_id = args.part_id
    num_parts = args.num_parts
    folds = args.folds
    save_npz = args.save_npz
    lowres_segmentations = args.lowres_segmentations
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    disable_tta = args.disable_tta
    step_size = args.step_size
    # interp_order = args.interp_order
    # interp_order_z = args.interp_order_z
    # force_separate_z = args.force_separate_z
    overwrite_existing = args.overwrite_existing
    mode = args.mode
    all_in_gpu = args.all_in_gpu
    model = args.c

    assert model in ["2d", "3d_lowres", "3d_fullres","3d_midres",
                     "3d_verylowres", "3d_cascade_fullres"], "-m must be 2d, 3d_lowres, 3d_fullres or " \
                                                                             "3d_cascade_fullres"

    if lowres_segmentations == "None":
        lowres_segmentations = None

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    assert all_in_gpu in ['None', 'False', 'True']
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    # we need to catch the case where model is 3d cascade fullres and the low resolution folder has not been set.
    # In that case we need to try and predict with 3d low res first
    # if model == "3d_cascade_fullres" and lowres_segmentations is None:
    #     print("lowres_segmentations is None. Attempting to predict 3d_lowres first...")
    #     assert part_id == 0 and num_parts == 1, "if you don't specify a --lowres_segmentations folder for the " \
    #                                             "inference of the cascade, custom values for part_id and num_parts " \
    #                                             "are not supported. If you wish to have multiple parts, please " \
    #                                             "run the 3d_lowres inference first (separately)"
    #     model_folder_name = join(network_training_output_dir, "3d_lowres", task_name, trainer_class_name + "__" +
    #                               args.plans_identifier)
    #     assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name
    #     lowres_output_folder = join(output_folder, "3d_lowres_predictions")
    #     predict_from_folder(model_folder_name, input_folder, lowres_output_folder, folds, False,
    #                         num_threads_preprocessing, num_threads_nifti_save, None, part_id, num_parts, not disable_tta,
    #                         overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
    #                         mixed_precision=not args.disable_mixed_precision,
    #                         step_size=step_size)
    #     lowres_segmentations = lowres_output_folder
    #     torch.cuda.empty_cache()
    #     print("3d_lowres done")


    model_folder_name = get_output_folder(args.d, args.tr, args.p, args.c)
    # model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" +
    #                           args.plans_identifier)
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    predict_start_time = time.time()

    predict_from_folder(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                        overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not args.disable_mixed_precision,
                        step_size=step_size, checkpoint_name=args.chk)
                        
    print('Total predict time: ', time.time()-predict_start_time)

if __name__ == "__main__":
    main()