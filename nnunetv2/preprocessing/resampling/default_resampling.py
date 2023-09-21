from collections import OrderedDict
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import resize
from nnunetv2.configuration import ANISO_THRESHOLD
import time 
import torch
import torch.nn as nn


def get_do_separate_z(spacing: Union[Tuple[float, ...], List[float], np.ndarray], anisotropy_threshold=ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def compute_new_shape(old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                      old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape


def resample_data_or_seg_to_spacing(data: np.ndarray,
                                    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                    is_seg: bool = False,
                                    order: int = 3, order_z: int = 0,
                                    force_separate_z: Union[bool, None] = False,
                                    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the same spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"

    shape = np.array(data[0].shape)
    new_shape = compute_new_shape(shape[1:], current_spacing, new_spacing)

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped


def resample_data_or_seg_to_shape(data: np.ndarray,
                                  new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                                  current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  is_seg: bool = False,
                                  order: int = 3, order_z: int = 0,
                                  force_separate_z: Union[bool, None] = False,
                                  separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    """
    needed for segmentation export. Stupid, I know. Maybe we can fix that with Leos new resampling functions
    """
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
   # print('new_shape:{}/{}--------------------'.format(data.shape, new_shape))
    if np.any(shape != new_shape):
        data = data.astype(np.float32)
        if new_shape[0] < 250:
            # import time 
            # start_time = time.time()
            # reshaped_final_data1 = nn.functional.interpolate(torch.from_numpy(data).unsqueeze(0).type(torch.float32),size=list(new_shape),scale_factor= None , mode = 'trilinear').numpy()[0]
            # print(f"use cpu time:{time.time()-start_time}")
            #start_time = time.time()
            reshaped_final_data = nn.functional.interpolate(torch.from_numpy(data).unsqueeze(0).type(torch.float32).cuda(),size=list(new_shape),scale_factor= None , mode = 'trilinear').cpu().numpy()[0]
            #print(f"use gpu time:{time.time()-start_time}")
            #np.testing.assert_almost_equal(reshaped_final_data1, reshaped_final_data, decimal=7, err_msg='', verbose=True) 
        else:
            reshaped_final_data = nn.functional.interpolate(torch.from_numpy(data).unsqueeze(0).type(torch.float32).cuda(),size=list(new_shape)).cpu().numpy()[0]
        #         # for c in range(data.shape[0]):
            #     print(f"resample data shape:{data.shape}")
            #     reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None])
            #     print(f"resample data or seg 2:{time.time() - start_time}")
            # reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        # print("no resampling necessary")
        return data
    #data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    #return data_reshaped


def resample_data_or_seg(data: np.ndarray, new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
                         is_seg: bool = False, axis: Union[None, int] = None, order: int = 3,
                         do_separate_z: bool = False, order_z: int = 0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    print(f"shape:{shape}")
    new_shape = np.array(new_shape)
   # print('new_shape:{}/{}--------------------'.format(data.shape, new_shape))
    if np.any(shape != new_shape):
        data = data.astype(np.float32)
        if new_shape[0] < 250:
            import time 
            start_time = time.time()
            reshaped_final_data1 = nn.functional.interpolate(torch.from_numpy(data).unsqueeze(0).type(torch.float32),size=list(new_shape),scale_factor= None , mode = 'trilinear').numpy()[0]
            print(f"use cpu time:{time.time()-start_time}")
            start_time = time.time()
            reshaped_final_data = nn.functional.interpolate(torch.from_numpy(data).unsqueeze(0).type(torch.float32).cuda(),size=list(new_shape),scale_factor= None , mode = 'trilinear').cpu().numpy()[0]
            print(f"use gpu time:{time.time()-start_time}")
            np.testing.assert_almost_equal(reshaped_final_data1, reshaped_final_data, decimal=7, err_msg='', verbose=True) 
        else:
            reshaped_final_data = nn.functional.interpolate(torch.from_numpy(data).unsqueeze(0).type(torch.float32),size=list(new_shape)).numpy()[0]
        #         # for c in range(data.shape[0]):
            #     print(f"resample data shape:{data.shape}")
            #     reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None])
            #     print(f"resample data or seg 2:{time.time() - start_time}")
            # reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        # print("no resampling necessary")
        return data
