import random
from typing import Tuple, Union
from batchgenerators.utilities.file_and_folder_operations import List
import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from batchgenerators.utilities.file_and_folder_operations import load_pickle
import os.path as osp
class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


class nnUNetDataLoader3DPartial(nnUNetDataLoaderBase):
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 different_partial_type_keys,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False,
                 is_partial=False):
        super().__init__(data, batch_size, patch_size, final_patch_size, 
                         label_manager, oversample_foreground_percent, 
                         sampling_probabilities, pad_sides, 
                         probabilistic_oversampling)
        self.different_partial_type_keys={k:v for k,v in 
            different_partial_type_keys.items() if len(v) >= batch_size}
        self.indices_all = list(data.keys())
        self.is_partial = is_partial
        ### TODO load tumor cases
        f = open('/data/xining.zq/flare_nnunet/nnunet_unsup/nnunetv2/training/dataloading/tumor_keys.txt')
        self.tumor_cases = []
        for line in f.readlines():
            line = line.rstrip('\r\n')
            self.tumor_cases.append(line)

        self.tumor_path = '/data/liupeng/nnUNet-master/DATASET/nnUNet_preprocessed/Dataset008_FLARE2023Filter/tumor_3d_midres'
        self.tumor_num = len(self.tumor_cases)
    
    def _random_select_partial_type(self):
        different_type_cnt = [len(v) for v in self.different_partial_type_keys.values()]
        # replace the number of partial type 14
        #index = different_type_cnt.index(1888)
 #      # print(different_type_cnt, index)
        #different_type_cnt[index] = 800
        total_cnt = sum(different_type_cnt)
        probs = [p / total_cnt for p in different_type_cnt]
        
        # random select a key
        selected_partial_type = random.choices(
            list(self.different_partial_type_keys.keys()), weights=probs)[0]
        return selected_partial_type
        
        
        
    def generate_train_batch(self):
        selected_partial_type = "1_2_3_4_5_6_7_8_9_10_11_12_13"
        if self.is_partial:
            selected_partial_type = self._random_select_partial_type()
            selected_cases = self.different_partial_type_keys[selected_partial_type]
            self.indices = list(set(self.indices_all).intersection(set(selected_cases)))
            while len(self.indices) < self.batch_size:
                selected_partial_type = self._random_select_partial_type()
                selected_cases = self.different_partial_type_keys[selected_partial_type]
                self.indices = list(set(self.indices_all).intersection(set(selected_cases)))
        self.annotated_classes_key = tuple(selected_partial_type.split("_"))
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        #print(selected_partial_type)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)
            data, seg, properties = self._data.load_case(i)
            if selected_partial_type == '1_2_3_4_5_6_7_8_9_10_11_12_13_14' and 'organ_bbox' not in properties:
               # print('*******pure tumor type***************')
                for seg_i in range(1,14):
                        properties['class_locations'][seg_i] = []

         #   print(properties.keys())

            if 'organ_bbox' in properties:
 #               print(properties['class_locations'][14])
               # print('unsupervised data, key:{}'.format(i))
                pseudo_bbox = properties['organ_bbox']
           #     print(pseudo_bbox)
                select_id = np.random.randint(self.tumor_num)
                tumor_img = osp.join(self.tumor_path, self.tumor_cases[select_id][:-4]+ ".npy")
                tumor_seg = osp.join(self.tumor_path,self.tumor_cases[select_id][:-4]+ "_seg.npy")
                #load_pickle(osp.join(self.tumor_path,self.tumor_cases[select_id]))

                tumor_seg = np.load(tumor_seg)[0]
                tumor_data = np.load(tumor_img)[0]
               # print('***********tumor_data:{}********'.format(tumor_data.shape))
### TODO perform cutmix with loaded organ box

                z,h,w = tumor_data.shape
                organ_z, organ_h, organ_w = pseudo_bbox[0][1]-pseudo_bbox[0][0],\
                                pseudo_bbox[1][1]-pseudo_bbox[1][0],\
                                pseudo_bbox[2][1]-pseudo_bbox[2][0]

                if z > organ_z:
                    crop_z = (z - organ_z) // 2
                    tumor_data = tumor_data[crop_z:crop_z+organ_z]
                    tumor_seg = tumor_seg[crop_z:crop_z+organ_z]
                if h > organ_h:
                    crop_h = (h - organ_h) // 2
                    tumor_data = tumor_data[:,crop_h:crop_h+organ_h,:]
                    tumor_seg = tumor_seg[:,crop_h:crop_h+organ_h,:]
                if w > organ_w:
                    crop_w = (w - organ_w) // 2
                    tumor_data = tumor_data[:,:,crop_w:crop_w+organ_w]
                    tumor_seg = tumor_seg[:,:,crop_w:crop_w+organ_w]

                z,h,w = tumor_data.shape
          #      print(tumor_data.shape)
                try:
                    begin_z = np.random.randint(pseudo_bbox[0][0], pseudo_bbox[0][1] - z)
                except:
                    begin_z = 0
                try:
                    begin_h = np.random.randint(pseudo_bbox[1][0], pseudo_bbox[1][1] - h)
                except:
                    begin_h = 0
                try:
                    begin_w = np.random.randint(pseudo_bbox[2][0], pseudo_bbox[2][1] - w)
                except:
                    begin_w = 0

                data[0,begin_z:begin_z+z, begin_h:begin_h+h, begin_w:begin_w+w] = tumor_data
                seg[0,begin_z:begin_z+z, begin_h:begin_h+h, begin_w:begin_w+w] = tumor_seg

                #### get new tumor locations
#                zz,hh,ww = np.meshgrid(np.arange(begin_z,begin_z+z),np.arange(begin_h,begin_h+h),np.arange( begin_w,begin_w+w))
#                zz,hh,ww  = zz.reshape(-1),hh.reshape(-1),ww.reshape(-1)
#                zero_pre = np.array([0]* len(zz))
#                tmp_loc = np.stack([zero_pre,zz,hh,ww],axis = -1)
                zero_pre,zz,hh,ww = np.where(seg == 14)
                tmp_loc = np.stack([zero_pre,zz,hh,ww],axis = -1)
                properties['class_locations'][14] = tmp_loc
              ##  print(properties['class_locations'][14] )
              ##  print(begin_z,begin_h,begin_w,z,h,w ,tmp_loc.shape)





            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 
                'keys': selected_keys, 
                'partial_type':[selected_partial_type]*self.batch_size}


class nnUNetDataLoader3DPartialVal(nnUNetDataLoaderBase):
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 different_partial_type_keys,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False,
                 is_partial=False):
        super().__init__(data, batch_size, patch_size, final_patch_size, 
                         label_manager, oversample_foreground_percent, 
                         sampling_probabilities, pad_sides, 
                         probabilistic_oversampling)
        self.different_partial_type_keys={k:v for k,v in 
            different_partial_type_keys.items() if len(v) >= 50}
        self.indices_all = list(data.keys())
        self.is_partial = is_partial
    
    def _random_select_partial_type(self):
        # random select a key
        selected_partial_type = random.choices(
            list(self.different_partial_type_keys.keys()))[0]
        return selected_partial_type
        
        
        
    def generate_train_batch(self):
        selected_partial_type = "1_2_3_4_5_6_7_8_9_10_11_12_13"
        if self.is_partial:
            selected_partial_type = self._random_select_partial_type()
            selected_cases = self.different_partial_type_keys[selected_partial_type]
            self.indices = list(set(self.indices_all).intersection(set(selected_cases)))
            while len(self.indices) < self.batch_size:
                selected_partial_type = self._random_select_partial_type()
                selected_cases = self.different_partial_type_keys[selected_partial_type]
                self.indices = list(set(self.indices_all).intersection(set(selected_cases)))
        self.annotated_classes_key = tuple(selected_partial_type.split("_"))
        selected_keys = self.get_indices()

        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            new_seg = np.zeros_like(seg)
            if 'tumor_organ_bbox' in properties:
                tmp_bbox = properties['tumor_organ_bbox']
                new_seg[0,tmp_bbox[0][0]:tmp_bbox[0][1],tmp_bbox[1][0]:tmp_bbox[1][1],tmp_bbox[2][0]:tmp_bbox[2][1]] = seg[0,tmp_bbox[0][0]:tmp_bbox[0][1],tmp_bbox[1][0]:tmp_bbox[1][1],tmp_bbox[2][0]:tmp_bbox[2][1]] 
     #           print('--------unique_labels:{},{}---'.format(i,np.unique(new_seg)))
                seg = new_seg

            ### update labels for tumor !!
            if '14' in selected_partial_type:
                if selected_partial_type == '1_2_3_4_5_6_7_8_9_10_11_12_13_14':
                    for seg_i in range(1,14):
                        properties['class_locations'][seg_i] = []
                else:
                    partial_type = [int(item) for item in selected_partial_type.split("_")]
                    for seg_i in range(1,14):
                        if seg_i not in partial_type:
                            properties['class_locations'][seg_i] = []
                #    seg[seg != 14] = 0
                #else:
                #    partial_type = [int(item) for item in selected_partial_type.split("_")]
                #    for seg_i in range(1,14):
                #        if seg_i not in partial_type:
                #            seg[seg == seg_i] = 0
             #  # print('*********partial type:{}, unique label:{}'.format(selected_partial_type, np.unique(seg)))

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 
                'keys': selected_keys, 
                'partial_type':[selected_partial_type]*self.batch_size}

if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
