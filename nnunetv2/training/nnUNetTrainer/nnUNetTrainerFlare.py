import os
import random
from tqdm import tqdm
import multiprocessing
from typing import Union, Tuple, List
import torch
import shutil
from nnunetv2.utilities.file_path_utilities import check_workers_busy
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, predict_sliding_window_return_logits
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.loss.compound_losses import DC_CE_Partial_loss,DC_CE_Partial_Filter_loss,DC_CE_Partial_MergeProb_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3DPartial,nnUNetDataLoader3DPartialVal
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime
from time import time, sleep
#import wandb


class nnUNetTrainerFlare(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 continue_training: bool = False,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        #self.num_epochs = 250
        print("*"*10,"Using nnUNetTrainerFlare From luckie","*"*10)
        self.different_partial_type_keys_json = join(
            self.preprocessed_dataset_folder_base, 
            "different_partial_type.json")
        self.case_to_partial_type_json = join(
            self.preprocessed_dataset_folder_base,
            "case_to_partial_type.json"
        )
        self.different_partial_type_keys = load_json(self.different_partial_type_keys_json)
        self.case_to_partial_type_dict = load_json(self.case_to_partial_type_json)
        self.save_every = 1
        self.num_iterations_per_epoch = 250 #250
        self.num_val_iterations_per_epoch = 10
        self.began_partial_epoch = 100
        self.num_epochs = 1000
        self.began_save_chk = 800
        self.experiment_name = self.__class__.__name__ + "__" \
                            + configuration + "__" + f'fold_{fold}_MaxOnlyTumor'
        #if not continue_training:
       # self.wandb_logger = wandb.init(name=self.experiment_name,
       #                                     project="FLARE2023",
       #                                     config = self.configuration_manager)
       # 
        self.class_name = [key for key, value in sorted(self.dataset_json['labels'].items(), 
                                                        key=lambda item: item[1])]
        print("class name: ",self.class_name)
        
        
        
    def _build_loss(self):
        # loss = DC_and_CE_loss(
        #     {'batch_dice': self.configuration_manager.batch_dice,
        #      'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
        #     {}, weight_ce=1, weight_dice=1,ignore_label=255, 
        #     dice_class=MemoryEfficientSoftDiceLoss)
        loss = DC_CE_Partial_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
            {}, weight_ce=1, weight_dice=1,ignore_label=255, 
            dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        #weights = [weights[0]] + [0]*(len(weights)-1)
        print(f"deep supervision weights:{weights}")
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append({})
                    splits[-1]['train'] = list(train_keys)
                    splits[-1]['val'] = list(test_keys)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def _get_different_partial_keys(self, partial_type):
        label_keys = self.different_partial_type_keys[partial_type]
        return label_keys
    
        

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()
        full_label_keys = self._get_different_partial_keys("1_2_3_4_5_6_7_8_9_10_11_12_13")
        tr_keys_full = list(set(tr_keys).intersection(set(full_label_keys)))
        val_keys_full = list(set(val_keys).intersection(set(full_label_keys)))
        #tr_keys_partial = list(set(tr_keys).intersection(set(tumor_label_keys)))
        #val_keys = list(set(val_keys).intersection(set(tumor_label_keys)))
        tr_keys_partial = [item for item in tr_keys if item not in full_label_keys]
        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr_full = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys_full,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)
        dataset_tr_partial = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys_partial,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)
        # dataset_tr = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys,
        #                            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        #                            num_images_properties_loading_threshold=0)
        dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)
        return dataset_tr_full,dataset_tr_partial, dataset_val  

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(True)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            self.print_to_log_file('unpacking done...')

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        (self.dataloader_train_full, 
         self.dataloader_train_partial, 
         self.dataloader_val) = self.get_dataloaders()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")
  
    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dl_tr_full, dl_tr_partial, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train_full = SingleThreadedAugmenter(dl_tr_full, tr_transforms)
            mt_gen_train_partial = SingleThreadedAugmenter(dl_tr_partial, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train_full = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr_full, transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_train_partial = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr_partial, transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
                                           transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
                                           wait_time=0.02)
        return mt_gen_train_full, mt_gen_train_partial, mt_gen_val

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr_full, dataset_tr_partial, dataset_val = self.get_tr_and_val_datasets()

        dl_tr_full = nnUNetDataLoader3DPartial(dataset_tr_full, self.batch_size,
                                    initial_patch_size,
                                    self.configuration_manager.patch_size,
                                    self.label_manager,
                                    self.different_partial_type_keys,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None)
        dl_tr_partial = nnUNetDataLoader3DPartial(dataset_tr_partial, self.batch_size,
                                    initial_patch_size,
                                    self.configuration_manager.patch_size,
                                    self.label_manager,
                                    self.different_partial_type_keys,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None,
                                    is_partial=True)
        
        dl_val = nnUNetDataLoader3DPartialVal(dataset_val, self.batch_size,
                                    self.configuration_manager.patch_size,
                                    self.configuration_manager.patch_size,
                                    self.label_manager,
                                    self.different_partial_type_keys,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None,
                                    is_partial=True)
        return dl_tr_full, dl_tr_partial, dl_val

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """
        change the way of computing mean dice, since organ and tumor are 
        evaluated separately.
        """
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = np.array([i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(tp, fp, fn)]])
        global_dc_per_class = np.where(global_dc_per_class==0, np.nan, 
                                       global_dc_per_class)
        organ_dice = np.nanmean(global_dc_per_class[:-1]) 
        tumor_dice = global_dc_per_class[-1]
        mean_fg_dice = np.nanmean([organ_dice,tumor_dice]) 
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, 
                        self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        fg_dice_dict = {a: b for a, b in zip(self.class_name[1:], global_dc_per_class)}
        #wandb_log_dict = dict(fg_dice_dict)
        #wandb_log_dict.update({'mean_dice':mean_fg_dice,'val_losses':loss_here})
        #self.wandb_logger.log(wandb_log_dict,
        #                      step=self.current_epoch)

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        # self.wandb_logger = wandb.init(name=self.experiment_name,
        #                                     project="FLARE2023",
        #                                     config = self.configuration_manager,
        #                                     id=checkpoint['wandb_id'],
        #                                     resume='must')
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
    
    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # todo find a solution for this stupid shit
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))
        if current_epoch % 50 == 0 or current_epoch > self.began_save_chk:
            self.save_checkpoint(join(self.output_folder, 
                                      f'checkpoint_ep{current_epoch}.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
        
    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                   # 'wandb_id': self.wandb_logger.id
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def train_step(self, batch: dict, partial: bool=False) -> dict:
        data = batch['data']
        target = batch['target']
        partial_type = [int(item) for item in batch['partial_type'][0].split("_")]
        partial_type = [torch.tensor(partial_type).to(self.device)]*len(target)
        # if partial:
        #     print("partial type: ",partial_type)
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # for patial data ignore background
        # if partial:
        #     new_target = []
        #     for item in target:
        #         item[item==0]=255
        #         new_target.append(item)
        #     target = new_target
        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # 
            l = self.loss(output, target, partial_type)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def _convert_out_seg_to_partial(self, out_seg, partial_type):
        mask = torch.isin(out_seg, partial_type)
        return torch.where(mask, out_seg, torch.zeros_like(out_seg))
        
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
          
        partial_type = [int(item) for item in batch['partial_type'][0].split("_")]
        partial_type = [torch.tensor(partial_type).to(self.device)]*len(target)
        #print(f"partial type: {partial_type}")
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target, partial_type)

        # we only need the output with the highest output resolution
        output = output[0]
        target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            #print("output shape: ",output.shape)
            output_seg = output.argmax(1)[:, None]
            # print(f"target unique:{target.unique()},partial type:{partial_type[0]}")
            # print("total pixel for 14: ",(target==14).sum())
            # print(f"tp before: {output_seg[target==14].sum()}")
            # print(f"before merge output_seg unique:{output_seg.unique()}")
            # convert to partial type
            output_seg = self._convert_out_seg_to_partial(output_seg, 
                                                          partial_type[0])
            #print(f"after merge output_seg unique:{output_seg.unique()}")
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None
        # mask = torch.zeros_like(target)
        # mask[target>0] = 1

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, 
                                        axes=axes, mask=mask)

        #print(f"tp: {tp}")
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            train_outputs_partial = []
            if epoch > self.began_partial_epoch:
                self.num_iterations_per_epoch = 120
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train_full)))
                if epoch > self.began_partial_epoch:
                        # use partial label data for training
                    train_outputs_partial.append(
                            self.train_step(
                                next(self.dataloader_train_partial),True))
                    
            self.on_train_epoch_end(train_outputs)
            if epoch > self.num_epochs//2.5:
                self.num_val_iterations_per_epoch = 120
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
    
    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        num_seg_heads = self.label_manager.num_segmentation_heads

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            print('*********val_keys:{}******'.format(val_keys))
            if self.is_ddp:
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []
            for k in dataset_val.keys():
                output_filename_truncated = join(validation_output_folder, k)
                if isfile(output_filename_truncated+".nii.gz"):
                    print(f"{k} is already predicted")
                    continue
                proceed = not check_workers_busy(segmentation_export_pool, results,
                                                 allowed_num_queued=2 * len(segmentation_export_pool._pool))
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_busy(segmentation_export_pool, results,
                                                     allowed_num_queued=2 * len(segmentation_export_pool._pool))

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))


                try:
                    prediction = predict_sliding_window_return_logits(self.network, data, num_seg_heads,
                                                                      tile_size=self.configuration_manager.patch_size,
                                                                      mirror_axes=self.inference_allowed_mirroring_axes,
                                                                      tile_step_size=0.5,
                                                                      use_gaussian=True,
                                                                      precomputed_gaussian=None,
                                                                      perform_everything_on_gpu=True,
                                                                      verbose=False,
                                                                      device=self.device).cpu().numpy()
                except RuntimeError:
                    prediction = predict_sliding_window_return_logits(self.network, data, num_seg_heads,
                                                                      tile_size=self.configuration_manager.patch_size,
                                                                      mirror_axes=self.inference_allowed_mirroring_axes,
                                                                      tile_step_size=0.5,
                                                                      use_gaussian=True,
                                                                      precomputed_gaussian=None,
                                                                      perform_everything_on_gpu=False,
                                                                      verbose=False,
                                                                      device=self.device).cpu().numpy()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = nnUNetDataset(expected_preprocessed_folder, [k],
                                                num_images_properties_loading_threshold=0)
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file = join(output_folder, k + '.npz')

                        # resample_and_save(prediction, target_shape, output_file, self.plans, self.configuration, properties,
                        #                   self.dataset_json)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json),
                            )
                        ))

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]), also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

    def _randomly_selected_val_keys_for_each_epoch(self):
        """
        Selects a random subset of the validation keys for each epoch.
        """
        _, val_keys = self.do_split()
        partial_type_selected_for_val = {'1_2_3_4_5_6_7_8_9_10_11_12_13_14':[],
                            '1_2_3_4_5_6_7_8_9_10_11_12_13':[],
                            '1_2_3_4_13_14':[],
                            '1_2_3_4_13':[],
                            'all':[]}
        random.shuffle(val_keys)
        final_val_keys = []
        for k in val_keys:
            partial_type = self.case_to_partial_type_dict[k]
            if partial_type not in partial_type_selected_for_val:
                continue
            if len(partial_type_selected_for_val[partial_type]) < 10:
                final_val_keys.append(k)
                partial_type_selected_for_val[partial_type].append(k)
                partial_type_selected_for_val['all'].append(k)
            if len(final_val_keys) == 40:
                break
        save_json(partial_type_selected_for_val,join(self.output_folder,'selected_val_keys.json'))
        return final_val_keys
    
    def perform_actual_validation_each_epoch(self, val_epoch: int = 0, 
                                             save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        num_seg_heads = self.label_manager.num_segmentation_heads

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            validation_output_folder = join(self.output_folder, f'validation_ep{val_epoch}')
            maybe_mkdir_p(validation_output_folder)
            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers
            if isfile(join(self.output_folder,f'selected_val_keys.json')):
                val_keys_dict = load_json(join(self.output_folder,f'selected_val_keys.json'))
                val_keys = val_keys_dict['all']
            else:
                val_keys = self._randomly_selected_val_keys_for_each_epoch()
                
                
            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            results = []
            for k in tqdm(dataset_val.keys()):
                output_filename_truncated = join(validation_output_folder, k)
                if isfile(output_filename_truncated+".nii.gz"):
                    print(f"{k} is already predicted")
                    continue
                proceed = not check_workers_busy(segmentation_export_pool, results,
                                                 allowed_num_queued=2 * len(segmentation_export_pool._pool))
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_busy(segmentation_export_pool, results,
                                                     allowed_num_queued=2 * len(segmentation_export_pool._pool))

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))


                try:
                    prediction = predict_sliding_window_return_logits(self.network, data, num_seg_heads,
                                                                      tile_size=self.configuration_manager.patch_size,
                                                                      mirror_axes=self.inference_allowed_mirroring_axes,
                                                                      tile_step_size=0.5,
                                                                      use_gaussian=True,
                                                                      precomputed_gaussian=None,
                                                                      perform_everything_on_gpu=True,
                                                                      verbose=False,
                                                                      device=self.device).cpu().numpy()
                except RuntimeError:
                    prediction = predict_sliding_window_return_logits(self.network, data, num_seg_heads,
                                                                      tile_size=self.configuration_manager.patch_size,
                                                                      mirror_axes=self.inference_allowed_mirroring_axes,
                                                                      tile_step_size=0.5,
                                                                      use_gaussian=True,
                                                                      precomputed_gaussian=None,
                                                                      perform_everything_on_gpu=False,
                                                                      verbose=False,
                                                                      device=self.device).cpu().numpy()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

            _ = [r.get() for r in results]


        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]), also_print_to_console=True)
            print(f"metrics mean keys:{metrics['mean'].keys()}")
            self.print_to_log_file("Mean Organ Dice: ", (sum([metrics['mean'][i]['Dice'] for i in range(1,14)]) / 13.0), 
                                   also_print_to_console=True)
            self.print_to_log_file("Mean Tumor Dice: ", (metrics['mean'][14]['Dice']), also_print_to_console=True)
            wandb_log_dict = {"mean organ dice": sum([metrics['mean'][i]['Dice'] for i in range(1,14)]) / 13.0,
                              "mean tumor dice": metrics['mean'][14]['Dice'],
                              "mean dice": (sum([metrics['mean'][i]['Dice'] for i in range(1,14)]) / 13.0 + metrics['mean'][14]['Dice'])/2.0}
            self.wandb_logger.log(wandb_log_dict,
                              step=val_epoch)

        compute_gaussian.cache_clear()

class nnUNetTrainerFlarePartialFilter(nnUNetTrainerFlare):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 continue_training: bool=False,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        #self.num_epochs = 250
        print("*"*10,"Using nnUNetTrainerFlarePartialFilter From luckie","*"*10)
        self.began_partial_filter_epoch = 400
        self.do_bg_filter = False
        
    def _build_loss(self):
        # loss = DC_and_CE_loss(
        #     {'batch_dice': self.configuration_manager.batch_dice,
        #      'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
        #     {}, weight_ce=1, weight_dice=1,ignore_label=255, 
        #     dice_class=MemoryEfficientSoftDiceLoss)
        loss = DC_CE_Partial_Filter_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
            {}, weight_ce=1, weight_dice=1,ignore_label=255, 
            dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        #weights = [weights[0]] + [0]*(len(weights)-1)
        print(f"deep supervision weights:{weights}")
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
    def train_step(self, batch: dict, partial: bool=False) -> dict:
        data = batch['data']
        target = batch['target']
        partial_type = [int(item) for item in batch['partial_type'][0].split("_")]
        partial_type = [torch.tensor(partial_type).to(self.device)] * len(target)
        do_bg_filter = [torch.BoolTensor([self.do_bg_filter])] * len(target)
        # if partial:
        #     print("partial type: ",partial_type)
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # for patial data ignore background
        # if partial:
        #     new_target = []
        #     for item in target:
        #         item[item==0]=255
        #         new_target.append(item)
        #     target = new_target
        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # 
            l = self.loss(output, target, partial_type, do_bg_filter)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    
    
    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            train_outputs_partial = []
            if epoch > self.began_partial_filter_epoch:
                self.do_bg_filter = True
            if epoch > self.began_partial_epoch:
                self.num_iterations_per_epoch = 100
                self.num_val_iterations_per_epoch = 120
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train_full)))
                if epoch > self.began_partial_epoch:
                        # use partial label data for training
                    train_outputs_partial.append(
                            self.train_step(
                                next(self.dataloader_train_partial),True))
                    
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
        
    def _convert_out_seg_to_partial(self, out_seg, partial_type):
        mask = torch.isin(out_seg, partial_type)
        return torch.where(mask, out_seg, torch.zeros_like(out_seg))
        
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
          
        partial_type = [int(item) for item in batch['partial_type'][0].split("_")]
        partial_type = [torch.tensor(partial_type).to(self.device)]*len(target)
        do_bg_filter = [torch.BoolTensor([self.do_bg_filter])] * len(target)
        #print(f"partial type: {partial_type}")
        data = data.to(self.device, non_blocking=True)
        
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target, partial_type, do_bg_filter)

        # we only need the output with the highest output resolution
        output = output[0]
        target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            # print(f"target unique:{target.unique()},partial type:{partial_type}")
            # print("total pixel for 14: ",(target==14).sum())
            # print(f"tp before: {output_seg[target==14].sum()}")
            # print(f"before merge output_seg unique:{output_seg.unique()}")
            # convert to partial type
            output_seg = self._convert_out_seg_to_partial(output_seg, 
                                                          partial_type[0])
            #print(f"after merge output_seg unique:{output_seg.unique()}")
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None
        # mask = torch.zeros_like(target)
        # mask[target>0] = 1

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        #(f"tp: {tp}")
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    
class nnUNetTrainerFlareMergeProb(nnUNetTrainerFlare):
    def __init__(self, plans: dict, configuration: str, fold: int, 
                 dataset_json: dict, unpack_dataset: bool = True,
                 continue_training: bool = False,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, 
                         continue_training, device)
        #self.num_epochs = 250
        print("*"*10,"Using nnUNetTrainerFlareMergeProb From luckie","*"*10)
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 120
        self.began_partial_epoch = 100
        self.num_epochs = 2000
        
    def _build_loss(self):
        # loss = DC_and_CE_loss(
        #     {'batch_dice': self.configuration_manager.batch_dice,
        #      'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
        #     {}, weight_ce=1, weight_dice=1,ignore_label=255, 
        #     dice_class=MemoryEfficientSoftDiceLoss)
        loss = DC_CE_Partial_MergeProb_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
            {}, weight_ce=1, weight_dice=1,ignore_label=255, 
            dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        #weights = [weights[0]] + [0]*(len(weights)-1)
        print(f"deep supervision weights:{weights}")
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
    