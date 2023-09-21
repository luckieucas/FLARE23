import os
from typing import Union, Tuple, List
import torch
import shutil
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.loss.compound_losses import DC_CE_Partial_loss,DC_CE_Partial_Filter_loss,DC_CE_Partial_MergeProb_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.data_loader_3d_filter import nnUNetDataLoader3DPartial,nnUNetDataLoader3DPartialVal
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime
from time import time, sleep
#import wandb
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerFlare import nnUNetTrainerFlare,nnUNetTrainerFlareMergeProb


class nnUNetTrainerFlarePseudoFilter(nnUNetTrainerFlareMergeProb):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 continue_training: bool = False,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, 
                         unpack_dataset,continue_training, device)
        #self.num_epochs = 250
        print("*"*10,"Using nnUNetTrainerFlarePseudo From luckie","*"*10)
        self.different_partial_type_keys_json = join(
            self.preprocessed_dataset_folder_base, 
            "different_partial_type_unsup_recover.json")
        self.case_to_partial_type_json = join(
            self.preprocessed_dataset_folder_base,
            "case_to_partial_type.json"
        )
        self.case_to_partial_type_dict = load_json(self.case_to_partial_type_json)
        self.save_every = 1
        self.num_iterations_per_epoch = 250 #250
        self.num_val_iterations_per_epoch = 20
      #  self.began_partial_epoch = 100
      #  self.num_epochs = 1000
        self.initial_lr = 3e-3
        self.began_partial_epoch = 0
        self.num_epochs = 500
        self.experiment_name = self.__class__.__name__ + "__" \
                            + configuration + "__" + f'fold_{fold}_MaxOnlyTumor'
        #if not continue_training:
     #   self.wandb_logger = wandb.init(name=self.experiment_name,
     #                                       project="FLARE2023",
     #                                       config = self.configuration_manager)
        
        self.class_name = [key for key, value in sorted(self.dataset_json['labels'].items(), 
                                                        key=lambda item: item[1])]
        print("class name: ",self.class_name)
        
    def _get_different_partial_keys(self):
        self.different_partial_type_keys = load_json(self.different_partial_type_keys_json)
        full_label_keys = self.different_partial_type_keys['1_2_3_4_5_6_7_8_9_10_11_12_13']
        tumor_label_keys = self.different_partial_type_keys['1_2_3_4_5_6_7_8_9_10_11_12_13_14']
        return full_label_keys,tumor_label_keys
    
        

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()
        full_label_keys,tumor_label_keys = self._get_different_partial_keys()
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
        if current_epoch % 50 == 0:
            self.save_checkpoint(join(self.output_folder, 
                                      f'checkpoint_ep{current_epoch}.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))
            self.save_checkpoint(join(self.output_folder, f'checkpoint_best_ep{current_epoch}.pth'))

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
            if len(partial_type[0]) < 14:
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
                self.num_val_iterations_per_epoch = 100
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
