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
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D,nnUNetDataLoader3DPartial
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
import wandb
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss


class nnUNetTrainerTumor(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 continue_training: bool = False,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        #self.num_epochs = 250
        print("*"*10,"Using nnUNetTrainerTumor From luckie","*"*10)
        self.num_iterations_per_epoch = 250 #250
        self.num_val_iterations_per_epoch = 100
        self.began_partial_epoch = 100
        self.num_epochs = 1000
        self.began_save_chk = 700
        
        
        
    def _build_loss(self):
        loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
            {}, weight_ce=1, weight_dice=1,ignore_label=255, 
            dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[2:] = 0
        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        #weights = [weights[0]] + [0]*(len(weights)-1)
        print(f"deep supervision weights:{weights}")
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
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