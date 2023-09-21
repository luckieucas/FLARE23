import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import numpy as np

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class DC_CE_Partial_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", 
                 weight_ce=1, weight_dice=1,ignore_label=255,ex=True,
                 dice_class=SoftDiceLoss):
        super(DC_CE_Partial_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ignore_label = ignore_label
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        #self.ce = nn.NLLLoss()
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ex_choice = ex
        self.weight_ce = 0
        self.weight_dice = weight_dice
        #self.apply_nonlin = softmax_helper_dim1
        print(f"mode:{aggregate}/ weight:[1:1] with exclusion:{ex}")

    def forward(self, net_output, target, partial_type):
        new_net_output, new_target = net_output.clone(), target.clone()
        #new_net_output = self.apply_nonlin(new_net_output)
        new_net_output, new_target = merge_prediction(new_net_output,
                                                new_target,
                                                partial_type)
        # filter other class output 
        dc_loss = self.dc(new_net_output, new_target)
        ce_loss = self.ce(new_net_output, new_target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        elif self.aggregate == "ce":
            result = ce_loss
        elif self.aggregate == "dc":
            result = dc_loss
        else:
            # reserved for other stuff (later?)
            raise NotImplementedError("nah son")
        return result


class DC_CE_Partial_MergeProb_loss(nn.Module):
    """
    for partial data, this loss first convert logits to prob and 
    merge prob to background class
    """
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", 
                 weight_ce=1, weight_dice=1,ignore_label=255,ex=True,
                 dice_class=SoftDiceLoss):
        super(DC_CE_Partial_MergeProb_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ignore_label = ignore_label
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        #self.ce = nn.NLLLoss()
        #self.dc = dice_class(apply_nonlin=None, **soft_dice_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ex_choice = ex
        self.weight_ce = 0
        self.weight_dice = weight_dice
        self.apply_nonlin = softmax_helper_dim1
        print(f"mode:{aggregate}/ weight:[1:1] with exclusion:{ex}")

    def forward(self, net_output, target, partial_type):
        new_net_output, new_target = net_output.clone(), target.clone()
        #new_net_output_soft = self.apply_nonlin(new_net_output)
        #print(f"dc old: {dc_old}, dc:{self.dc(new_net_output_soft, new_target)}")
        #print(f"ce old: {ce_old}, ce:{self.ce(torch.log(new_net_output_soft), new_target.squeeze().type(torch.cuda.LongTensor))}")
        # if partial_type[0]==14:
        #     new_net_output, new_target = merge_prediction(new_net_output,
        #                                                    new_target,
        #                                                    partial_type)
        # else:
 #       print(partial_type)
        if len(partial_type) < 14:
            new_net_output, new_target = merge_prediction_max(new_net_output,
                                                           new_target,
                                                           partial_type)
        # filter other class output 
        dc_loss = self.dc(new_net_output, new_target)
        ce_loss = self.ce(new_net_output, new_target)
        # ce_loss = self.ce(torch.log(new_net_output_soft), 
        #                   new_target.squeeze().type(torch.cuda.LongTensor))
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        elif self.aggregate == "ce":
            result = ce_loss
        elif self.aggregate == "dc":
            result = dc_loss
        else:
            # reserved for other stuff (later?)
            raise NotImplementedError("nah son")
        return result



class DC_CE_Partial_Filter_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", 
                 weight_ce=1, weight_dice=1,ignore_label=255,ex=True,
                 dice_class=SoftDiceLoss):
        super(DC_CE_Partial_Filter_loss, self).__init__()
        print("*"*10,"Using DC_CE_Partial_Filter_loss","*"*10)
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ignore_label = ignore_label
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        #self.ce = nn.NLLLoss()
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ex_choice = ex
        self.weight_ce = 0
        self.weight_dice = weight_dice
        #self.apply_nonlin = softmax_helper_dim1
        print(f"mode:{aggregate}/ weight:[1:1] with exclusion:{ex}")

    def forward(self, net_output, target, partial_type, do_bg_filter):
        new_net_output, new_target = net_output.clone(), target.clone()
        merge_classes = [item for item in range(1,15) if item not in partial_type]
        if do_bg_filter:
            new_net_output_soft = torch.softmax(new_net_output, dim=1)
            max_prob,max_index = torch.max(new_net_output_soft,dim=1)
            mask = torch.logical_not((max_prob>0.8) & torch.isin(max_index, torch.tensor(merge_classes).cuda())).unsqueeze(1)
        else:
            mask = None
        #new_net_output = self.apply_nonlin(new_net_output)
        new_net_output, new_target = merge_prediction(new_net_output,
                                                new_target,
                                                partial_type)
        # filter other class output 
        
        dc_loss = self.dc(new_net_output, new_target,loss_mask=mask)
        new_target[mask==0] = 255
        ce_loss = self.ce(new_net_output, 
                          new_target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        elif self.aggregate == "ce":
            result = ce_loss
        elif self.aggregate == "dc":
            result = dc_loss
        else:
            # reserved for other stuff (later?)
            raise NotImplementedError("nah son")
        return result



def merge_prediction(output, target, partial_type):
    '''
        cur_task: GT task
        default_task: net_output task
    '''
    
    try:
        merge_classes = [item for item in range(0,15) if item not in partial_type]
    except:
        print(f"partial error:{partial_type}")
    #print(f"merge prediction partial type: {partial_type}, merge classes: {merge_classes}")
    merge_output_bg = output[:, merge_classes, :, :].sum(dim=1, keepdim=True)
    output_fg = output[:, partial_type, :, :]
    new_output = torch.cat([merge_output_bg, 
                            output_fg], dim=1)
    for i,label in enumerate(partial_type):
        target[target==label] = i+1
    return new_output, target


def merge_prediction_max(output, target, partial_type):
    '''
        cur_task: GT task
        default_task: net_output task
    '''
 #   partial_type = partial_type.append(14)
  #  print(partial_type)
    #if 14 not in partial_type:
    #    partial_type = torch.cat((partial_type.view(-1),14*torch.ones_like(partial_type)[0]))
    #    print(partial_type)
        #partial_type.append(14)
    #print(partial_type)
    try:
        merge_classes = [item for item in range(0,14) if item not in partial_type]
    except:
        print(f"partial error:{partial_type}")
    #print(f"merge prediction partial type: {partial_type}, merge classes: {merge_classes}")
   # merge_classes = 0
    merge_output_bg, _ = output[:, merge_classes, :, :].max(dim=1, keepdim=True)
    output_fg = output[:, partial_type, :, :]
    new_target = torch.zeros_like(target)
    if 14 not in partial_type:
        new_output = torch.cat([merge_output_bg, 
                            output_fg, output[:,14,...].unsqueeze(1)], dim=1)
    else:
        new_output = torch.cat([merge_output_bg,
                            output_fg], dim=1)
   
    for i,label in enumerate(partial_type):
        new_target[target==label] = i+1

    if 14 not in partial_type:
        new_target[target==14] = len(partial_type) +1 
    #print(new_output.shape, torch.unique(target))
   # print(partial_type,merge_classes, new_output.shape, torch.unique(new_target))
    return new_output, new_target
