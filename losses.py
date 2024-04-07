import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, log_loss=False):
        super(DiceLoss, self).__init__()
        self.ingore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, input, target):
        smooth = 1.

        # Apply softmax to input (model output)
        input = torch.softmax(input, dim=1)

        dice_loss = 0.

        for class_index in range(input.size(1)):
            valid = (target != self.ingore_index)
            input_flat = input[:, class_index, :, :][valid].contiguous().view(-1)
            target_flat = (target == class_index)[valid].contiguous().view(-1) # binary target for class_index

            intersection = (input_flat * target_flat).sum()

            class_loss = 1 - ((2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth))
            dice_loss += class_loss

        mean_dice = dice_loss/input.size(1) # average loss over all classes

        if self.log_loss:
            mean_dice = -torch.log(mean_dice)

        return mean_dice



class IoULoss(nn.Module):
    def __init__(self, ignore_index=255, log_loss=False):
        super(IoULoss, self).__init__()
        self.ignore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, input, target):
        smooth = 1e-6

        # Apply softmax to input (model output)
        input = torch.softmax(input, dim=1)

        iou_loss = 0.

        for class_index in range(input.size(1)):
            valid = (target != self.ignore_index)
            input_flat = input[:, class_index, :, :][valid].contiguous().view(-1)
            target_flat = (target == class_index)[valid].contiguous().view(-1) # binary target for class_index

            intersection = (input_flat * target_flat).sum()

            class_loss = 1 - ((intersection + smooth) / (input_flat.sum() + target_flat.sum() - intersection + smooth))
            iou_loss += class_loss

        mean_iou = iou_loss/input.size(1) # average loss over all classes

        if self.log_loss:
            mean_iou = -torch.log(mean_iou)

        return mean_iou

class FocalLoss(_Loss):
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean', ignore_index: int = None):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute the negative log-likelihood for each input-target pair
        neg_log_pt = F.nll_loss(F.log_softmax(input, dim=1), target, reduction='none', ignore_index=self.ignore_index)
        # Model's estimated probability: p_t = exp(-logpt) uses the log probability of the ground truth class (hence the minus)
        pt = torch.exp(-neg_log_pt)
        # Focal term (1 - pt)^gamma
        focal_term = (1 - pt).pow(self.gamma)
        # Compute the loss
        focal_loss = focal_term * neg_log_pt # minus sign in the logpt

        if self.reduction == 'mean':
            loss =  focal_loss.mean()
        elif self.reduction == 'sum':
            loss =  focal_loss.sum()
        else:
            loss = focal_loss

        # print(f"Focal loss: {loss}")

        return loss