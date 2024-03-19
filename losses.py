import torch
import torch.nn as nn


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

            dice_loss = 1 - ((2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth))
        
        mean_dice = dice_loss/input.size(1) # average loss over all classes

        if self.log_loss:
            mean_dice = -torch.log(mean_dice)

        return mean_dice
