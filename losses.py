import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, ingore_index=255):
        super(DiceLoss, self).__init__()
        self.ingore_index = ingore_index

    def forward(self, input, target):
        smooth = 1.

        valid = (target != self.ingore_index)
        input_flat = input[valid].contigous().view(-1)
        target_flat = target[valid].contigous().view(-1)

        intersection = (input_flat * target_flat).sum()

        dice_loss = 1 - ((2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth))
        
        return 1 - dice_loss
    
class DiceBCELoss(nn.Module):
    def __init__(self, ingore_index=255):
        super(DiceBCELoss, self).__init__()
        self.ingore_index = ingore_index
        self.dice = DiceLoss(ingore_index)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        dice_loss = self.dice(input, target)
        BCE_loss = self.bce(input, target)

        return dice_loss + BCE_loss