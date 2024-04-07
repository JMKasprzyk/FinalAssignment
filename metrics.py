import torch



# def IoU_score(target, prediction, ignore_index=255):
#     """
#     The implementation of the Intersection over Union (IoU) score, based on the paper: https://fse.studenttheses.ub.rug.nl/18139/1/AI_BA_2018_FlorisvanBeers.pdf Equation (2.3)
#     Args:
#         target (_type_): _description_
#         prediction (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     mask = target != ignore_index
#     T = target[mask].flatten() # => T
#     P = prediction[mask].flatten() # => P

#     intersection = torch.sum(T * P)
#     epsilon = 0.1
#     IoU = (intersection + epsilon) / (torch.sum(T) + torch.sum(P) - intersection + epsilon)

#     return IoU

class Metrics:
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index

    def dice_score(self, input, target):
        smooth = 1.

        # Apply softmax to input (model output)
        input = torch.softmax(input, dim=1)

        dice_score = 0.

        for class_index in range(input.size(1)):
            valid = (target != self.ignore_index)
            input_flat = input[:, class_index, :, :][valid].contiguous().view(-1)
            target_flat = (target == class_index)[valid].contiguous().view(-1) # binary target for class_index

            intersection = (input_flat * target_flat).sum()

            class_score = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
            dice_score += class_score

        mean_dice = dice_score/input.size(1) # average score over all classes

        return mean_dice

    def IoU_score(self, input, target):
        smooth = 1e-6

        # Apply softmax to input (model output)
        input = torch.softmax(input, dim=1)

        iou_score = 0.

        for class_index in range(input.size(1)):
            valid = (target != self.ignore_index)
            input_flat = input[:, class_index, :, :][valid].contiguous().view(-1)
            target_flat = (target == class_index)[valid].contiguous().view(-1) # binary target for class_index

            intersection = (input_flat * target_flat).sum()

            class_score = (intersection + smooth) / (input_flat.sum() + target_flat.sum() - intersection + smooth)
            iou_score += class_score

        mean_iou = iou_score/input.size(1) # average score over all classes

        return mean_iou