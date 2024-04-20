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

    def Dice_score(self, input, target, weighted=False):
        smooth = 1.

        # Apply softmax to input (model output)
        input = torch.softmax(input, dim=1)

        dice_score = 0.

        if not weighted:
            for class_index in range(input.size(1)):
                valid = (target != self.ignore_index)
                input_flat = input[:, class_index, :, :][valid].contiguous().view(-1)
                target_flat = (target == class_index)[valid].contiguous().view(-1) # binary target for class_index

                intersection = (input_flat * target_flat).sum()

                class_score = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
                dice_score += class_score

            mean_dice = dice_score/input.size(1) # average score over all classes

            return mean_dice
        
        if weighted:
            total_pixels = target.ne(self.ignore_index).sum().item()  # total number of valid pixels
    
            for class_index in range(input.size(1)):
                valid = (target != self.ignore_index)
                input_flat = input[:, class_index, :, :][valid].contiguous().view(-1)
                target_flat = (target == class_index)[valid].contiguous().view(-1)  # binary target for class_index
    
                intersection = (input_flat * target_flat).sum()
    
                class_score = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
                class_weight = target.eq(class_index).sum().item() / total_pixels
                dice_score += class_weight * class_score
    
            return dice_score

    def IoU_score(self, input, target, weighted=False):
        smooth = 1e-6

        # Apply softmax to input (model output)
        input = torch.softmax(input, dim=1)

        class_iou_scores = []

        for class_index in range(input.size(1)):
            valid = (target != self.ignore_index)
            input_flat = input[:, class_index, :, :][valid].contiguous().view(-1)
            target_flat = (target == class_index)[valid].contiguous().view(-1) # binary target for class_index

            intersection = (input_flat * target_flat).sum()

            class_score = (intersection + smooth) / (input_flat.sum() + target_flat.sum() - intersection + smooth)
            class_iou_scores.append(class_score)

        if not weighted:
            total_iou = sum(class_iou_scores)
            mean_iou = total_iou / len(class_iou_scores) # average score over all classes
            return mean_iou

        if weighted:
            weighted_iou = 0.0
            total_pixels = target.ne(self.ignore_index).sum().item() # total number of valid pixels
            for class_index, iou in enumerate(class_iou_scores):
                class_weight = target.eq(class_index).sum().item() / total_pixels
                weighted_iou += class_weight * iou

            return weighted_iou
