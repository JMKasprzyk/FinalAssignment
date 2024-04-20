import torch
import numpy as np
import matplotlib.pyplot as plt


import utils
from utils import LABELS

from metrics import Metrics


def mask_to_rgb(mask, class_to_color):
    """
    Converts a numpy mask with multiple classes indicated by integers to a color RGB mask.

    Parameters:
        mask (numpy.ndarray): The input mask where each integer represents a class.
        class_to_color (dict): A dictionary mapping class integers to RGB color tuples.

    Returns:
        numpy.ndarray: RGB mask where each pixel is represented as an RGB tuple.
    """
    # Get dimensions of the input mask
    height, width = mask.shape

    # Initialize an empty RGB mask
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Iterate over each class and assign corresponding RGB color
    for class_idx, color in class_to_color.items():
        # Mask pixels belonging to the current class
        class_pixels = mask == class_idx
        # Assign RGB color to the corresponding pixels
        rgb_mask[class_pixels] = color

    return rgb_mask

def renormalize_image(image):
    """
    Renormalizes the image to its original range.
    
    Args:
        image (numpy.ndarray): Image tensor to renormalize.
    
    Returns:
        numpy.ndarray: Renormalized image tensor.
    """
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]  
    renormalized_image = image * std + mean
    return renormalized_image

def visualize_segmentation_cityscapes(model, dataloader, num_examples=5, global_title='UNet'):

    # Create a mapping from trainId to color
    trainId_to_color_pred = {label.trainId: label.color for label in LABELS if label.trainId != 255}
    device = torch.device('cpu') # 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    metrics = Metrics()
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            if i >= num_examples:
                break

            outputs = model(images)
            print(outputs.shape)
            # outputs = torch.softmax(outputs, dim=1) # the metrics already aply the softmax inside the functions
            masks = (masks*255).long().squeeze()     #*255 because the id are normalized between 0-1
            masks = utils.map_id_to_train_id(masks).to(device)
            # Calculate scores
            iou_score = metrics.IoU_score(input=outputs, target=masks)
            weighted_iou_score = metrics.IoU_score(input=outputs, target=masks, weighted=True)
            dice_score = metrics.Dice_score(input=outputs, target=masks)
            weighted_dice_score = metrics.Dice_score(input=outputs, target=masks, weighted=True)

            outputs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, 1)

            images = images.cpu().numpy()

            masks = masks.cpu().numpy()
            predicted = predicted.cpu().numpy()

            for j in range(images.shape[0]):
                image = renormalize_image(images[j].transpose(1, 2, 0))

                mask = masks[j].squeeze()
                pred_mask = predicted[j]

                mask_rgb = mask_to_rgb(mask, trainId_to_color_pred)
                pred_mask_rgb = mask_to_rgb(pred_mask, trainId_to_color_pred)

                fig = plt.figure(figsize=(10, 5))
                plt.suptitle(f'{global_title}', fontweight='bold', fontsize=14)
                fig.text(0.5, 0.88, f'IoU: {iou_score:.2f}          weighted IoU: {weighted_iou_score:.2f}          Dice score: {dice_score:.2f}            weighted Dice: {weighted_dice_score:.2f}', ha='center', va='center', fontsize=12, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.title('Image')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(mask_rgb)
                plt.title('Ground Truth Mask')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(pred_mask_rgb)
                plt.title('Predicted Mask')
                plt.axis('off')

                plt.tight_layout()

                plt.show()