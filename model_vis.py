import torch
import numpy as np
import matplotlib.pyplot as plt

import utils
from utils import LABELS


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

def visualize_segmentation_cityscapes(model, dataloader, num_examples=5):

    # Create a mapping from trainId to color
    trainId_to_color_pred = {label.trainId: label.color for label in LABELS if label.trainId != 255}
    # trainId_to_color_gt = {label.id: label.color for label in LABELS if label.trainId != 255}
    # trainId_to_name = {label.trainId: label.name for label in LABELS if label.trainId != 255}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            images, masks = images.to('cpu'), masks.to('cpu') # moving the images and masks tensors to CPU asuming the model was moved CPU
            if i >= num_examples:
                break
            
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, 1)

            images = images.numpy()
            masks = (masks*255).long().squeeze()     #*255 because the id are normalized between 0-1
            masks = utils.map_id_to_train_id(masks).to(device)
            masks = masks.cpu().numpy()
            predicted = predicted.cpu().numpy()

            for j in range(images.shape[0]):
                image = renormalize_image(images[j].transpose(1, 2, 0))

                mask = masks[j].squeeze()
                pred_mask = predicted[j]

                mask_rgb = mask_to_rgb(mask, trainId_to_color_pred)
                pred_mask_rgb = mask_to_rgb(pred_mask, trainId_to_color_pred)

                unique_classes_gt = np.unique(mask)
                unique_classes_pred = np.unique(pred_mask)

                unique_classes_gt = np.delete(unique_classes_gt, [0, -1])
                unique_classes_pred = np.delete(unique_classes_pred, 0)

                unique_classes_gt[unique_classes_gt == 255] = 0
                unique_classes_pred[unique_classes_pred == 255] = 0

                # classes_gt = [trainId_to_name[int(idx)] for idx in unique_classes_gt]
                # classes_pred = [trainId_to_name[int(idx)] for idx in unique_classes_pred]

                plt.figure(figsize=(10, 5))
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

                plt.show()