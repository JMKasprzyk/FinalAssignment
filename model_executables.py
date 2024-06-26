import torch
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim

import losses as L
import metrics as M

import wandb
import utils

import os

def train_model(model, train_loader, val_loader, num_epochs=5, patience=3, optimizer=None, criterion=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize best_val_loss here
    # best_val_loss = float('inf')

    # Create an instance of Metrics class
    metrics = M.Metrics(ignore_index=255)

    for epoch in range(num_epochs):
        # Initialize accuracy variables for each epoch
        total_correct_train, total_samples_train = 0, 0
        total_correct_val, total_samples_val = 0, 0

        running_train_loss = 0.0
        running_train_dice_score = 0.0
        running_train_iou_score = 0.0
        running_val_loss = 0.0
        running_val_dice_score = 0.0
        running_val_iou_score = 0.0

        # TRAINING
        model.train()
        for inputs, masks in train_loader:
            # Move inputs and masks to the GPU
            inputs, masks = inputs.to(device), masks.to(device)
            # FORWARD PASS
            outputs = model(inputs)
            # print(outputs.shape)
            masks = (masks*255).long().squeeze()     #*255 because the id are normalized between 0-1
            masks = utils.map_id_to_train_id(masks).to(device)
            loss = criterion(outputs, masks)
            # BACKWARD PASS
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            # Calcualte the metrics
            # Dice scrore
            dice_scrore = metrics.dice_score(outputs, masks)
            running_train_dice_score += dice_scrore.item()
            # IoU score
            iou_score = metrics.IoU_score(outputs, masks, weighted=True)
            running_train_iou_score += iou_score.item()
            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct_train += (predicted == masks.long().squeeze()).sum().item()
            total_samples_train += masks.numel()
        train_epoch_loss = running_train_loss / len(train_loader)
        train_epoch_dice_score = running_train_dice_score / len(train_loader)
        train_epoch_iou_score = running_train_iou_score / len(train_loader)
        # VALIDATION
        model.eval()
        with torch.no_grad():
            for val_inputs, val_masks in val_loader:
                # Move inputs and masks to the GPU
                val_inputs, val_masks = val_inputs.to(device), val_masks.to(device)
                val_outputs = model(val_inputs)
                val_masks = (val_masks*255).long().squeeze()     #*255 because the id are normalized between 0-1
                val_masks = utils.map_id_to_train_id(val_masks).to(device)
                val_loss = criterion(val_outputs, val_masks)
                running_val_loss += val_loss.item()
                # Calcualte the metrics
                # Dice score
                val_dice_score = metrics.dice_score(val_outputs, val_masks)
                running_val_dice_score += val_dice_score.item()
                # IoU score
                val_iou_score = metrics.IoU_score(val_outputs, val_masks, weighted=True)
                running_val_iou_score += val_iou_score.item()
                # Compute validation accuracy
                _, val_predicted = torch.max(val_outputs, 1)
                total_correct_val += (val_predicted == val_masks.long().squeeze()).sum().item()
                total_samples_val += val_masks.numel()
            epoch_val_loss = running_val_loss / len(val_loader)
            epoch_val_dice_score = running_val_dice_score / len(val_loader)
            epoch_val_iou_score = running_val_iou_score / len(val_loader)
        # # Early stopping functionality
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     num_consecutive_epoch_without_improve = 0
        # else:
        #     num_consecutive_epoch_without_improve += 1
        # if num_consecutive_epoch_without_improve >= patience:
        #     # Save the model to the checkpoint file
        #     save_checkpoint(model, optimizer, epoch, best_val_loss)
        #     print(f"EARLY STOPPING INVOKED: Training halted after {num_consecutive_epoch_without_improve} epochs without improvement.")
        #     break
        # Calculate and print accuracy
        accuracy_train = total_correct_train / total_samples_train
        accuracy_val = total_correct_val / total_samples_val
        # Write to the log file
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f},\
        Train Accuracy: {accuracy_train:.4f}, Train Dice Score: {train_epoch_dice_score:.4f},\
        Train IoU Score: {train_epoch_iou_score:.4f},\
        Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {accuracy_val:.4f},\
        Val Dice Score: {epoch_val_dice_score:.4f}, Val IoU Score: {epoch_val_iou_score:.4f}\n')

def train_model_noval(model, train_loader, num_epochs=5, lr=0.01, patience=3):
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize best_train_loss here
    best_train_loss = float('inf')

    # Create and open a text file
    with open('training_log.txt', 'w') as file:
        for epoch in range(num_epochs):
            # Initialize accuracy variables for each epoch
            total_correct_train, total_samples_train = 0, 0

            # TRAINING
            model.train()
            running_train_loss = 0.0
            for inputs, masks in train_loader:
                # Move inputs and masks to the GPU
                inputs, masks = inputs.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                masks = (masks*255).long().squeeze()     #*255 because the id are normalized between 0-1
                masks = utils.map_id_to_train_id(masks).to(device)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

                # Compute training accuracy
                _, predicted = torch.max(outputs, 1)
                total_correct_train += (predicted == masks.long().squeeze()).sum().item()
                total_samples_train += masks.numel()

            train_epoch_loss = running_train_loss / len(train_loader)

            # Early stopping functionality based on training loss
            if train_epoch_loss < best_train_loss:
                best_train_loss = train_epoch_loss
                num_consecutive_epoch_without_improve = 0
            else:
                num_consecutive_epoch_without_improve += 1

            if num_consecutive_epoch_without_improve >= patience:
                print(f"EARLY STOPPING INVOKED: Training halted after {num_consecutive_epoch_without_improve} epochs without improvement.")
                break

            # Calculate and print accuracy
            accuracy_train = total_correct_train / total_samples_train

            # Write to the log file
            file.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {accuracy_train:.4f}\n')

    # Saving the model after the entire training process went interupted
    save_checkpoint(model, optimizer, epoch, best_train_loss)


def train_model_wandb(model, train_loader, val_loader, num_epochs=5, criterion=None, optimizer=None, patience=3, checkpoint_dir='checkpoints'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # best_val_loss = float('inf')

    # Create an instance of Metrics class
    metrics = M.Metrics(ignore_index=255)

    for epoch in range(num_epochs):
        # Initialize accuracy variables for each epoch
        total_correct_train, total_samples_train = 0, 0
        total_correct_val, total_samples_val = 0, 0

        running_train_loss = 0.0
        running_train_dice_score = 0.0
        running_train_iou_score = 0.0
        running_val_loss = 0.0
        running_val_dice_score = 0.0
        running_val_iou_score = 0.0
        # TRAINING
        model.train()
        
        for inputs, masks in train_loader:
            # Move inputs and masks to the GPU
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            # FORWARD PASS
            outputs = model(inputs)
            # print(outputs.shape)
            masks = (masks*255).long().squeeze()     #*255 because the id are normalized between 0-1
            masks = utils.map_id_to_train_id(masks).to(device)
            # print(torch.unique(masks))
            loss = criterion(outputs, masks)
            # BACKWARD PASS
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            # Calcualte the metrics
            # Dice scrore
            dice_scrore = metrics.Dice_score(outputs, masks)
            running_train_dice_score += dice_scrore.item()
            # IoU score
            iou_score = metrics.IoU_score(outputs, masks, weighted=True)
            running_train_iou_score += iou_score.item()
            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct_train += (predicted == masks.long().squeeze()).sum().item()
            total_samples_train += masks.numel()
        train_epoch_loss = running_train_loss / len(train_loader)
        train_epoch_dice_score = running_train_dice_score / len(train_loader)
        train_epoch_iou_score = running_train_iou_score / len(train_loader)
        # VALIDATION
        model.eval()
        with torch.no_grad():
            for val_inputs, val_masks in val_loader:
                # Move inputs and masks to the GPU
                val_inputs, val_masks = val_inputs.to(device), val_masks.to(device)
                val_outputs = model(val_inputs)
                val_masks = (val_masks*255).long().squeeze()     #*255 because the id are normalized between 0-1
                val_masks = utils.map_id_to_train_id(val_masks).to(device)
                val_loss = criterion(val_outputs, val_masks)
                running_val_loss += val_loss.item()
                # Calcualte the metrics
                # Dice score
                val_dice_score = metrics.Dice_score(val_outputs, val_masks)
                running_val_dice_score += val_dice_score.item()
                # IoU score
                val_iou_score = metrics.IoU_score(val_outputs, val_masks, weighted=True)
                running_val_iou_score += val_iou_score.item()
                # Compute validation accuracy
                _, val_predicted = torch.max(val_outputs, 1)
                total_correct_val += (val_predicted == val_masks.long().squeeze()).sum().item()
                total_samples_val += val_masks.numel()
            epoch_val_loss = running_val_loss / len(val_loader)
            epoch_val_dice_score = running_val_dice_score / len(val_loader)
            epoch_val_iou_score = running_val_iou_score / len(val_loader)
        # Save checkpoint every 2nd epoch starting from the 50th epoch
        if (epoch + 1) % 2 == 0 and epoch >= 49:
            save_checkpoint(model,epoch, checkpoint_dir)

        # Calculate and print accuracy
        accuracy_train = total_correct_train / total_samples_train
        accuracy_val = total_correct_val / total_samples_val
        # Log metrics to wandb
        wandb.log({
            'Epoch': epoch + 1,
            'Train Loss': train_epoch_loss,
            'Train Accuracy': accuracy_train,
            'Val Loss': epoch_val_loss,
            'Val Accuracy': accuracy_val,
            'Train Dice Score': train_epoch_dice_score,
            'Train IoU Score': train_epoch_iou_score,
            'Val Dice Score': epoch_val_dice_score,
            'Val IoU Score': epoch_val_iou_score
        })
        # Write to the log file
        # file.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {accuracy_train:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {accuracy_val:.4f}\n')

def train_model_wandb_noval(model, train_loader, num_epochs=5, lr=0.01, patience=3):
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize best_train_loss here
    # best_train_loss = float('inf')
    previous_train_loss = float('inf')
    stagnation_count = 0


    # Create and open a text file
    for epoch in range(num_epochs):
        # Initialize accuracy variables for each epoch
        total_correct_train, total_samples_train = 0, 0
        # TRAINING
        model.train()
        running_train_loss = 0.0
        for inputs, masks in train_loader:
            # Move inputs and masks to the GPU
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            masks = (masks*255).long().squeeze()     #*255 because the id are normalized between 0-1
            masks = utils.map_id_to_train_id(masks).to(device)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct_train += (predicted == masks.long().squeeze()).sum().item()
            total_samples_train += masks.numel()
        train_epoch_loss = running_train_loss / len(train_loader)
        # Calculate and print accuracy
        accuracy_train = total_correct_train / total_samples_train
        # Log metrics to WandB
        wandb.log({
            'Epoch': epoch + 1,
            'Train Loss': train_epoch_loss,
            'Train Accuracy': accuracy_train
        })
        # Write to the log file
        # file.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {accuracy_train:.4f}\n')
        # EARLY STOPPING
        # Check if training loss stagnated over three consecutive epochs
        if train_epoch_loss >= previous_train_loss:
            stagnation_count += 1
        else:
            stagnation_count = 0  # Reset count if there's improvement
        # Update the previous training loss for the next iteration
        previous_train_loss = train_epoch_loss
        # Save checkpoint every 5th epoch
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model,epoch)
        if stagnation_count >= patience:
            # file.write(f"EARLY STOPPING INVOKED: Training halted after {epoch + 1} epochs without substantial improvement in training loss.")
            break

    # Saving the model after the entire training process went interupted
    save_checkpoint(model, epoch)

def save_checkpoint(model, epoch, checkpoint_folder='checkpoints'):
    # Create the folder if it doesn't exist
    os.makedirs(checkpoint_folder, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_folder, f'model_checkpoint_epoch_{epoch+1}.pth')
    torch.save( model.state_dict(), checkpoint_path)


