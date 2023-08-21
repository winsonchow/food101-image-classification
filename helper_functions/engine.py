"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_model(model, dataloader, loss_fn, optimizer, device):
    model.to(device) # Move the model to the specified device
    model.train() # Set the model to training mode
    
    # Initialize variables to keep track of loss and accuracy
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Iterate over the batches in the dataloader
    for inputs, targets in dataloader:
        # Move inputs and targets to the specified device
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad() # Zero the gradients in the optimizer
        outputs = model(inputs) # Perform the forward pass through the model
        loss = loss_fn(outputs, targets) # Compute the loss using the specified loss function
        loss.backward() # Perform backpropagation to compute gradients
        optimizer.step() # Update model weights using the optimizer
        
         # Calculate the total loss for the epoch
        total_loss += loss.item() * inputs.size(0)
        
        # Calculate the number of correct predictions for accuracy calculation
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        
        # Track the total number of samples processed
        total_samples += targets.size(0)
        
    # Calculate the average training loss and accuracy for the epoch
    train_loss = total_loss / len(dataloader.dataset)
    train_accuracy = correct_predictions / total_samples
    
    return train_loss, train_accuracy





def test_model(model, dataloader, loss_fn, device):
    model.to(device) # Move the model to the specified device
    model.eval() # Set the model to evaluation mode

    # Initialize variables to keep track of loss and accuracy
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Disable gradient calculation for inference
    with torch.no_grad():
        # Iterate over the batches in the dataloader
        for inputs, targets in dataloader:
            # Move inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs) # Perform the forward pass through the model
            loss = loss_fn(outputs, targets) # Compute the loss using the specified loss function

            # Calculate the total loss for the epoch
            total_loss += loss.item() * inputs.size(0)

            # Calculate the number of correct predictions for accuracy calculation
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()

            # Track the total number of samples processed
            total_samples += targets.size(0)

    # Calculate the average testing loss and accuracy for the dataset
    test_loss = total_loss / len(dataloader.dataset)
    test_accuracy = correct_predictions / total_samples

    return test_loss, test_accuracy





def train_and_test_model(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device, writer):
    # Move the model to the specified device
    model.to(device)

    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")
        
        # Training step
        train_loss, train_accuracy = train_model(model, train_dataloader, loss_fn, optimizer, device)
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        
        # Testing step
        test_loss, test_accuracy = test_model(model, test_dataloader, loss_fn, device)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)
        
        # Print evaluation metrics for the epoch
        print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
        print("=" * 50)
        
        # Log metrics to SummaryWriter
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Test', test_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/Test', test_accuracy, epoch)
        
            # Close the SummaryWriter
            writer.close()
        else:
            pass
    
    evaluation_metrics = {
        'train_loss': train_loss_list,
        'train_acc': train_accuracy_list,
        'test_loss': test_loss_list,
        'test_acc': test_accuracy_list
    }
    
    return evaluation_metrics