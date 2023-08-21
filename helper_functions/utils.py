import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime




def save_model(model, target_dir, model_name):
    """
    Saves a PyTorch model to a target directory.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        target_dir (str): The directory where the model will be saved.
        model_name (str): The filename for the saved model. Should include either ".pth" or ".pt" as the file extension.
    """
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Define the complete path to save the model
    model_path = os.path.join(target_dir, model_name)
    
    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved to {model_path}")

    
    
    

def create_writer(experiment_name, model_name, extra=None):
    # Get the current date in YYYY-MM-DD format
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Construct the log directory path
    log_dir = os.path.join("runs", current_date, experiment_name, model_name)
    if extra:
        log_dir = os.path.join(log_dir, extra)

    
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    
    # Create and return the SummaryWriter instance
    return SummaryWriter(log_dir)