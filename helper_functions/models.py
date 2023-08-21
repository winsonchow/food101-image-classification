"""
Contains a PyTorch model code to instantiate a SimpleVGG model.
"""

import torch
from torch import nn
    
# class SimpleVGG(nn.Module):
#     """
#     Creates a Simple VGG architecture.
    
#     Args:
#         input_shape: An integer indicating number of input channels.
#         hidden_units: An integer indicating number of hidden units between layers.
#         output_shape: An integer indicating number of output units.
#     """    
#     def __init__(self, input_shape: int, hidden_units: int, output_shape: int, device: str) -> None:
#         super().__init__()
        
#         # Feature extraction layers
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels=input_shape,
#                       out_channels=hidden_units,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1,
#                       device=device),
            
#             nn.ReLU(inplace=True),
            
#             nn.MaxPool2d(kernel_size=2,
#                          stride=2),
            
#             nn.Conv2d(in_channels=hidden_units,
#                       out_channels=hidden_units,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1,
#                       device=device),
            
#             nn.ReLU(inplace=True),
            
#             nn.MaxPool2d(kernel_size=2,
#                          stride=2),
#         )
        
#         # Classifier layers (fully connected)
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=hidden_units*13*13, out_features=output_shape)
#         )
        
#     def forward(self, x):
#         # Forward pass through the features extraction layers
#         x = self.features(x)
        
#         # Forward pass through the classifier layers
#         x = self.classifier(x)
        
#         # Return the model
#         return x


class SimpleVGG(nn.Module):
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      padding=0),  
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13, out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

    
    
    
    
def feature_extractor(weights, base_model, base_model_name, num_classes, device):
    """
    Creates a feature extractor model.

    Args:
        weights (str): Pre-trained weights to initialize the base model.
        base_model (torch.nn.Module): Base model architecture to use.
        base_model_name (str): Name of the model
        num_classes (int): Number of classes for the new classifier head.
        device (str): Target device for the model.
        seed (int, optional): Seed for reproducibility. Default is 42.

    Returns:
        torch.nn.Module: Feature extractor model.
    """
    # Set seed for reproducibility
    # set_seed(seed)
    
    # Load pre-trained weights and send the base model to the target device
    model = base_model(pretrained=True, weights=weights).to(device)
    
    # Freeze base model layers
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Change the classifier head
    num_features = base_model.fc.in_features
    new_classifier = torch.nn.Linear(num_features, num_classes)
    model.fc = new_classifier
    
    # Set a custom name for the model
    model.name = base_model_name
    
    print(f"[INFO] Created new {model.name} model.")
    return base_model