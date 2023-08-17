"""
Contains a PyTorch model code to instantiate a SimpleVGG model.
"""

import torch
from torch import nn
    
class SimpleVGG(nn.Module):
    """
    Creates a Simple VGG architecture.
    
    Args:
        input_shape: An integer indicating number of input channels.
        hidden_units: An integer indicating number of hidden units between layers.
        output_shape: An integer indicating number of output units.
    """
    
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        
        # Classifier layers (fully connected)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13, out_features=output_shape)
        )
        
    def forward(self, x):
        # Forward pass through the features extraction layers
        x = self.features(x)
        
        # Forward pass through the classifier layers
        x = self.classifier(x)
        
        # Return the model
        return x