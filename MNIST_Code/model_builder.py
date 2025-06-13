from noise import apply_noise

import torch

from torch import nn


class WSmodel(nn.Module):
    """
    Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, noise: float):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(in_features=input_shape, out_features=hidden_units, bias=False),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_units, out_features=output_shape, bias=False),
            # No Sigmoid here, BCEWithLogitsLoss applies it automatically
            # nn.Sigmoid(),
        )
        self.noise = noise

    def forward(self, x: torch.Tensor):
        
        #print("Before noise - Input Tensor (X):", x)
        x = apply_noise(x, noise_range=(1-self.noise, 1+self.noise))
        #print("After noise in x:", x)

        # Forward pass through the layer stack with noise applied
        x = self.layer_stack[0](x)  # Flatten
        x = apply_noise(self.layer_stack[1](x), noise_range=(1-self.noise, 1+self.noise))  # Linear layer with noise
        #print("After noise in linear:", x)

        x = apply_noise(self.layer_stack[2](x), noise_range=(1-self.noise, 1+self.noise))  # ReLU with noise
        #print("After noise in ReLU:", x)

        # Apply noise to classifier layer
        x = apply_noise(self.classifier[0](x), noise_range=(1-self.noise, 1+self.noise))
        #print("After noise in linear in classifier:", x)

        return x
    

    
    
