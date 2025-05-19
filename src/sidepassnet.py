import torch
import torch.nn as nn

class SidePassNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=264, output_dim=10, side_layers=[128, 64]):
        super(SidePassNet, self).__init__()
        
        # Main network layers
        self.main_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        ])
        
        # Calculate total features to collect from main network
        # We'll concat outputs from all layers
        self.main_features = input_dim + hidden_dim + hidden_dim + output_dim
        
        # Side network layers
        side_dims = [self.main_features] + side_layers + [output_dim]
        self.side_layers = nn.ModuleList()
        
        for i in range(len(side_dims) - 1):
            self.side_layers.append(nn.Linear(side_dims[i], side_dims[i+1]))
            if i < len(side_dims) - 2:  # No activation after final layer
                self.side_layers.append(nn.Tanh())
        
        # Final activation for both paths
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        # Store all layer outputs for side network
        layer_outputs = [x]  # Start with input
        
        # Forward pass through main network layers
        current = x
        for i, layer in enumerate(self.main_layers):
            current = layer(current)
            layer_outputs.append(current)
        
        main_output = current
        
        # Concatenate all layer outputs for side network
        all_features = torch.cat([output.view(output.size(0), -1) for output in layer_outputs], dim=1)
        
        # Forward pass through side network
        side_current = all_features
        for layer in self.side_layers:
            side_current = layer(side_current)
        
        # Apply log_softmax to both outputs
        main_output = self.log_softmax(main_output)
        side_output = self.log_softmax(side_current)
        
        # Return both outputs (or you could average them, or just return the side output)
        return side_output
    
if __name__ == "__main__":
    nn = SidePassNet()
    print(nn)