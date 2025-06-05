import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mnist import MNIST
import time

class SidePassNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=264, output_dim=10, side_layers=[128, 64]):
        super(SidePassNet, self).__init__()
        
        # Main network layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.main_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        ])
        
        # Calculate total features to collect from main network
        # We only need to concatenate after each activation layer
        self.main_features = input_dim + hidden_dim + hidden_dim + output_dim
        print(f"Side network input dimensions: {self.main_features}")
        
        # Side network layers
        side_dims = [self.main_features] + side_layers + [output_dim]
        self.side_layers = nn.ModuleList()
        
        for i in range(len(side_dims) - 1):
            self.side_layers.append(nn.Linear(side_dims[i], side_dims[i+1]))
            if i < len(side_dims) - 2:  # No activation after final layer
                self.side_layers.append(nn.Tanh())
        
        # Final activation for both paths
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        # Debug flag
        self.debug = False
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Backpropagation will automatically flow backward through this exact path:
        1. First through the side network layers (from last to first)
        2. Then through the main network layers (from last to first)
        3. All parameters in both paths will receive gradients automatically
        """
        batch_size = x.size(0)
        
        # First collect the input
        main_input = x.view(batch_size, -1)  # Flatten
        
        # Forward pass through main network layers and collect activations
        current = main_input
        post_layer1 = self.main_layers[0](current)  # Linear 1
        post_act1 = self.main_layers[1](post_layer1)  # Tanh 1
        post_layer2 = self.main_layers[2](post_act1)  # Linear 2  
        post_act2 = self.main_layers[3](post_layer2)  # Tanh 2
        main_output = self.main_layers[4](post_act2)  # Linear 3 (output)
        
        # Debug dimension issues
        if self.debug:
            print(f"main_input shape: {main_input.shape}")
            print(f"post_act1 shape: {post_act1.shape}")
            print(f"post_act2 shape: {post_act2.shape}")
            print(f"main_output shape: {main_output.shape}")
        
        # Concatenate all activations for side network
        all_features = torch.cat([
            main_input,     # Input (784)
            post_act1,      # After first hidden layer (264)
            post_act2,      # After second hidden layer (264) 
            main_output     # Final output (10)
        ], dim=1)
        
        if self.debug:
            print(f"all_features shape: {all_features.shape}")
            expected_dim = self.input_dim + self.hidden_dim + self.hidden_dim + self.output_dim
            print(f"Expected dimension: {expected_dim}")
        
        # Forward pass through side network
        side_current = all_features
        for i, layer in enumerate(self.side_layers):
            side_current = layer(side_current)
            if self.debug:
                print(f"After side layer {i}, shape: {side_current.shape}")
        
        # Apply log_softmax to both outputs
        main_output = self.log_softmax(main_output)
        side_output = self.log_softmax(side_current)
        
        # Return both outputs as a tuple - decide which to use during training
        return main_output, side_output
    
def train_model(model, train_loader, test_loader, device, epochs=5, learning_rate=0.01, 
                use_main_output=False, use_side_output=True, alpha=0.5):
    """
    Train the SidePassNet model with proper backpropagation.
    
    The backpropagation happens automatically in these steps:
    1. Calculate loss from outputs (main and/or side)
    2. Call loss.backward()
    3. PyTorch's autograd automatically propagates gradients through:
       - Side network (if side_output is used in loss)
       - Main network (if main_output is used, or due to dependency through side network)
    4. Update all parameters with optimizer.step()
    
    Parameters:
        model: The SidePassNet model
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        device: Device to run on (CPU, CUDA, or MPS)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        use_main_output: Whether to use the main path output in loss calculation
        use_side_output: Whether to use the side path output in loss calculation
        alpha: Weight for main_output in combined loss: alpha*main_loss + (1-alpha)*side_loss
    
    Returns:
        train_losses: List of training losses per epoch
        test_accuracies: List of test accuracies per epoch
    """
    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Move model to device
    model.to(device)
    
    train_losses = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        start_time = time.perf_counter()
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            main_output, side_output = model(data)
            
            # Calculate loss based on configuration
            if use_main_output and use_side_output:
                # Using both outputs with weighted combination
                main_loss = criterion(main_output, target)
                side_loss = criterion(side_output, target)
                loss = alpha * main_loss + (1 - alpha) * side_loss
            elif use_main_output:
                # Using only main output
                loss = criterion(main_output, target)
            else:  # use_side_output only
                # Using only side output (default)
                loss = criterion(side_output, target)
            
            # Backward pass - THIS IS WHERE BACKPROPAGATION HAPPENS
            # PyTorch automatically computes gradients for ALL parameters involved in computing the loss
            # This includes both side network and main network since they're connected
            loss.backward()
            
            # Optimize - update all parameters based on computed gradients
            optimizer.step()
            
            running_loss += loss.item()
        
        end_time = time.perf_counter()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Test accuracy
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                main_output, side_output = model(data)
                
                # Use side_output for prediction by default
                output = side_output if use_side_output else main_output
                
                # Get predictions
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        test_accuracies.append(accuracy)
        
        print(f'Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}, '
              f'Accuracy: {accuracy:.4f}, Time: {end_time - start_time:.2f}s')
    
    return train_losses, test_accuracies

def load_mnist_data(batch_size=128):
    """Load MNIST data and create DataLoader objects."""
    # Initialize MNIST data loader
    mndata = MNIST('/Users/andrewceniccola/Desktop/cajal/old_work/src/raw')

    # Load the training data
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    
    # Set device to MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert images to normalized float tensors
    train_images = torch.tensor(train_images, dtype=torch.float32, device=device) / 255.0
    test_images = torch.tensor(test_images, dtype=torch.float32, device=device) / 255.0
    
    # Convert labels to long tensors (not one-hot for NLLLoss)
    train_labels = torch.tensor(train_labels, dtype=torch.long, device=device)
    test_labels = torch.tensor(test_labels, dtype=torch.long, device=device)
    
    # Create datasets
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, device

def visualize_gradients(model):
    """
    Function to visualize the gradient flow through the network.
    This helps to understand how backpropagation affects both paths.
    """
    # Get all parameters
    main_params = []
    side_params = []
    
    for i, layer in enumerate(model.main_layers):
        if hasattr(layer, 'weight'):
            main_params.append((f"Main Layer {i} Weight", layer.weight))
    
    for i, layer in enumerate(model.side_layers):
        if hasattr(layer, 'weight'):
            side_params.append((f"Side Layer {i} Weight", layer.weight))
    
    # Print gradient info
    print("\nGradient Magnitudes:")
    
    print("\nMain Netexplanfwork:")
    for name, param in main_params:
        if param.grad is not None:
            print(f"{name}: {param.grad.abs().mean().item():.6f}")
        else:
            print(f"{name}: No gradient")
    
    print("\nSide Network:")
    for name, param in side_params:
        if param.grad is not None:
            print(f"{name}: {param.grad.abs().mean().item():.6f}")
        else:
            print(f"{name}: No gradient")

def compare_training_modes(model_class, train_loader, test_loader, device, epochs=5):
    """
    Compare different training configurations to demonstrate backpropagation effects.
    
    Parameters:
        model_class: The model class to instantiate
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        device: Device to run on
        epochs: Number of training epochs
    """
    configurations = [
        {
            "name": "Main Path Only",
            "use_main": True,
            "use_side": False,
            "alpha": 1.0
        },
        {
            "name": "Side Path Only",
            "use_main": False,
            "use_side": True,
            "alpha": 0.0
        },
        {
            "name": "Both Paths (50/50)",
            "use_main": True,
            "use_side": True,
            "alpha": 0.5
        },
        {
            "name": "Both Paths (70% Main, 30% Side)",
            "use_main": True,
            "use_side": True,
            "alpha": 0.7
        }
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\n--- Training with {config['name']} ---")
        
        # Create a new model instance for each configuration
        model = model_class().to(device)
        
        # Store initial weights
        initial_weights = {name: param.clone().detach() for name, param in model.named_parameters()}
        
        # Train the model with this configuration
        train_losses, test_accuracies = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            learning_rate=0.001,
            use_main_output=config["use_main"],
            use_side_output=config["use_side"],
            alpha=config["alpha"]
        )
        
        # Store results
        results[config["name"]] = {
            "final_accuracy": test_accuracies[-1],
            "accuracies": test_accuracies,
            "losses": train_losses
        }
        
        # --- Weight Comparison --- 
        print(f"\n--- Weight Comparison for {config['name']} after {epochs} epochs ---")
        for name, current_param in model.named_parameters():
            if current_param.requires_grad:
                initial_param = initial_weights[name]
                mean_abs_current = current_param.data.abs().mean().item()
                mean_abs_initial = initial_param.data.abs().mean().item()
                mean_abs_diff = (current_param.data - initial_param.data).abs().mean().item()
                print(f"Layer: {name}")
                print(f"  Initial Mean Abs Weight: {mean_abs_initial:.6f}")
                print(f"  Current Mean Abs Weight: {mean_abs_current:.6f}")
                print(f"  Mean Abs Difference:   {mean_abs_diff:.6f}")

        # Check gradient flow after final training
        print("\nGradient Flow Analysis:")
        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        main_output, side_output = model(data[:10])
        
        # Calculate loss according to the configuration
        criterion = nn.NLLLoss()
        if config["use_main"] and config["use_side"]:
            main_loss = criterion(main_output, target[:10])
            side_loss = criterion(side_output, target[:10])
            loss = config["alpha"] * main_loss + (1 - config["alpha"]) * side_loss
        elif config["use_main"]:
            loss = criterion(main_output, target[:10])
        else:  # use_side only
            loss = criterion(side_output, target[:10])
        
        # Calculate gradients
        model.zero_grad()
        loss.backward()
        
        # Analyze gradients
        main_grad_avg = 0
        side_grad_avg = 0
        main_count = 0
        side_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mag = param.grad.abs().mean().item()
                if "main_layers" in name:
                    main_grad_avg += grad_mag
                    main_count += 1
                elif "side_layers" in name:
                    side_grad_avg += grad_mag
                    side_count += 1
        
        if main_count > 0:
            main_grad_avg /= main_count
        if side_count > 0:
            side_grad_avg /= side_count
            
        print(f"Average Main Network Gradient: {main_grad_avg:.6f}")
        print(f"Average Side Network Gradient: {side_grad_avg:.6f}")
        print(f"Gradient Ratio (Main/Side): {main_grad_avg/side_grad_avg if side_grad_avg > 0 else 'N/A'}")
    
    # Print summary
    print("\n--- Final Results ---")
    for name, result in results.items():
        print(f"{name}: Accuracy = {result['final_accuracy']:.4f}")
    
    return results
    
if __name__ == "__main__":
    # Create model and parameters
    batch_size = 128
    epochs = 3
    
    # Load data
    train_loader, test_loader, device = load_mnist_data(batch_size)
    
    # First do a quick test to ensure the model works properly
    model = SidePassNet().to(device)
    model.debug = True
    
    print("\nVerifying model with a test forward pass...")
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    main_output, side_output = model(data[:5])  # Small batch for test
    print("Model test successful!")
    
    # Turn off debug and run the comparison
    model.debug = False
    print("\nRunning comparison of different backpropagation configurations...")
    
    # Compare different training configurations
    results = compare_training_modes(
        model_class=SidePassNet,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs
    )