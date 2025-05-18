from mnist import MNIST
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Neural network for MNIST handwriting recognition
# Optimized for GPU usage on M3 Mac with MPS backend

class MNISTNet(nn.Module):
    """
    Neural network for MNIST classification using PyTorch nn.Module
    Optimized for GPU acceleration on M3 Mac using MPS backend
    """
    def __init__(self, input_dim=784, hidden_dim=264, output_dim=10, dropout_rate=0.2):
        super(MNISTNet, self).__init__()
        
        # Network architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.LogSoftmax(dim=1)
        )
        
        # Initialize weights using He initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, test_loader, device, epochs=1, learning_rate=0.01):
    """
    Train the model using optimized GPU operations
    
    Parameters:
        model: PyTorch model
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        device: Device to run the model on (CPU or MPS)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    
    Returns:
        accuracy: Accuracy on test set
        train_time: Time taken for training
        inference_time: Time taken for inference
    """
    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Move model to device
    model.to(device)
    
    # Train the model
    model.train()
    train_start = time.perf_counter()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")
    
    train_end = time.perf_counter()
    train_time = train_end - train_start
    
    # Test the model
    model.eval()
    inference_start = time.perf_counter()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
    
    inference_end = time.perf_counter()
    inference_time = inference_end - inference_start
    
    accuracy = correct / total
    
    return accuracy, train_time, inference_time

def benchmark_batch_sizes(batch_sizes=[32, 64, 128, 256, 512, 1024]):
    """
    Benchmark different batch sizes to find the optimal configuration for M3 GPU
    
    Parameters:
        batch_sizes: List of batch sizes to test
    """
    # Initialize MNIST data loader
    mndata = MNIST('/Users/andrewceniccola/Desktop/cajal/MNIST/raw')

    # Load the training data
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    
    # Set device to MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running benchmark on device: {device}")
    
    # Convert images to normalized float tensors
    train_images = torch.tensor(train_images, dtype=torch.float32, device=device) / 255.0
    test_images = torch.tensor(test_images, dtype=torch.float32, device=device) / 255.0
    
    # One-hot encode labels
    train_label_indices = torch.tensor([train_labels], device=device).t()
    train_labels = torch.zeros(len(train_labels), 10, device=device)
    train_labels.scatter_(1, train_label_indices, 1)
    
    test_label_indices = torch.tensor([test_labels], device=device).t()
    test_labels = torch.zeros(len(test_labels), 10, device=device)
    test_labels.scatter_(1, test_label_indices, 1)
    
    print(f"Data loaded: {len(train_images)} training samples, {len(test_images)} test samples")
    
    # Store results for each batch size
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing batch size: {batch_size} ---")
        
        # Create data loaders with current batch size
        train_dataset = TensorDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        test_dataset = TensorDataset(test_images, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        # Create model
        model = MNISTNet(input_dim=784, hidden_dim=264, output_dim=10)
        
        # Run training and measure performance
        accuracy, train_time, inference_time = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=1,
            learning_rate=0.01
        )
        
        # Store results
        results[batch_size] = {
            'accuracy': accuracy,
            'train_time': train_time,
            'inference_time': inference_time,
            'total_time': train_time + inference_time
        }
        
        print(f"Batch size {batch_size}: Accuracy = {accuracy:.4f}, Train time: {train_time:.4f}s, Inference time: {inference_time:.4f}s")
    
    # Find optimal batch size
    optimal_batch_size = min(results.keys(), key=lambda x: results[x]['total_time'])
    
    print("\n--- Benchmark Results ---")
    print(f"Optimal batch size for M3 Mac GPU: {optimal_batch_size}")
    print(f"Optimal performance: Accuracy = {results[optimal_batch_size]['accuracy']:.4f}, Time = {results[optimal_batch_size]['total_time']:.4f}s")
    
    # Plot results
    batch_sizes = list(results.keys())
    train_times = [results[bs]['train_time'] for bs in batch_sizes]
    inference_times = [results[bs]['inference_time'] for bs in batch_sizes]
    accuracies = [results[bs]['accuracy'] for bs in batch_sizes]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot training and inference times
    ax1.plot(batch_sizes, train_times, 'b-o', label='Training Time')
    ax1.plot(batch_sizes, inference_times, 'r-o', label='Inference Time')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Training and Inference Times vs. Batch Size on M3 Mac GPU')
    ax1.grid(True)
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(batch_sizes, accuracies, 'g-o')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs. Batch Size')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('batch_size_benchmark.png')
    print("Benchmark plot saved to batch_size_benchmark.png")
    
    return results, optimal_batch_size

def run_mnist_benchmark(batch_size=256):
    """Train an MNIST model with GPU acceleration on M3 Mac."""
    tries = 5  # Reduced number of tries to demonstrate performance
    
    # Initialize MNIST data loader
    mndata = MNIST('/Users/andrewceniccola/Desktop/cajal/MNIST/raw')

    # Load the training data
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    
    # Set device to MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert images to normalized float tensors
    train_images = torch.tensor(train_images, dtype=torch.float32, device=device) / 255.0
    test_images = torch.tensor(test_images, dtype=torch.float32, device=device) / 255.0
    
    # One-hot encode labels
    train_label_indices = torch.tensor([train_labels], device=device).t()
    train_labels = torch.zeros(len(train_labels), 10, device=device)
    train_labels.scatter_(1, train_label_indices, 1)
    
    test_label_indices = torch.tensor([test_labels], device=device).t()
    test_labels = torch.zeros(len(test_labels), 10, device=device)
    test_labels.scatter_(1, test_label_indices, 1)
    
    # Create data loaders for batch processing
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    print(f"Data loaded: {len(train_images)} training samples, {len(test_images)} test samples")
    
    # Track benchmark results
    accuracies = []
    train_times = []
    inference_times = []
    
    start_time = time.perf_counter()
    
    for i in range(tries):
        # Set seed for reproducibility
        torch.manual_seed(1200 + i)
        
        # Create a model instance
        model = MNISTNet(input_dim=784, hidden_dim=264, output_dim=10)
        
        # Train and evaluate model
        accuracy, train_time, inference_time = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=1,
            learning_rate=0.01
        )
        
        accuracies.append(accuracy)
        train_times.append(train_time)
        inference_times.append(inference_time)
        
        print(f"Trial {i+1}: Accuracy = {accuracy:.4f}, Train time: {train_time:.4f}s, Inference time: {inference_time:.4f}s")
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Print performance results
    print("\n--- Performance Results ---")
    print(f"Average accuracy: {sum(accuracies)/len(accuracies):.4f}")
    print(f"Average train time: {sum(train_times)/len(train_times):.4f} seconds")
    print(f"Average inference time: {sum(inference_times)/len(inference_times):.4f} seconds")
    print(f"Total execution time: {total_time:.4f} seconds")
    
    # Performance comparison with original implementation
    print("\n--- GPU Optimization Results ---")
    print(f"Batch size: {batch_size}")
    print(f"Using PyTorch nn.Module and optimized operations")
    print(f"Using MPS backend on M3 Mac")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST Training on M3 Mac GPU')
    parser.add_argument('--benchmark', action='store_true', help='Run batch size benchmark')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_batch_sizes()
    else:
        run_mnist_benchmark(batch_size=args.batch_size)

'''# Initialize MNIST data loader
mndata = MNIST('/Users/andrewceniccola/Desktop/cajal/MNIST/raw')

# Load the training data
images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# Get a random index
index = random.randrange(0, len(images)) 

# Display the image
print(mndata.display(images[index])) '''