from mnist import MNIST
import random

# Initialize MNIST data loader
mndata = MNIST('/Users/andrewceniccola/Desktop/cajal/MNIST/raw')

# Load the training data
images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# Get a random index
index = random.randrange(0, len(images)) 

# Display the image
print(mndata.display(images[index])) 