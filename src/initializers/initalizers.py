import numpy as np
import random
from typing import Callable

def init_weights(inp: int, outp: int, technique: str = "xavier", distribution: str = "uniform") -> np.ndarray:
    """
    Initialize weights matrix using specified technique and distribution.
    
    Args:
        inp: Number of input neurons
        outp: Number of output neurons
        technique: Weight initialization technique ("xavier" or "he")
        distribution: Distribution type ("uniform" or "normal")
        
    Returns:
        numpy.ndarray: Initialized weight matrix of shape (inp, outp)
    """
    if technique == "xavier":
        rand_func = xavier(inp, outp, distribution)
    elif technique == "he":
        rand_func = he(inp, distribution)
    else:
        raise ValueError(f"Invalid technique: {technique}")
    
    # Create weight matrix using vectorized operation
    weights = np.zeros((inp, outp))
    for i in range(inp):
        for j in range(outp):
            weights[i, j] = rand_func()
    
    return weights 

def xavier(inp: int, outp: int, distribution: str = "uniform") -> Callable[[], float]:
    """
    Xavier/Glorot initialization function generator. For sigmoid and tanh activation functions (0 mean activations). 
    
    Args:
        inp: Number of input neurons
        outp: Number of output neurons
        distribution: Distribution type ("uniform" or "normal")
        
    Returns:
        Callable: Function that generates random numbers according to Xavier initialization
    """
    
    
    if distribution == "uniform":
        def rand_func():
            r = np.sqrt(6 / (inp + outp))
            return random.uniform(-r, r)
    elif distribution == "normal":
        def rand_func():
            r = np.sqrt(2 / (inp + outp))
            return random.gauss(0, r)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")
    
    return rand_func

def he(inp: int, distribution: str = "uniform") -> Callable[[], float]:
    """
    He initialization function generator. For ReLU activation functions (non-zero mean activations). 
    
    Args:
        inp: Number of input neurons
        distribution: Distribution type ("uniform" or "normal")
        
    Returns:
        Callable: Function that generates random numbers according to He initialization
    """
    
    if distribution == "uniform":
        def rand_func():
            r = np.sqrt(6 / inp)
            return random.uniform(-r, r)
    elif distribution == "normal":
        def rand_func():
            r = np.sqrt(2 / inp)
            return random.gauss(0, r)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")
    
    return rand_func

if __name__ == "__main__":
    # Test the initialization
    weights = init_weights(6, 6)
    print(weights) 