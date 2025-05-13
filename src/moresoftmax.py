import numpy as np

class NeuralNetworkLayer:
    """
    A conceptual class representing a layer in a neural network,
    specifically focusing on the softmax output and its derivative
    in the context of the user's backpropagation formula.
    """
    def __init__(self, temperature: float = 1.0):
        """
        Initializes the layer with a given temperature.

        Args:
            temperature (float): The temperature for the softmax function.
                                 Must be positive. Defaults to 1.0.
        """
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError("Temperature must be a positive number.")
        self.temperature = float(temperature)
        # In a full network, self.outputs would store activations.
        # self.outputs[-1] would be the softmax probabilities from the output layer.

    def set_temperature(self, temperature: float):
        """Updates the temperature for the layer."""
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError("Temperature must be a positive number.")
        self.temperature = float(temperature)

    def softmax_derivative(self, softmax_probabilities: np.ndarray) -> float:
        """
        Calculates the factor for the output layer's delta calculation.

        This function is designed to be used in the specific formula:
        deltas_output_layer = (y_true - softmax_outputs) * self.softmax_derivative(softmax_outputs)

        To ensure that deltas_output_layer correctly represents (softmax_outputs - y_true) / temperature,
        which is the standard derivative of Categorical Cross-Entropy loss with respect to
        the logits of a softmax layer (dL/dZ), this function must return -1.0 / self.temperature.

        Args:
            softmax_probabilities (np.ndarray): The output probabilities from the softmax layer.
                While part of the signature based on the calling context `self.softmax_derivative(self.outputs[-1])`,
                this specific input is not used in the calculation of -1.0 / T. It could be used
                for input validation (e.g., checking shape or sum) if desired.

        Returns:
            float: The scaling factor -1.0 / self.temperature.
        """
        # Validate the input type, though it's not used in the calculation itself.
        if not isinstance(softmax_probabilities, np.ndarray):
            # In a real scenario, you might raise an error or handle different input types.
            # For this specific derivative, we'll just proceed.
            print("Warning: softmax_probabilities input type is not np.ndarray, but it's unused in this derivative.")

        return -1.0 / self.temperature

    # --- Helper function for testing: Apply softmax (not strictly part of the derivative) ---
    def apply_softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Computes the softmax of a vector with the layer's temperature.
        (Provided for context and to generate test inputs).
        """
        if not isinstance(logits, np.ndarray):
            raise TypeError("Logits must be a NumPy array.")
        
        scaled_logits = logits / self.temperature
        # Subtract max for numerical stability
        max_logit = np.max(scaled_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_logit)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return softmax_probs

# --- Test for the softmax_derivative function in the given context ---
def run_contextual_softmax_derivative_test():
    """
    Tests that self.softmax_derivative, when used in the user's specified
    delta calculation, produces the standard dL/dZ for CCE with Softmax.
    """
    print("Running Contextual Softmax Derivative Test...\n")

    # Test Case 1: Standard Temperature T=1.0
    print("--- Test Case 1: Temperature = 1.0 ---")
    temp1 = 1.0
    layer1 = NeuralNetworkLayer(temperature=temp1)

    # Example: 2 samples, 3 classes
    logits1 = np.array([[2.0, 1.0, 0.1],  # Sample 1 logits
                        [0.5, 1.5, 1.0]]) # Sample 2 logits
    y_true1 = np.array([[1, 0, 0],        # Sample 1 true label (one-hot)
                        [0, 1, 0]])       # Sample 2 true label (one-hot)

    # These would be self.outputs[-1] in the network
    softmax_outputs_S1 = layer1.apply_softmax(logits1)
    print(f"Softmax Outputs (S) for T={temp1}:\n{softmax_outputs_S1}")
    print(f"True Labels (y):\n{y_true1}")

    # Expected delta (dL/dZ) for CCE with Softmax: (S - y) / T
    expected_deltas1 = (softmax_outputs_S1 - y_true1) / temp1
    print(f"Expected deltas (S - y) / T:\n{expected_deltas1}")

    # Calculate using the user's formula structure
    # deltas[-1] = (y - self.outputs[-1]) * self.softmax_derivative(self.outputs[-1])
    derivative_factor1 = layer1.softmax_derivative(softmax_outputs_S1)
    print(f"Calculated softmax_derivative factor (-1/T): {derivative_factor1}")
    actual_deltas1 = (y_true1 - softmax_outputs_S1) * derivative_factor1
    print(f"Actual deltas using (y - S) * softmax_derivative(S):\n{actual_deltas1}")

    assert np.allclose(actual_deltas1, expected_deltas1), "Test Case 1 FAILED"
    print("Test Case 1 PASSED\n")


    # Test Case 2: Higher Temperature T=2.5
    print("--- Test Case 2: Temperature = 2.5 ---")
    temp2 = 2.5
    layer2 = NeuralNetworkLayer(temperature=temp2)

    logits2 = np.array([[3.0, 0.0, 1.0]]) # 1 sample, 3 classes
    y_true2 = np.array([[0, 0, 1]])       # True label

    softmax_outputs_S2 = layer2.apply_softmax(logits2)
    print(f"Softmax Outputs (S) for T={temp2}:\n{softmax_outputs_S2}")
    print(f"True Labels (y):\n{y_true2}")

    expected_deltas2 = (softmax_outputs_S2 - y_true2) / temp2
    print(f"Expected deltas (S - y) / T:\n{expected_deltas2}")

    derivative_factor2 = layer2.softmax_derivative(softmax_outputs_S2)
    print(f"Calculated softmax_derivative factor (-1/T): {derivative_factor2}")
    actual_deltas2 = (y_true2 - softmax_outputs_S2) * derivative_factor2
    print(f"Actual deltas using (y - S) * softmax_derivative(S):\n{actual_deltas2}")

    assert np.allclose(actual_deltas2, expected_deltas2), "Test Case 2 FAILED"
    print("Test Case 2 PASSED\n")

    # Test Case 3: Lower Temperature T=0.5
    print("--- Test Case 3: Temperature = 0.5 ---")
    temp3 = 0.5
    layer3 = NeuralNetworkLayer(temperature=temp3)

    # Using same logits and y_true as Test Case 1 for comparison
    softmax_outputs_S3 = layer3.apply_softmax(logits1) # logits1 from TC1
    print(f"Softmax Outputs (S) for T={temp3}:\n{softmax_outputs_S3}")
    print(f"True Labels (y) (same as TC1):\n{y_true1}")

    expected_deltas3 = (softmax_outputs_S3 - y_true1) / temp3
    print(f"Expected deltas (S - y) / T:\n{expected_deltas3}")

    derivative_factor3 = layer3.softmax_derivative(softmax_outputs_S3)
    print(f"Calculated softmax_derivative factor (-1/T): {derivative_factor3}")
    actual_deltas3 = (y_true1 - softmax_outputs_S3) * derivative_factor3
    print(f"Actual deltas using (y - S) * softmax_derivative(S):\n{actual_deltas3}")

    assert np.allclose(actual_deltas3, expected_deltas3), "Test Case 3 FAILED"
    print("Test Case 3 PASSED\n")

    print("All contextual tests completed successfully.")

if __name__ == "__main__":
    run_contextual_softmax_derivative_test()
