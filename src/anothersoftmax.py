import numpy as np

class SoftmaxActivationLayer:
    """
    A class representing a layer with softmax activation,
    or a component that handles softmax activation and its derivative.
    """
    def __init__(self, temperature: float = 1.0):
        """
        Initializes the layer with a given temperature.

        Args:
            temperature (float): The temperature for the softmax function. Must be positive.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature
        # In a full network, self.outputs[i] would likely be stored after a forward pass.
        # For this function, the relevant output (softmax probabilities) is passed as an argument.

    def set_temperature(self, temperature: float):
        """Updates the temperature for the layer."""
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature

    def activation_function_derivative(self, softmax_probabilities: np.ndarray) -> np.ndarray:
        """
        Computes the element-wise derivative component S_i * (1 - S_i) / T.

        This value corresponds to the diagonal elements of the Jacobian matrix of the
        softmax function (dS_i/dz_i), each divided by the temperature T.
        Mathematically, for each probability S_i in softmax_probabilities,
        it calculates S_i * (1 - S_i) / T.

        Args:
            softmax_probabilities (np.ndarray): A NumPy array of probabilities,
                which are the output of the softmax function (S). This would correspond
                to `self.outputs[i]` in your backpropagation formula if layer `i`
                is this softmax layer.

        Returns:
            np.ndarray: A NumPy array of the same shape as softmax_probabilities,
                        containing the S_i * (1 - S_i) / T values.

        Important Note on Usage in Backpropagation:
        The formula `deltas[i-1] = np.dot(deltas[i], self.weights[i].T) * self.activation_function_derivative(self.outputs[i])`
        assumes an element-wise multiplication with the activation derivative.
        If `self.outputs[i]` are softmax probabilities, using only this diagonal component
        is a simplification. Standard backpropagation through softmax either uses the
        full Jacobian matrix or a specific simplified form when combined with a loss
        function like Cross-Entropy (e.g., dL/dZ = (S - Y)/T for logits Z).
        Ensure this simplification is appropriate for your neural network architecture
        and learning algorithm.
        """
        if not isinstance(softmax_probabilities, np.ndarray):
            raise TypeError("Input 'softmax_probabilities' must be a NumPy array.")
        if np.any(softmax_probabilities < 0) or np.any(softmax_probabilities > 1):
            # Softmax probabilities should be between 0 and 1.
            # Allowing for small floating point inaccuracies.
            if not np.allclose(np.sum(softmax_probabilities, axis=-1), 1.0, atol=1e-7) and softmax_probabilities.size > 0 :
                 print(f"Warning: Input 'softmax_probabilities' may not be valid probabilities (sum={np.sum(softmax_probabilities, axis=-1)}).")


        # Element-wise computation: S * (1 - S) / T
        derivative_values = softmax_probabilities * (1 - softmax_probabilities) / self.temperature
        return derivative_values

    # Example softmax function (for context and testing)
    def apply_softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Computes the softmax of a vector with the layer's temperature.
        """
        scaled_logits = logits / self.temperature
        # Subtract max logit for numerical stability
        max_logit = np.max(scaled_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_logit)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return softmax_probs

# --- Test for the activation_function_derivative ---
def run_softmax_derivative_test():
    """
    Tests the activation_function_derivative method for correctness.
    """
    print("Running Softmax Derivative Component Test...\n")

    # Test Case 1
    print("--- Test Case 1 ---")
    layer1 = SoftmaxActivationLayer(temperature=1.0)
    # Assume these are the outputs of the softmax function (self.outputs[i])
    s1 = np.array([0.7, 0.2, 0.1])
    print(f"Softmax Probabilities (S): {s1}")
    print(f"Temperature (T): {layer1.temperature}")

    expected_derivative1 = s1 * (1 - s1) / layer1.temperature
    # Expected: [0.7*0.3/1.0, 0.2*0.8/1.0, 0.1*0.9/1.0] = [0.21, 0.16, 0.09]
    print(f"Expected derivative S(1-S)/T: {expected_derivative1}")

    actual_derivative1 = layer1.activation_function_derivative(s1)
    print(f"Actual derivative S(1-S)/T:   {actual_derivative1}")

    assert np.allclose(actual_derivative1, expected_derivative1), "Test Case 1 FAILED"
    print("Test Case 1 PASSED\n")

    # Test Case 2: Different temperature and probabilities
    print("--- Test Case 2 ---")
    layer2 = SoftmaxActivationLayer(temperature=2.0)
    s2 = np.array([0.8, 0.1, 0.05, 0.05])
    print(f"Softmax Probabilities (S): {s2}")
    print(f"Temperature (T): {layer2.temperature}")

    expected_derivative2 = s2 * (1 - s2) / layer2.temperature
    # Expected:
    # 0.8 * 0.2 / 2.0 = 0.16 / 2.0 = 0.08
    # 0.1 * 0.9 / 2.0 = 0.09 / 2.0 = 0.045
    # 0.05 * 0.95 / 2.0 = 0.0475 / 2.0 = 0.02375
    # Result: [0.08, 0.045, 0.02375, 0.02375]
    print(f"Expected derivative S(1-S)/T: {expected_derivative2}")

    actual_derivative2 = layer2.activation_function_derivative(s2)
    print(f"Actual derivative S(1-S)/T:   {actual_derivative2}")

    assert np.allclose(actual_derivative2, expected_derivative2), "Test Case 2 FAILED"
    print("Test Case 2 PASSED\n")

    # Test Case 3: Batch of probabilities (e.g., multiple samples)
    print("--- Test Case 3 (Batch Processing) ---")
    layer3 = SoftmaxActivationLayer(temperature=0.5)
    # Batch of 2 samples, 3 classes each
    s3_batch = np.array([
        [0.6, 0.3, 0.1],  # Sample 1
        [0.1, 0.8, 0.1]   # Sample 2
    ])
    # Ensure they sum to 1 (approx) for each sample
    # s3_batch[0] = s3_batch[0] / np.sum(s3_batch[0])
    # s3_batch[1] = s3_batch[1] / np.sum(s3_batch[1])


    print(f"Softmax Probabilities (S) (Batch): \n{s3_batch}")
    print(f"Temperature (T): {layer3.temperature}")

    expected_derivative3_batch = s3_batch * (1 - s3_batch) / layer3.temperature
    print(f"Expected derivative S(1-S)/T (Batch): \n{expected_derivative3_batch}")

    actual_derivative3_batch = layer3.activation_function_derivative(s3_batch)
    print(f"Actual derivative S(1-S)/T (Batch):   \n{actual_derivative3_batch}")

    assert np.allclose(actual_derivative3_batch, expected_derivative3_batch), "Test Case 3 FAILED"
    print("Test Case 3 PASSED\n")

    print("All tests completed.")

if __name__ == "__main__":
    run_softmax_derivative_test()

