import numpy as np

def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Computes the softmax of a vector with a temperature parameter.

    Args:
        logits: A 1D NumPy array of raw scores (logits).
        temperature: A float representing the temperature. Must be positive.
                     Higher T -> softer probabilities, Lower T -> sharper probabilities.

    Returns:
        A 1D NumPy array of probabilities.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    # Subtract max logit for numerical stability (prevents overflow)
    scaled_logits = logits / temperature
    max_logit = np.max(scaled_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(scaled_logits - max_logit)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return softmax_probs

def softmax_derivative_with_temperature(softmax_output: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Computes the Jacobian matrix of the softmax function with temperature.

    Args:
        softmax_output: A 1D NumPy array of probabilities (output of softmax_with_temperature).
        temperature: A float representing the temperature. Must be positive.

    Returns:
        A 2D NumPy array representing the Jacobian matrix (dS_i / dz_j).
        The shape will be (len(softmax_output), len(softmax_output)).
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    s = softmax_output.reshape(-1, 1)  # Ensure s is a column vector
    # The Jacobian matrix J can be computed as:
    # J = (1/T) * (diag(s) - s @ s.T)
    # where diag(s) is a diagonal matrix with s on the diagonal.
    # s @ s.T is the outer product of s with itself.

    jacobian_matrix = (1.0 / temperature) * (np.diagflat(s) - np.dot(s, s.T))
    return jacobian_matrix

def softmax_grad_for_backprop(softmax_output: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Computes the gradient dS_i/dz_j in a way that's often more directly usable
    in backpropagation when combined with the upstream gradient.
    This is essentially the Jacobian matrix.

    Args:
        softmax_output: A 1D NumPy array of probabilities (output of softmax_with_temperature).
        temperature: A float representing the temperature. Must be positive.

    Returns:
        A 2D NumPy array representing the Jacobian matrix dS_i/dz_j.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    n = len(softmax_output)
    jacobian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                jacobian[i, j] = (1.0 / temperature) * softmax_output[i] * (1 - softmax_output[i])
            else:
                jacobian[i, j] = -(1.0 / temperature) * softmax_output[i] * softmax_output[j]
    return jacobian


# --- Example Usage ---
if __name__ == "__main__":
    # Example logits
    logits_example = np.array([2.0, 1.0, 0.1])
    temperature_example = 1.5 # Try different temperatures: 0.5, 1.0, 2.0

    print(f"Logits: {logits_example}")
    print(f"Temperature: {temperature_example}\n")

    # 1. Compute softmax with temperature
    probabilities = softmax_with_temperature(logits_example, temperature_example)
    print(f"Softmax probabilities: {probabilities}")
    print(f"Sum of probabilities: {np.sum(probabilities):.4f}\n") # Should be close to 1.0

    # 2. Compute the Jacobian matrix using the optimized method
    jacobian1 = softmax_derivative_with_temperature(probabilities, temperature_example)
    print("Jacobian matrix (optimized method):")
    print(jacobian1)
    print("\n")

    # 3. Compute the Jacobian matrix using the direct formula (for verification)
    jacobian2 = softmax_grad_for_backprop(probabilities, temperature_example)
    print("Jacobian matrix (direct formula method):")
    print(jacobian2)
    print("\n")

    # Verification: Check if the two methods produce the same result
    if np.allclose(jacobian1, jacobian2):
        print("Both Jacobian computation methods produce similar results.\n")
    else:
        print("Jacobian computation methods differ!\n")


    # How this derivative is typically used in backpropagation:
    # Let dL_dsoftmax be the gradient of the loss with respect to the softmax output.
    # This gradient would come from the layer above (or from the loss function itself).
    # For example, if using cross-entropy loss with one-hot encoded labels y_true:
    # dL_dsoftmax = probabilities - y_true  (This is a simplification for CE + Softmax)
    # More generally, it's the upstream gradient.

    # Let's assume an arbitrary upstream gradient for demonstration
    # (In a real scenario, this comes from the loss function's derivative w.r.t. softmax outputs)
    num_classes = len(logits_example)
    # Example: if the loss increased most when the first probability increased.
    dL_dsoftmax_example = np.array([0.5, -0.3, -0.2])
    print(f"Example upstream gradient (dL/dSoftmax): {dL_dsoftmax_example}\n")

    # The gradient of the loss with respect to the logits (dL/dLogits) is:
    # dL_dLogits = dL_dSoftmax @ Jacobian
    # Or, more efficiently for softmax:
    # If L is Cross-Entropy loss and S is Softmax output, P is predicted probability, Y is true label:
    # dL/dz_j = P_j - Y_j (when T=1).
    # For a general loss, and with temperature:
    # dL/dz_j = sum_i (dL/dS_i * dS_i/dz_j)
    # This is equivalent to dL_dsoftmax @ jacobian_matrix

    dL_dlogits = np.dot(dL_dsoftmax_example, jacobian1) # or jacobian2
    # Note: The order of multiplication matters. If dL_dsoftmax_example is a row vector (1, N)
    # and Jacobian is (N, N), then dL_dsoftmax_example @ Jacobian gives (1, N).
    # If dL_dsoftmax_example was a column vector (N,1), it would be Jacobian.T @ dL_dsoftmax_example

    print(f"Gradient of Loss w.r.t. Logits (dL/dLogits): {dL_dlogits}")

    # --- Example with a different temperature ---
    temperature_low = 0.5
    print(f"\n--- Example with Lower Temperature: {temperature_low} ---")
    probabilities_low_t = softmax_with_temperature(logits_example, temperature_low)
    print(f"Softmax probabilities (T={temperature_low}): {probabilities_low_t}")
    jacobian_low_t = softmax_derivative_with_temperature(probabilities_low_t, temperature_low)
    print("Jacobian matrix (optimized method):")
    print(jacobian_low_t)
    dL_dlogits_low_t = np.dot(dL_dsoftmax_example, jacobian_low_t)
    print(f"Gradient of Loss w.r.t. Logits (dL/dLogits, T={temperature_low}): {dL_dlogits_low_t}")

    temperature_high = 5.0
    print(f"\n--- Example with Higher Temperature: {temperature_high} ---")
    probabilities_high_t = softmax_with_temperature(logits_example, temperature_high)
    print(f"Softmax probabilities (T={temperature_high}): {probabilities_high_t}")
    jacobian_high_t = softmax_derivative_with_temperature(probabilities_high_t, temperature_high)
    print("Jacobian matrix (optimized method):")
    print(jacobian_high_t)
    dL_dlogits_high_t = np.dot(dL_dsoftmax_example, jacobian_high_t)
    print(f"Gradient of Loss w.r.t. Logits (dL/dLogits, T={temperature_high}): {dL_dlogits_high_t}")

    # Special case: Cross-Entropy Loss with Softmax
    # If you are using categorical cross-entropy loss *directly after* the softmax layer,
    # the combined derivative dL/dz simplifies nicely to (S - Y) / T,
    # where S is the softmax output (probabilities) and Y is the one-hot encoded true label.
    # This avoids explicitly forming the Jacobian.

    y_true_example = np.array([1, 0, 0]) # Example one-hot true label
    print("\n--- Simplified Gradient for Cross-Entropy Loss (Example) ---")
    print(f"True labels (one-hot): {y_true_example}")

    # For T=1
    dL_dz_ce_T1 = probabilities - y_true_example # (This is for T=1)
    print(f"dL/dz for CE loss (T=1, if these were T=1 probs): {dL_dz_ce_T1}")

    # For arbitrary T
    # Using the probabilities calculated with temperature_example (1.5)
    dL_dz_ce_T_custom = (probabilities - y_true_example) / temperature_example
    print(f"dL/dz for CE loss (T={temperature_example}): {dL_dz_ce_T_custom}")

    # Let's verify this with the Jacobian method for T=temperature_example
    # For cross-entropy, dL/dS_i = -Y_i / S_i
    # However, this can be numerically unstable if S_i is very small.
    # It's usually better to use the combined derivative.
    # The dL_dsoftmax_example = np.array([0.5, -0.3, -0.2]) was arbitrary.
    # If we use the actual derivative for CE loss:
    # dL/dS_i for CE is -y_i/S_i.
    # The upstream gradient for CE loss is actually simpler when combined with softmax.
    # The dL/dSoftmax for CE loss is -y_true / probabilities (element-wise).
    # This is often not computed explicitly. Instead, the combined dL/dLogits is used.

    # The formula dL/dz = (S - Y) / T is the most direct way if CE loss is used.
    # The Jacobian is useful if you have a different loss function after softmax,
    # or if you need to analyze the sensitivities more directly.

