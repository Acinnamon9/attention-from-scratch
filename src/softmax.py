# Numerically-stable softmax for row-wise application.
import numpy as np

def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute the softmax of x along `axis` in a numerically stable way."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp