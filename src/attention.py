"""Scaled dot-product attention implemented in NumPy.


This file intentionally avoids frameworks and autograd. The functions
include shape checks and optional verbose prints to help you inspect the
intermediate matrices during learning experiments.
"""


from typing import Optional
import numpy as np
from .softmax import stable_softmax




def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: Optional[np.ndarray] = None, verbose: bool = False) -> np.ndarray:
    # Compute scaled dot-product attention.


    # Args:
    # Q: queries, shape (n_q, d_k)
    # K: keys, shape (n_k, d_k)
    # V: values, shape (n_k, d_v)
    # mask: optional mask broadcastable to (n_q, n_k); masked positions should be True where we want to mask
    # verbose: if True, prints intermediate matrices


    # Returns:
    # output: shape (n_q, d_v)

    # Basic shape checks
    assert Q.ndim == 2 and K.ndim == 2 and V.ndim == 2, "Q, K, V must be 2-D"
    n_q, d_k_q = Q.shape
    n_k, d_k = K.shape
    n_k_v, d_v = V.shape
    assert d_k_q == d_k, f"Q and K embed dims must match (got {d_k_q} vs {d_k})"
    assert n_k == n_k_v, f"K and V must have same first dimension (got {n_k} vs {n_k_v})"


    # 1) similarity
    scores = Q.dot(K.T) / np.sqrt(d_k)


    if verbose:
        print("scores (raw):\n", scores)


    # 2) apply mask if provided (mask True indicates masked positions)
    if mask is not None:
        # mask should be broadcastable to scores shape
        scores = np.where(mask, -1e9, scores)
        if verbose:
            print("scores (masked):", scores)


    # 3) softmax to get attention weights
    weights = stable_softmax(scores, axis=-1)
    if verbose:
        print("weights (row-wise softmax):\n", weights)
        print("weights row sums:\n", weights.sum(axis=1))


    # 4) weighted sum of values
    output = weights.dot(V)


    if verbose:
        print("output (weights @ V):\n", output)


    return output




if __name__ == "__main__":
    # Quick demo to show matrix shapes and values
    np.set_printoptions(precision=3, suppress=True)
    Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    K = Q.copy()
    V = np.array([[10, 0, 0, 0], [0, 20, 0, 0], [0, 0, 30, 0]], dtype=float)


    out = scaled_dot_product_attention(Q, K, V, verbose=True)
    print('\nDemo finished')