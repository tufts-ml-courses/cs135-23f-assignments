import numpy as np
from scipy.special import expit as sigmoid

def relu(z_NB):
    return np.maximum(z_NB, 0.0)


def identity(z_NB):
    return z_NB


def softmax(z_NC):
    """ Compute the softmax activation for a 2D array of values.

    Args
    ----
    z_NC : 2D NumPy array, shape (n_samples, n_classes)
        Input scalar values, can be anywhere on real line (-inf, +inf)

    Returns
    -------
    p_NC : 2D NumPy array, shape (n_samples, n_classes)
        Each row represents probability vector over C possible classes
        For each data sample (row), the sum of probabilities across all classes will be 1.

    Examples:
    --------
    >>> z_13 = np.array([[2.0, 1.0, 0.1]])
    >>> softmax_z_13 = softmax(z_13)
    >>> print(np.round(softmax_z_13, 2))
    [[0.66 0.24 0.1 ]]
    >>> print(np.sum(softmax_z_13, axis=1))
    [1.]

    >>> z_23 = np.array([[10_000, 10_000, 5_000], [1, 1, 1]])
    >>> softmax_z_23 = softmax(z_23)
    >>> print(np.round(softmax_z_23, 2))
    [[0.5  0.5  0.  ]
     [0.33 0.33 0.33]]
    >>> print(np.sum(softmax_z_23, axis=1))
    [1. 1.]
    """

    # Subtract the maximum value for numerical stability
    numer_NC = np.exp(z_NC - np.max(z_NC, axis=1, keepdims=True))
    return numer_NC / numer_NC.sum(axis=1, keepdims=True)

