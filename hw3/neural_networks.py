import numpy as np

from activations import (
    sigmoid, softmax, relu, identity)

def predict_0_hidden_layer(x_NF, w_arr, b_arr, output_activation):
    """ Make predictions for a neural net with no hidden layers

    This function demonstrates 3 special cases for an MLP with 0 hidden layers
    1. identity activation : equivalent to linear regression
    2. sigmoid activation : equivalent to binary logistic regression
    3. softmax activation : equivalent to multi-class logistic regression

    Args
    ----
    x_NF : 2D numpy array, shape (N, F) = (n_samples, n_features)
        Input features

    w_arr : 1D or 2D numpy array, shape (n_features, n_outputs)
        For single output, this may be a 1D array of shape (n_features,)

    b_arr : 1D numpy array, shape (n_outputs,)
        For single output, this may be a scalar float

    output_activation : callable
        Activation function for the output layer.
        Given an input array, must return output array of same shape

    Returns
    -------
    yhat_NC : 1D or 2D numpy array:
        shape (N,C) = (n_samples, n_outputs) if n_outputs > 1, else shape (N,C) = (n_samples,)
        Predicted values using the specified neural network configuration
        
        Suppose we had N=3 examples, F=1 features, and n_outputs = 1
        * if output_activation == identity, return array of real values
            e.g., input: [x1, x2, x3] --> output:[2.5, -6.7, 12]
        * if output_activation == sigmoid, return an array of probabilities
            e.g., input: [x1, x2, x3] --> output:[0.3, 0.8, 1.0]

        Suppose we had N=2 examples, F=1 features, and n_outputs = 3
        * if output_activation == softmax, return an array of proba vectors.
            e.g., input: [x1, x2] --> output:[[0.2, 0.4, 0.4], [0.8, 0.2, 0.]]

    Examples
    --------
    See doctest_neural_nets.py

    """
    return None  # TODO: fixme


def predict_n_hidden_layer(
        x_NF, w_list, b_list,
        hidden_activation=relu, output_activation=softmax):
    """ Make predictions for an MLP with zero or more hidden layers

    Parameters:
    -----------
    x_NF : numpy array of shape (n_samples, n_features)
        Input data for prediction.

    w_list : list of numpy array, length is n_layers
        Each entry represents 2D weight array for corresponding layer
        Shape of each entry is (n_inputs, n_outputs)
        Layers are ordered from input to output in predictive order

    b_list : list of numpy array, length is n_layers
        Each entry represents the intercept aka bias array for a specific layer
        Shape of each entry is (n_outputs,)
        Layers are ordered from input to output in predictive order

    hidden_activation : callable, optional (default=relu)
        Activation function for all hidden layers.

    output_activation : callable, optional (default=softmax)
        Activation function for the output layer.

    Returns:
    --------
    yhat_NC : 1D or 2D numpy array:
        shape (N,C) = (n_samples, n_outputs) if n_outputs > 1, else shape (N,C) = (n_samples,)
        Predicted values (for regression) or probabilities (if classification)
        Each row corresponds to corresponding row of x_NF input array.

        Suppose we had N=2 examples, F=1 features, and n_outputs = 1
        * if output_activation == sigmoid, return an array of proba vectors of label 1.
            e.g., input: [x1, x2] --> output:[[0.2], [0.8]]

        Suppose we had N=2 examples, F=1 features, and n_outputs = 3
        * if output_activation == softmax, return an array of proba vectors.
            e.g., input: [x1, x2] --> output:[[0.2, 0.4, 0.4], [0.8, 0.2, 0.]]

    Examples
    _______
    See doctest_neural_nets.py
    """
    n_layers = len(w_list)
    assert n_layers == len(b_list)

    # Forward propagation: start from the input layer
    out_arr = x_NF

    for layer_id in range(n_layers):
        # Get w and b arrays for current layer
        w_arr = w_list[layer_id]
        b_arr = b_list[layer_id]

        # Perform the linear operation: X · w + b
        out_arr = None  # TODO: fixme

        # Perform the non-linear activation of current layer
        out_arr = None  # TODO: fixme

    out_arr = np.squeeze(out_arr)  # reduce unnecessary dimension for single output

    return out_arr
