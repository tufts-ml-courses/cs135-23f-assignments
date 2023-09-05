'''
hw0_split.py

Summary
-------
Complete the problem below to demonstrate your comprehension of NumPy.

You can do a basic check of the doctests via:
$ python -m doctest hw0_split.py

Examples
--------
>>> x_LF = np.asarray([
... [0, 11],
... [0, 22],
... [0, 33],
... [-2, 44],
... [-2, 55],
... [-2, 66],
... ])
>>> xcopy_LF = x_LF.copy() # preserve what input was before the call
>>> train_MF, test_NF = split_into_train_and_test(
...     x_LF, frac_test=2/6, random_state=np.random.RandomState(0))
>>> train_MF.shape
(4, 2)
>>> test_NF.shape
(2, 2)
>>> print(train_MF)
[[-2 66]
 [ 0 33]
 [ 0 22]
 [-2 44]]
>>> print(test_NF)
[[ 0 11]
 [-2 55]]

# Verify that input array did not change due to function call
>>> np.allclose(x_LF, xcopy_LF)
True

References
----------
For more about RandomState, see:
https://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
'''

import numpy as np

def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    ''' Divide provided array into train and test sets along first dimension

    User can provide random number generator object to ensure reproducibility.

    Args
    ----
    x_all_LF : 2D np.array, shape = (n_total_examples, n_features) (L, F)
        Each row is a feature vector
    frac_test : float, fraction between 0 and 1
        Indicates fraction of all L examples to allocate to the "test" set
        Returned test set will round UP if frac_test * L is not an integer.
        e.g. if L = 10 and frac_test = 0.31, then test set has N=4 examples
    random_state : np.random.RandomState instance or integer or None
        If int, will create RandomState instance with provided value as seed
        If None, defaults to current numpy random number generator np.random.

    Returns
    -------
    x_train_MF : 2D np.array, shape = (n_train_examples, n_features) (M, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    x_test_NF : 2D np.array, shape = (n_test_examples, n_features) (N, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    Post Condition
    --------------
    This function should be side-effect free. Provided input array x_all_LF
    should not change at all (not be shuffled, etc.)
    '''
    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(int(random_state))
    if not hasattr(random_state, 'rand'):
        raise ValueError("Not a valid random number generator")

    # Random shuffle of row ids corresponding to all L provided examples
    L, F = x_all_LF.shape
    shuffled_ids_L = random_state.permutation(np.arange(L))

    # Determine the number of test examples N
    N = int(np.ceil(L * float(frac_test)))
    # Keep remaining M examples as training
    M = L - N

    # TODO use the first M row ids in shuffled_ids_L to make x_train_MF
    # TODO use the remaining N row ids to make x_test_NF
    # HINT Use integer indexing

    # TODO return both x_train_MF and x_test_NF
    return None, None
