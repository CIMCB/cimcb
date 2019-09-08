import numpy as np


def wmean(x, weights):
    """Returns Weighted Mean. Ignores NaNs and handles infinite weights.

    Parameters
    ----------
    x: array-like [n_samples]
        An array-like object that contains the data.

    weights: array-like [n_samples]
        An array-like object that contains the corresponding weights.

    Returns
    ----------------------------------
    m: number
        The weighted mean.
    """

    # Flatten x and weights
    x = x.flatten()
    weights = weights.flatten()

    # Find NaNs
    nans = np.isnan(x)
    infs = np.isinf(weights)

    # If all x are nans, return np.nan
    if nans.all() == True:
        m = np.nan
        return m

    # If there are infinite weights, use the corresponding x
    if infs.any() == True:
        m = np.nanmean(x[infs])
        return m

    # Set NaNs to zero
    x[nans] = 0
    weights[nans] = 0
    
    # Normalize the weights + calculate Weighted Mean
    weights = weights / np.sum(weights)
    m = np.matmul(weights, x)
    return m
