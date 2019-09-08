import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from .wmean import wmean


def knnimpute(x, k=3):
    """kNN missing value imputation using Euclidean distance.

    Parameters
    ----------
    x: array-like
        An array-like object that contains the data with NaNs.

    k: positive integer excluding 0, (default 3)
        The number of nearest neighbours to use.

    Returns
    -------
    z: array-like
        An array-like object corresponding to x with NaNs imputed.
    """

    # Tranpose x so we treat columns as features, and rows as samples
    x = x.T

    # Error check for k value
    if type(k) is not int:
        raise ValueError("k is not an integer")
    if k < 1:
        raise ValueError("k must be greater than zero")
    k_max = x.shape[1] - 1
    if k_max < k:
        raise ValueError("k value is too high. Max k value is {}".format(k_max))

    # z is the returned array with NaNs imputed
    z = x.copy()

    # Use columns without NaNs for knnimpute
    nan_check = np.isnan(x)
    no_nan = np.where(sum(nan_check.T) == 0, 1, 0)

    # Error check that not all columns have NaNs
    x_no_nan = x[no_nan == 1]
    if x_no_nan.size == 0:
        raise ValueError("All colummns of the input data contain missing values. Unable to impute missing values.")

    # Calculate pairwise distances between columns, and covert to square-form distance matrix
    pair_dist = pdist(x_no_nan.T, metric="euclidean")
    sq_dist = squareform(pair_dist)

    # Make diagonals negative and sort
    dist = np.sort(sq_dist - np.eye(sq_dist.shape[0], sq_dist.shape[1])).T
    dist_idx = np.argsort(sq_dist - np.eye(sq_dist.shape[0], sq_dist.shape[1])).T

    # Find where neighbours are equal distance
    equal_dist_a = np.diff(dist[1:].T, 1, 1).T == 0
    equal_dist_a = equal_dist_a.astype(int)  # Convert to integer
    equal_dist_b = np.zeros(len(dist))
    equal_dist = np.concatenate((equal_dist_a, [equal_dist_b]))  # Concatenate

    # Get rows and cols for missing values
    nan_idx = np.argwhere(nan_check)
    nan_rows = nan_idx[:, 0]
    nan_cols = nan_idx[:, 1]
    # Make sure rows/cols are in a list (note: this happens when there is 1 missing value)
    if type(nan_rows) is not np.ndarray:
        nan_rows = [nan_rows]
        nan_cols = [nan_cols]

    # Impute each NaN value
    for i in range(len(nan_rows)):

        # Error check for rows with all NaNs
        if np.isnan(x[nan_rows[i], :]).all() == True:
            warnings.warn("Row {} contains all NaNs, so Row {} is imputed with zeros.".format(nan_rows[i], nan_rows[i]), Warning)

        # Create a loop from 1 to len(dist_idx) - k
        lastk = len(dist_idx) - k
        loopk = [1]
        while lastk > loopk[-1]:
            loopk.append(loopk[-1] + 1)

        # Impute
        for j in loopk:
            L_a = equal_dist[j + k - 2 :, nan_cols[i]]
            L = np.where(L_a == 0)[0][0]  # equal_dist neighbours

            x_vals_r = nan_rows[i]
            x_vals_c = dist_idx[j : j + k + L, nan_cols[i]]
            x_vals = x[x_vals_r, x_vals_c]
            weights = 1 / dist[1 : k + L + 1, nan_cols[i]]
            imp_val = wmean(x_vals, weights)  # imputed value
            if imp_val is not np.nan:
                z[nan_rows[i], nan_cols[i]] = imp_val
                break

    # Transpose z
    z = z.T
    return z
