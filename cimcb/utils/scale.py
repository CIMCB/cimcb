import numpy as np


def scale(x, axis=0, ddof=1, method="auto", mu="default", sigma="default", return_mu_sigma=False):
    """Scales x (which can include nans) with method: 'auto', 'pareto', 'vast', or 'level'.

    Parameters
    ----------
    x: array-like
        An array-like object that contains the data.

    axis: integer or None, (default 0)
        The axis along which to operate

    ddof: integer, (default 1)
        The degrees of freedom correction. Note, by default ddof=1 unlike scipy.stats.zscore with ddof=0.

    method: string, (default "auto")
        Method used to scale x. Accepted methods are 'auto', 'pareto', 'vast' and 'level'.

    mu: number or "default", (default "default")
        If mu is provided it is used, however, by default it is calculated.

    sigma: number or "default",  (default "default")
        If sigma is provided it is used, however, by default it is calculated.

    return_mu_sigma: boolean, (default False)
        If return_mu_sigma is True, mu and sigma are returned instead of z. Note, this is useful if mu and sigma want to be stored for future use.

    Returns if return_mu_sigma = False
    ----------------------------------
    z: array-like
        An array-like object that contains the scaled data.

    Returns if return_mu_sigma = True
    ---------------------------------
    mu: number
        Calculated mu for x given axis and ddof.

    sigma: number
        Calculated sigma for x given axis and ddof.
    """

    x = np.array(x)

    # Simplier if we tranpose X if axis=1 (return x.T after the calculations)
    if axis == 1:
        x = x.T

    # Expand dimension if array is 1d
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)

    # Calculate mu and sigma if set to 'default' (ignoring nans)
    if mu is "default":
        mu = np.nanmean(x, axis=0)
    if sigma is "default":
        sigma = np.nanstd(x, axis=0, ddof=ddof)
        sigma = np.where(sigma == 0, 1, sigma)  # if a value in sigma equals 0 it is converted to 1

    # Error check before scaling
    if len(mu) != len(x.T):
        raise ValueError("Length of mu array does not match x matrix.")
    if len(sigma) != len(x.T):
        raise ValueError("Length of sigma array does not match x matrix.")

    # Scale based on selected method
    if method is "auto":
        z = (x - mu) / sigma
    elif method is "pareto":
        z = (x - mu) / np.sqrt(sigma)
    elif method is "vast":
        z = ((x - mu) / sigma) * (mu / sigma)
    elif method is "level":
        z = (x - mu) / mu
    else:
        raise ValueError("Method has to be either 'auto', 'pareto', 'vast', or 'level'.")

    # Return x.T if axis = 1
    if axis == 1:
        z = z.T

    if return_mu_sigma is True:
        return mu, sigma
    else:
        return z
