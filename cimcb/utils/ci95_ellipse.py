import math
import numpy as np
from sklearn.decomposition import PCA


def ci95_ellipse(data, type="pop"):
    """ Construct a 95% confidence ellipse using PCA.

    Parameters
    ----------
    data : array-like, shape = [n_samples, 2]
        data[:,0] must represent x coordinates
        data[:,1] must represent y coordinates

    type : string, optional (default='pop')
        It must be 'pop' or 'mean'

    Returns
    -------
    ellipse: array-like, shape = [100, 2]
        ellipse[:,0] represents x coordinates of ellipse
        ellipse[:,1] represents y coordinates of ellipse

    outside: array-like, shape = [n_samples, 1]
        returns an 1d array (of 0/1) with length n_samples
        0 : ith sample is outside of ellipse
        1 : ith sample is inside of ellipse
    """

    # Build and fit PCA model
    pca = PCA()
    pca.fit(data)
    coeff = pca.components_
    score = pca.transform(data)
    eigvals = pca.explained_variance_

    # Calculate rotation angle
    phi = math.atan2(coeff[0, 1], coeff[0, 0])

    # This angle is between -pi and pi.
    # Let's shift it such that the angle is between 0 and 2pi
    if phi < 0:
        phi += 2 * math.pi

    # Get the coordinates of the data mean
    n = len(data)
    m = np.mean(data, axis=0)
    x0 = m[0]
    y0 = m[1]

    # Get the 95% confidence interval error ellipse
    # inverse of the chi-square cumulative distribution for  p = 0.05 & 2 d.f. = 5.9915
    chisquare_val = 5.9915
    if type is "pop":
        a = math.sqrt(chisquare_val * eigvals[0])
        b = math.sqrt(chisquare_val * eigvals[1])
    elif type is "mean":
        a = math.sqrt(chisquare_val * eigvals[0] / n)
        b = math.sqrt(chisquare_val * eigvals[1] / n)
    else:
        raise ValueError("type has to be 'pop' or 'mean'.")

    # the ellipse in x and y coordinates
    theta_grid = np.linspace(0, 2 * math.pi, num=100)
    ellipse_x_r = a * np.cos(theta_grid)
    ellipse_y_r = b * np.sin(theta_grid)

    # Define a rotation matrix
    R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    # let's rotate the ellipse to some angle phi
    r_ellipse = np.dot(np.vstack((ellipse_x_r, ellipse_y_r)).T, R)

    # Draw the error ellipse
    x = r_ellipse[:, 0] + x0
    y = r_ellipse[:, 1] + y0
    ellipse = np.stack((x, y), axis=1)

    outside = []
    for i in range(len(score)):
        metric = (score[i, 0] / a) ** 2 + (score[i, 1] / b) ** 2
        if metric > 1:
            outside.append(1)
        else:
            outside.append(0)

    return ellipse, outside
