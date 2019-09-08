import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def binary_metrics(y_true, y_pred, cut_off=0.5, parametric=True, k=None):
    """ Return a dict of binary stats with the following metrics: R2, auc, accuracy, precision, sensitivity, specificity, and F1 score.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Binary label for samples (0s and 1s).

    y_pred : array-like, shape = [n_samples]
        Predicted y score for samples.

    cut_off : number, (default 0.5)
        A value for y_pred greater-than or equal to the cut_off will be treated as 1, otherwise it will be treated as 0 for the confusion matrix.

    parametric : boolean, (default True)
        If parametric is True, calculate R2.

    Returns
    -------
    stats: dict
        dict containing calculated R2, auc, accuracy, precision, sensitivity, specificity, and F1 score.
    """

    # Convert to array
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Error checks
    if y_true_arr.ndim != 1:
        raise ValueError("y_true should only have 1 dimension.")
    if y_pred_arr.ndim != 1:
        raise ValueError("y_pred should only have 1 dimension.")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("The number of values in y_true should match y_pred.")
    if np.array_equal(sorted(set(y_true_arr)), [0, 1]) is False:
        raise ValueError("y_true should only contain 0s and 1s")

    # Get confusion matrix
    try:
        y_pred_round = np.where(y_pred_arr >= cut_off, 1, 0)
    except RuntimeWarning:
        raise ValueError("Kevin: This warning says there are nans. Something is not right if y predicted are nans.")
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_round).ravel()

    # Binary statistics dictionary
    stats = {}
    if parametric is True:
        stats["R²"] = 1 - (sum((y_true_arr - y_pred_arr) ** 2) / sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    else:
        stats["R²"] = np.nan

    try:
        stats["AUC"] = roc_auc_score(y_true_arr, y_pred_arr)
    except ValueError:
        raise ValueError("You need to lower the learning_rate! This is a common issue when using the ‘mean_squared_error’ loss function called exploding gradients. 'At an extreme, the values of weights can become so large as to overflow and result in NaN values' (REF: https://machinelearningmastery.com/exploding-gradients-in-neural-networks/).")

    stats["ACCURACY"] = safe_div((tp + tn), (tp + tn + fp + fn))
    stats["PRECISION"] = safe_div((tp), (tp + fp))
    stats["SENSITIVITY"] = safe_div((tp), (tp + fn))
    stats["SPECIFICITY"] = safe_div((tn), (tn + fp))
    stats["F1-SCORE"] = safe_div((2 * tp), (2 * tp + fp + fn))

    # Additional: AIC/BIC/SSE
    n = len(y_true)
    resid = y_true - y_pred
    rss = sum(resid ** 2)
    if rss == 0:
        stats["SSE"] = 0
        stats["AIC"] = 0
        stats["BIC"] = 0
    else:
        stats["SSE"] = rss / n
        if k is None:
            stats["AIC"] = 0
            stats["BIC"] = 0
        else:
            stats["AIC"] = 2 * k - 2 * np.log(rss)
            stats["BIC"] = n * np.log(rss / n) + k * np.log(n)
    return stats


def safe_div(a, b):
    """Return np.nan if the demoninator is 0."""
    if b == 0:
        return np.nan
    return a / b
