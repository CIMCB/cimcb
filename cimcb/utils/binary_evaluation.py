import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn import metrics
import scipy


def binary_evaluation(y_true, y_pred):
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

    # Binary statistics dictionary
    stats = {}

    stats["R²"] = 1 - (sum((y_true_arr - y_pred_arr) ** 2) / sum((y_true_arr - np.mean(y_true_arr)) ** 2))

    fpr, tpr, thresholds = metrics.roc_curve(y_true_arr, y_pred_arr, pos_label=1)
    stats["AUC"] = metrics.auc(fpr, tpr)

    try:
        stats["ManW P-Value"] = scipy.stats.mannwhitneyu(y_pred_arr[y_true_arr == 0], y_pred_arr[y_true_arr == 1], alternative="two-sided")[1]
    except ValueError:
        stats["ManW P-Value"] = 1

    return stats
