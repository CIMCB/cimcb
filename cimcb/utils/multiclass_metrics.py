import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score


def multiclass_metrics(y_true, y_pred, cut_off=0.5, parametric=True, k=None):
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

    # Error checks -> later

    # Get confusion matrix
    try:
        y_pred_round = np.zeros(y_pred_arr.shape)
        idx = y_pred_arr.argmax(axis=-1)
        for i in range(len(idx)):
            y_pred_round[i, idx[i]] = 1

    except RuntimeWarning:
        raise ValueError("Kevin: This warning says there are nans. Something is not right if y predicted are nans.")
    conf = multilabel_confusion_matrix(y_true_arr, y_pred_round).ravel()
    n_groups = len(conf) / 4
    tn = conf[0::4]
    fp = conf[1::4]
    fn = conf[2::4]
    tp = conf[3::4]

    # Multi-Class Stats Dictionary (Macro Average)
    stats = {}

    # R^2 (macro R^2)
    ones = np.ones(int(n_groups))
    RSS = sum((y_true_arr - y_pred_arr) ** 2)
    TSS = sum((y_true_arr - np.mean(y_true_arr, axis=0)) ** 2)
    R2 = ones - (RSS / TSS)
    R2macro = sum(R2) / n_groups
    stats["R²"] = R2macro

    try:
        stats["AUC"] = roc_auc_score(y_true_arr, y_pred_arr, average='macro')
    except ValueError:
        raise ValueError("You need to lower the learning_rate! This is a common issue when using the ‘mean_squared_error’ loss function called exploding gradients. 'At an extreme, the values of weights can become so large as to overflow and result in NaN values' (REF: https://machinelearningmastery.com/exploding-gradients-in-neural-networks/).")

    stats["ACCURACY"] = safe_div(np.sum(safe_div((tp + tn), (tp + tn + fp + fn))), n_groups)
    stats["PRECISION"] = safe_div(np.sum(safe_div((tp), (tp + fp))), n_groups)
    stats["SENSITIVITY"] = safe_div(np.sum(safe_div((tp), (tp + fn))), n_groups)
    stats["SPECIFICITY"] = safe_div(np.sum(safe_div((tn), (tn + fp))), n_groups)
    stats["F1-SCORE"] = safe_div(np.sum(safe_div((2 * tp), (2 * tp + fp + fn))), n_groups)

    stats["SSE"] = 0
    stats["AIC"] = 0
    stats["BIC"] = 0
    # Per Group
    # stats["ACCURACYgroup"] = safe_div((tp + tn), (tp + tn + fp + fn))
    # stats["PRECISIONgroup"] = safe_div((tp), (tp + fp))
    # stats["SENSITIVITYgroup"] = safe_div((tp), (tp + fn))
    # stats["SPECIFICITYgroup"] = safe_div((tn), (tn + fp))
    # stats["F1-SCOREgroup"] = safe_div((2 * tp), (2 * tp + fp + fn))
    return stats


def safe_div(a, b):
    """Return np.nan if the demoninator is 0."""
    try:
        if b == 0:
            return np.nan
    except ValueError:
        if 0 in b:
            return np.nan
    return a / b
