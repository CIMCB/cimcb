import numpy as np
from sklearn import preprocessing


def color_scale(x, method="tanh", beta=None):

    # Initially scale between 0 and 1
    scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 1))
    x_init = scaler.fit_transform(x[:, np.newaxis]).flatten()

    # Methods of transformation
    if method == "linear":
        x_tr = x_init
    elif method == "sq":
        x_tr = x_init ** 2
    elif method == "sqrt":
        x_tr = np.sqrt(x_init)
    elif method == "tan":
        x_tr = 1 + np.tan(beta * (1 + x_init))
    elif method == "tanh+1":
        x_tr = 1 + np.tanh(beta * (-1 + x_init))
    elif method == "tanh":
        x_tr_init = np.tanh(beta * (-1 + x_init))
        x_tr = scaler.fit_transform(x_tr_init[:, np.newaxis]).flatten()
    else:
        print("An incorrect method for color_scale was selected, so it set to 'tanh'. Supported methods are 'linear', 'sq', 'sqrt', 'tanh', and 'tanh+1'.")
        x_tr = x_init

    return x_tr
