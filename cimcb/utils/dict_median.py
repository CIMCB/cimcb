import numpy as np


def dict_median(dict_list):
    """Calculate median for all keys in dictionary."""
    median_dict = {}
    for key in dict_list[0].keys():
        value = []
        for i in dict_list:
            value.append(i[key])
        value_arr = np.array(value)
        if np.isnan(value_arr).any() == True:
            median_dict[key] = np.nan
        else:
            median_dict[key] = np.median(value)
    return median_dict
