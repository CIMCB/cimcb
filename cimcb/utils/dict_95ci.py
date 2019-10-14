import numpy as np


def dict_95ci(dict_list):
    median_dict = {}
    for key in dict_list.keys():
        value = dict_list[key]
        value_arr = np.array(value)
        if np.isnan(value_arr).any() == True:
            median_dict[key] = np.nan
        else:
            append_low = np.percentile(value_arr, 2.5)
            append_mid = np.percentile(value_arr, 50)
            append_upp = np.percentile(value_arr, 95.7)
            median_dict[key] = [append_low, append_upp, append_mid]
    return median_dict
