import numpy as np


def dict_perc(dict_list, ci=95):
    perc_dict = {}
    for key in dict_list[0].keys():
        value = []
        for i in dict_list:
            value.append(i[key])
        value_arr = np.array(value)
        if np.isnan(value_arr).any() == True:
            perc_dict[key] = [np.nan, np.nan]
        else:
            lower_alpha = (100 - ci) / 2
            upper_alpha = 100 - lower_alpha
            lower_ci = np.percentile(value_arr, lower_alpha)
            upper_ci = np.percentile(value_arr, upper_alpha)
            perc_dict[key] = [lower_ci, upper_ci]
    return perc_dict
