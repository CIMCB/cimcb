import numpy as np


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        value = []
        for i in dict_list:
            value.append(i[key])
        mean_dict[key] = np.mean(value)
    return mean_dict
