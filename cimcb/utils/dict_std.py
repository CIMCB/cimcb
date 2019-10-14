import numpy as np


def dict_std(dict_list):
    std_dict = {}
    for key in dict_list[0].keys():
        value = []
        for i in dict_list:
            value.append(i[key])
        std_dict[key] = np.std(value, ddof=1)
    return std_dict
