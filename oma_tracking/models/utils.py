"""Utility functions for the models module.
"""


def convert_float_to_int(d):
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, float) and value.is_integer():
            new_dict[key] = int(value)
        else:
            new_dict[key] = value
    return new_dict
def convert_dict(d):
    new_dict = {}
    for key, value in d.items():
        new_dict[key] = convert_float_to_int(d[key])
    return new_dict