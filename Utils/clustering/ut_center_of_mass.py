import numpy as np

def ut_center_of_mass(position, weight):
    return np.sum( position * np.tile(weight, [1,2]) / np.sum(np.abs(weight)), axis=0 )