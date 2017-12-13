# This function finds the position of the minimum number of elements in a vector that account for at least an arbitrary
# percentage of its sum

import numpy as np
from Utils.programming import ut_closest


def ut_cumsum_thresh(vec, thresh):
    # Make sure it is sorted
    so_vec  =   np.sort(vec)[::-1]
    # Make cumulative sum of the vector
    cs_vec  =   np.cumsum(vec) / np.sum(vec)
    # Find closest value to threhsold
    idV     =   ut_closest.main([thresh], cs_vec)[0]
    return idV


"""
if __name__=='__main__':
    testVec = [ 8.65149475e+00, 6.27908843e+00, 5.70647630e+00, 3.175840011e+00, 2.25142967e+00, 1.60536612e+00,\
                1.25880165e+00, 1.18261177e+00, 1.08151639e+00, 9.20756125e-01, 8.28966861e-01, 7.33857899e-01,\
                6.72265238e-01, 5.98971364e-01, 5.39847595e-01, 4.93306681e-01, 4.71384643e-01, 4.01087513e-01,\
                3.22719939e-01, 3.02918214e-01, 2.53234728e-01, 2.52277603e-01, 2.09808741e-01, 1.82961631e-01,\
                1.54777180e-01, 1.36185351e-01, 1.19093019e-01, 6.71630810e-02, 6.20105636e-02, 4.84460646e-02,\
                4.24317677e-02, 1.23354840e-02, 9.66976158e-03, 8.71251346e-03, 4.13582952e-03, 1.33477294e-03,\
                6.01023730e-31, 2.23580165e-31, 9.23567935e-32]
    ut_cumsum_thresh(testVec, 0.95)
"""