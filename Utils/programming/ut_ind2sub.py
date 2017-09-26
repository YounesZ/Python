""" This function is a direct translation of MATLAB's ind2sub.m
    Returns the subscript indices of an array of indices
    *** Assumes a 2D matrix size """

import numpy as np
from array import array

def main(siz, ind):
    # make sure indices are within range
    if np.max(ind) > np.prod(siz):
        raise ValueError('One of the indices is out of matrix range')
    if np.min(ind) < 0:
        raise ValueError('One of the indices is negative')

    # Convert indices
    nEl     =   len(ind)
    row     =   [0] * nEl
    col     =   [0] * nEl
    for ii in range(nEl):
        # Fill containers
        row[ii]     =   ind[ii]%siz[0]
        col[ii]     =   int(np.floor(ind[ii]/siz[0]))

    return row, col



# Driver
if __name__ == "__main__":
    main()