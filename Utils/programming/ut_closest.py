""" This function returns the indices of the closest values to a reference vector"""

import numpy as np


def main(values, ref_vec):

    #Loop on values
    nEl     =   len(values)
    idX     =   [0]*nEl
    Vl      =   [0]*nEl
    for ii in range(nEl):
        # Subtract
        subV    =   np.sqrt( np.sum( ( np.array(ref_vec) - np.tile(np.array(values[ii]), [len(ref_vec), 1]) ) ** 2, axis=1 ) )
        ix      =   np.argmin( subV )

        # Add to vector
        idX[ii] =   ix
        Vl[ii]  =   ref_vec[ix]
    return idX, Vl




# Driver
if __name__ == "__main__":
    main()
