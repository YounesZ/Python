""" This function returns the indices of the closest values to a reference vector"""

import numpy as np


def main(values, ref_vec):

    #Loop on values
    nEl     =   len(values)
    idX     =   [0]*nEl
    for ii in range(nEl):
        # Subtract
        subV    =   abs( np.array(ref_vec) - values[ii] )
        ix      =   np.argmin( subV )

        # Add to vector
        idX[ii]    =   ix

    return idX




# Driver
if __name__ == "__main__":
    main()
