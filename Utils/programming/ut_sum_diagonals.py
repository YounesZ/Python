""" This function sums the values of a square matrix along the diagonals """

import numpy as np
from Utils.programming import ut_ind2sub

def main(matrix):

    # Matrix dimensions
    shp     =   np.shape(matrix)
    nDiag   =   2 * shp[0] - 1
    sumD    =   np.zeros([1, nDiag])

    # Loop and sum along diagonals
    startI  =   list(range(shp[0]-1,-1,-1)) + list(range(shp[0],np.prod(shp)-1,shp[1]))
    nInd    =   list(range(shp[0])) + list(range(shp[0]-2,-1,-1))
    lElem   =   np.multiply( list(range(shp[0])), shp[0]+1)
    for ii  in range(nDiag):
        # list all indices
        allI    =   startI[ii] + lElem[:nInd[ii]+1]
        # Convert to subscripts
        r,c     =   ut_ind2sub.main(shp, allI)
        sumD[0,ii] = np.sum( [matrix[r[xxx]][c[xxx]] for xxx in list(range(len(r)))] )

    return sumD


# Driver
if __name__ == "__main__":
    main()