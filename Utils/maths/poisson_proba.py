""" This function computes the Poisson probability for different values given some expected value lambda"""

import numpy as np
import math

def main(mylambda, values):

    proba = np.zeros([1, len(values)])
    for x in range(len(values)):
        lambda_n = mylambda ** values[x]
        n_fact = math.factorial(values[x])
        proba[0, x] = lambda_n / n_fact * math.exp(-mylambda)
    return proba


# Driver
if __name__ == "__main__":
    main()