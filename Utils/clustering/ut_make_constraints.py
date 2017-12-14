# This function prepares the constraints matrix for the COP-KMEANS classifier
# inputs:
# -------
#   -   vec1    :   vector of indices of the samples that should belong to the first class
#   -   vec2    :   vector of indices of the samples that should belong to the second class
#

def ut_make_constraints(vec1, vec2):
    # Make must-links: class1
    constraints     =   []
    for ix in range(len(vec1)-1):
        for iy in range(ix+1, len(vec1)):
            constraints.append([vec1[ix], vec1[iy], 1])
    # Make must-links: class2
    for ix in range(len(vec2) - 1):
        for iy in range(ix+1, len(vec2)):
            constraints.append([vec2[ix], vec2[iy], 1])
    # Make dont-links: cross-class
    for ix in range(len(vec1)):
        for iy in range(len(vec2)):
            constraints.append([vec1[ix], vec2[iy], -1])
    return constraints
