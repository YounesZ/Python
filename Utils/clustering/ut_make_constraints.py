# This function prepares the constraints matrix for the COP-KMEANS classifier
# inputs:
# -------
#   -   vec1    :   vector of indices of the samples that should belong to the first class
#   -   vec2    :   vector of indices of the samples that should belong to the second class
#   -   vec3    :   vector of indices of the samples that should belong to the third class
#

def ut_make_constraints(vec1, vec2, vec3):

    # --- MUST LINKS
    # class1
    constraints     =   []
    for ix in range(len(vec1)-1):
        for iy in range(ix+1, len(vec1)):
            constraints.append([vec1[ix], vec1[iy], 1])
    # class2
    for ix in range(len(vec2) - 1):
        for iy in range(ix+1, len(vec2)):
            constraints.append([vec2[ix], vec2[iy], 1])
    # class3
    for ix in range(len(vec3) - 1):
        for iy in range(ix + 1, len(vec3)):
            constraints.append([vec3[ix], vec3[iy], 1])

    # --- DON'T LINKS
    # classes 1 and 2
    for ix in range(len(vec1)):
        for iy in range(len(vec2)):
            constraints.append([vec1[ix], vec2[iy], -1])
    # classes 1 and 3
    for ix in range(len(vec1)):
        for iy in range(len(vec3)):
            constraints.append([vec1[ix], vec3[iy], -1])
    # classes 2 and 3
    for ix in range(len(vec2)):
        for iy in range(len(vec3)):
            constraints.append([vec2[ix], vec3[iy], -1])

    return constraints
