def ut_difference( vec1, vec2):

    # List of indices
    for ii in vec2:
        if ii in vec1:
            vec1.pop( vec1.index(ii) )
    return vec1