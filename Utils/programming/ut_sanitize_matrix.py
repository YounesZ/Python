import pandas as pd

def ut_sanitize_matrix(matrix, reference=None):
    # Remove NaNs
    if type(matrix)==pd.core.frame.DataFrame:
        if reference is None:
            matrix  =   matrix[~matrix.isnull().any(axis=1).values]
        else:
            matrix  =   matrix[~reference.isnull().any(axis=1).values]
    else:
        print('Unprocessable data type - need to add case')
    return matrix
