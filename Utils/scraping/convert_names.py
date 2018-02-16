# Convert problematic player names from the trophy nominee CSV files

def convert_names(df, colId):

    # Conversion dictionary
    convD   =   {   'ZIGMUND PALFFY'            :   'ZIGGY PALFFY',\
                    'MIKE CAMMALLERI'           :   'MICHAEL CAMMALLERI',\
                    'ALEX STEEN'                :   'ALEXANDER STEEN',
                    'ALEX BURROWS'              :   'ALXANDRE BURROWS',
                    'JEAN-PIERRE DUMONT'        :   'J-P DUMONT',
                    'P.A. PARENTEAU'            :   'PA PARENTEAU',
                    'PIERRE-ALEXANDRE PARENTEAU':   'PA PARENTEAU'}

    # --- Replace entries in DataFrame
    # Convert names
    df.replace({colId:convD}, inplace=True)

    return df