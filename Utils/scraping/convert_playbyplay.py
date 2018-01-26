import pandas as pd
import numpy as np
import pickle
import csv
from Utils.programming.ut_find_folders import *
from Utils.scraping.convert_names import *


def to_pandas(repoPbP):
    # --- List all seasons
    allS = ut_find_folders(repoPbP, True)
    # Loop on seasons
    for isea in allS:
        # --- ART Ross nominees
        isea2   =   isea.replace('Season_', '')
        # Load csv file
        csvF    =   path.join( repoPbP, isea, 'playbyplay_'+isea2+'.csv' )
        df_pbp  =   pd.read_csv(csvF, engine='python')

        # --- SELKE nominees
        # Load csv file
        csvF    =   path.join(repoPbP, isea, 'roster_'+isea2+'.csv' )
        df_rst  =   pd.read_csv(csvF, engine='python')

        # --- PICKLE IT OUT
        svname  =   path.join( repoPbP, isea, 'converted_data.p')
        with open(svname, 'wb') as f:
            pickle.dump({'playbyplay':df_pbp, 'roster':df_rst}, f)



# LAUNCHER
# ========
# Paths
repoPbP =   '/home/younesz/Documents/Databases/Hockey/PlayByPlay'
to_pandas(repoPbP)

"""
"""