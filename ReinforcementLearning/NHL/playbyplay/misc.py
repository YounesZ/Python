import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from Utils.programming.ut_find_folders import *



# Visualize goal difference: show home advantage
# ==============================================
# list seasons
repoPbP =   '/home/younesz/Documents/Databases/Hockey/PlayByPlay'
allS    =   ut_find_folders(repoPbP, True)
allG    =   pd.DataFrame()

for iS in allS:
    # Load data
    dt  =   pickle.load( open(path.join(repoPbP, iS, 'converted_data.p'), 'rb') )['playbyplay']
    # Keep only final lines for each game
    dt  =   dt.drop_duplicates(subset=['gcode', 'season'], keep='last')[['away.score', 'home.score']]
    # Concatenate
    allG=   pd.concat( (allG, dt), axis=0 )
# Viz
ann     =   "Win percentage for home team: %.1f %%" %(np.sum(allG['home.score']>allG['away.score'])/len(allG)*100)
plt.figure(); plt.hist(allG['home.score']-allG['away.score'])
plt.xlabel('Home team goal diff')
plt.ylabel('Nb of games')
plt.annotate(ann, xy=(-9,5000), xytext=(-9,5000))


# Visualize shot difference: show home advantage
# ==============================================
# list seasons
repoPbP =   '/home/younesz/Documents/Databases/Hockey/PlayByPlay'
allS    =   ut_find_folders(repoPbP, True)
allG    =   pd.DataFrame()

for iS in allS:
    # Load data
    dt  =   pickle.load( open(path.join(repoPbP, iS, 'converted_data.p'), 'rb') )['playbyplay']
    # Keep only shot lines for each game
    dt  =   dt[dt['etype']=='SHOT']
    # Check if home or away shots
    dt['homeshot']  =   dt['ev.team']==dt['hometeam']
    dt['awayshot']  =   dt['ev.team']==dt['awayteam']
    # Group events by game and season
    dt  =   dt.groupby(['season', 'gcode'])[['homeshot', 'awayshot']].sum(axis=0)
# Viz
ann     =   "Overshoot percentage: %.1f %%" %(np.sum(dt['homeshot']>dt['awayshot'])/len(dt)*100)
plt.figure(); plt.hist(dt['homeshot']-dt['awayshot'])
plt.xlabel('Home team overshoot')
plt.ylabel('Nb of games')
plt.annotate(ann, xy=(-35,300), xytext=(-35,300))

