import pickle
import matplotlib.pyplot as plt
from ReinforcementLearning.NHL.playerstats.nhl_player_stats import *
from ReinforcementLearning.NHL.playbyplay.playbyplay_data import *
from ReinforcementLearning.NHL.playbyplay.state_space_data import *

# =======================
# ==== FIRST SET POINTERS

# Pointers to the data
# repoCode    =   '/Users/younes_zerouali/Documents/Stradigi/Code/NHL_stats_SL'
# repoCode    =   '/Users/luisd/dev/NHL_stats'
repoCode    =   '/home/younesz/Documents/Code/NHL_stats_SL'
db_root     =   '/home/younesz/Documents/Databases/Hockey'        #This is the location of the Hockey database
# db_root     =   '/Users/younes_zerouali/Documents/Stradigi/Databases/Hockey'
# db_root     =   '/Users/luisd/dev/NHL_stats/data'
repoPbP     =   path.join(db_root, 'PlayByPlay')
repoPSt     =   path.join(db_root, 'PlayerStats/player')
repoModel   =   path.join(repoCode, 'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_backup_trainedonallseasons_rankstatprediction')
repoModel   =   path.join(repoCode, 'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_perceptron_1layer_10units_relu')
repoSave    =   None #path.join(repoCode, 'ReinforcementLearning/NHL/playbyplay/data')


# =======================
# ==== RETRIEVE INFOS

seasons =   ut_find_folders(repoPbP)
# Prep containers
gmInfo  =   pd.DataFrame(columns=['allE', 'filtE'])

# Loop on seasons
print('Launching analysis\n')
for iS in seasons:

    print('\tAnalysing season %s (%i/%i): ' % (iS, seasons.index(iS) + 1, len(seasons)))
    # List all games
    iSea    =   Season(db_root, int(iS.replace('Season_','')[:4]) )
    games   =   iSea.games_id

    # Loop on games
    count   =   0
    for iG in games['gcode']:

        # pull data
        iGame   =   iSea.pick_game(iG)
        iGame.pull_line_shifts('both', minduration=20)
        allE =  len(iGame.lineShifts)

        # Filter for regulartime and equalstrength
        iGame.pick_regulartime()
        iGame.pick_equalstrength()
        filtE = len(iGame.lineShifts)

        # Store
        gmInfo = pd.concat((gmInfo, pd.DataFrame(np.reshape([allE, filtE], [1, -1]), columns=['allE', 'filtE'])), ignore_index=True)

        count += 1
        if count % 50 == 0:
            # the exact output you're looking for:
            stdout.write("Game %i/%i: [%-40s] %d%%, completed" % (
            count, len(games), '=' * int(count / len(games) * 40), 100 * count / len(games)))
            stdout.flush()

    print('done\n')
pickle.dump( gmInfo, open('/home/younesz/Desktop/test.p', 'wb') )


# =======================
# ==== VISUALIZE

Fig  =  plt.figure()
Ax1  =  Fig.add_subplot(121)
Ax1.hist( np.reshape(gmInfo['allE'], [-1,1]) )
Ax1.set_xlabel('Number of events (raw)')
Ax1.set_ylabel('Number of games')
Ax1.set_title('Distribution: total nb of events per game')

Ax2  =  Fig.add_subplot(122)
Ax2.hist( np.reshape(gmInfo['filtE'], [-1,1]) )
Ax2.set_xlabel('Number of events (filtered)')
Ax2.set_ylabel('Number of games')
Ax2.set_title('Distribution: nb of filtered events per game')


