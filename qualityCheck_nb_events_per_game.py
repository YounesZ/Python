from ReinforcementLearning.NHL.playbyplay.state_space_data import *
from ReinforcementLearning.NHL.playbyplay.game import Game
from Utils.base import get_logger, get_git_root
from ReinforcementLearning.NHL.config import Config

# =======================
# ==== FIRST SET POINTERS

# Pointers to the data
# repoCode    =   '/Users/younes_zerouali/Documents/Stradigi/Code/NHL_stats_SL'
# repoCode    =   '/Users/luisd/dev/NHL_stats'
repoCode    =   get_git_root()
db_root     =   Config().data_dir        #This is the location of the Hockey database
# db_root     =   '/Users/younes_zerouali/Documents/Stradigi/Databases/Hockey'
# db_root     =   '/Users/luisd/dev/NHL_stats/data'
repoPbP     =   path.join(db_root, 'PlayByPlay')
repoPSt     =   path.join(db_root, 'PlayerStats/player')
repoModel   =   path.join(repoCode, 'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_backup_trainedonallseasons_rankstatprediction')
repoModel   =   path.join(repoCode, 'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_perceptron_1layer_10units_relu')
repoSave    =   None #path.join(repoCode, 'ReinforcementLearning/NHL/playbyplay/data')

a_logger = get_logger(name="a_logger", debug_log_file_name="/tmp/a_logger.log")
print("Debug will be written in {}".format(a_logger.handlers[1].baseFilename))

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
    iSea    =   Season(a_logger, db_root, repo_model=repoModel, year_begin=int(iS.replace('Season_','')[:4]) )

    # Loop on games
    count   =   0
    num_games = len(iSea.games_info)
    for iG in iSea.games_info['gcode']:

        # pull data
        iGame = Game(iSea, gameId=iG)
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
            count, num_games, '=' * int(count / num_games * 40), 100 * count / num_games))
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


