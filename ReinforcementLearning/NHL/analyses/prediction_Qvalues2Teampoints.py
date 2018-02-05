"""
This script implements an analysis pipeline for assessing the predictive power of TD(lambda)-learned Qvalues on the
number of points a team gets for a given season/set of games
"""

# ==== FIRST    :   IMPORT MODULES
import pickle
import numpy as np
from os import path, makedirs
from shutil import copyfile
from Utils.programming.ut_find_folders import *
from Utils.programming.ut_clone_directory import  *
from ReinforcementLearning.NHL.playbyplay.state_space_data import HockeySS
from ReinforcementLearning.NHL.playerstats.nhl_player_stats import do_ANN_training, do_clustering_multiyear
from ReinforcementLearning.NHL.playbyplay.playbyplay_data import Season, Game



def get_players_classes(repoModel, data_for_game, preprocessing, classifier, number_of_games):
    """
    Calculates (dataframe with) all player's classes.
    Updates the 'data for game' structure with it; also returns it.
    Usage:
        repoModel = ... # here goes the directory where your model is saved.
        # Montreal received Ottawa on march 13, 2013
        gameId = get_game_id(home_team_abbr='MTL', date_as_str='2013-03-13')
        season      =   '20122013'
        mtlott      =   Game(repoPbP, repoPSt, season, gameId=gameId )
        #
        players_classes = get_players_classes(repoModel, mtlott, number_of_games=30)
        # this is equivalent to ask 'mtlott' for the data; so:
        assert players_classes.equals(mtlott.player_classes)
    """
    # Pick players stats - last 'n' games
    data_for_game.pull_line_shifts(team='both', minduration=20)
    data_for_game.pick_regulartime()
    data_for_game.pick_equalstrength()
    data_for_game.pull_players_classes(preprocessing, classifier, nGames=number_of_games)
    return data_for_game



# ==== NEXT     :   SET POINTERS
root        =   '/Users/younes_zerouali/Documents/Stradigi'
root        =   '/home/younesz/Documents'
root_db     =   path.join(root, 'Databases', 'Hockey')
repoPSt     =   path.join(root_db, 'PlayerStats/player')
repoPbP     =   path.join(root_db, 'PlayByPlay')
repoCode    =   path.join(root, 'Code', 'NHL_stats_SL')
repoModel   =   path.join(repoCode, 'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_perceptron_1layer_10units_relu')

"""
# ==== NEXT     :   LOOP ON SEASONS, LEAVE-ONE-OUT
seasons     =   ut_find_folders( path.join(root_db, 'PlayByPlay'), True )
for iSea in seasons:

    # == step1: Train the model for players classification
    # Compute the model
    keep_seasons                =   list( set(seasons).difference(iSea) )
    normalizer, pca, dtCols, _  =   do_ANN_training( repoPSt, repoPbP, repoCode, repoModel, allS_p=keep_seasons)
    global_centers              =   do_clustering_multiyear(repoModel, repoPSt, repoPbP, dtCols, normalizer, pca, root)
    # Make sure the model is backed up for future use
    dst = path.join(repoModel, 'MODELS', 'MODEL_perceptron_1layer_10units_relu_LOO_'+iSea)
    ut_clone_directory(repoModel, dst)

    # == step2: Learn Q-table for the computed model
    # Compute Q-table
    repoSave    =   path.join(root_db, 'processed')
    HSS         =   HockeySS(root_db)
    HSS.list_all_games()
    HSS.pull_RL_data(repoModel, repoSave)
    HSS.teach_RL_agent(repoSave)
    # Backup Q-table
    lsFiles     =   listdir(repoSave)
    ispickle    =   [True if '.p' in x else False for x in lsFiles]
    lsFiles     =   list( compress(lsFiles, ispickle) )
    dirname     =   path.join(repoSave, 'stable', 'rankstats_LOO_'+iSea)
    if not path.exists(dirname):
        makedirs(dirname)
    [copyfile( path.join(repoSave, x), dirname ) for x in lsFiles]

    # == step3: predict nb of points (gamewise)
    # Setup var names
    Qvalues         =   HSS.RL_action_values
    preprocessing   =   pickle.load( open(path.join(dst, 'baseVariables.p'), 'rb') )
    classifier      =   ANN_classifier()
    classifier.ann_reload_model(dirname)
    # Retrieve data
    # List all games
    loo_sea         =   Season(root_db, int(iSea.replace('Season_', '')[:4]) )
    loo_sea.list_game_ids()

    # List all teams
    allTeams = loo_sea.games_id[['hometeam', 'awayteam']]
    allTeams = np.unique(np.concatenate(allTeams.values))

    # Initialize empty containers
    seaInfo = dict([(x, pd.DataFrame(columns=['gameCode', 'avgQ', 'points', 'goalsFor', 'goalsAg'])) for x in allTeams])
    count = 0
    for iG in loo_sea.games_id['gcode'].values:

        # Get game data
        gameData    =   loo_sea.pick_game(iG)
        gameData    =   get_players_classes(repoModel, gameData, preprocessing, classifier, 30)

        # Get game score
        homeTeam    =   HSS.games_lst.iloc[iG]['hometeam']
        goals = gameData.df[gameData.df['gcode'] == gameCode]
        goals = goals[goals['etype'] == 'GOAL']['ev.team']
        homeTeam = HSS.games_lst.iloc[iG]['hometeam']
        goals = gameData.df[gameData.df['gcode'] == gameCode]
        goals = goals[goals['etype'] == 'GOAL']['ev.team']
        points = int((goals == homeTeam).sum() > (goals != homeTeam).sum()) * 2
        if gameData.df[gameData.df['gcode'] == gameCode].iloc[-1]['period'] > 3:
            points = np.maximum(1, points)

        # Get state-space
        playersCode = gameData.encode_line_players()
        linesCode = np.array([[gameData.recode_line(linedict, a) for a in b] for b in playersCode])
        perCode = gameData.recode_period(gameData.lineShifts['period'])
        difCode = gameData.recode_differential(gameData.lineShifts['differential'])

        # Get info for home team
        qv_shifts = [Qvalues[w, x, y, z] for w, x, y, z in zip(perCode, difCode, linesCode[:, 1], linesCode[:, 0])]

        # Append this game to the team's data
        seaInfo[homeTeam] = pd.concat((seaInfo[homeTeam], pd.DataFrame(
            np.reshape([gameCode, np.mean(qv_shifts), points, (goals == homeTeam).sum(), (goals != homeTeam).sum()],
                       [1, 5]), columns=['gameCode', 'avgQ', 'points', 'goalsFor', 'goalsAg'])), ignore_index=True)

        count += 1
        if count % 100 == 0:
            stdout.write('\r')
            # the exact output you're looking for:
            stdout.write("Game %i/%i: [%-40s] %d%%, completed" % (
            count, len(HSS.games_lst), '=' * int(count / len(HSS.games_lst) * 40), 100 * count / len(HSS.games_lst)))
            stdout.flush()


"""
