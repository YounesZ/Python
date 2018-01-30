from ReinforcementLearning.NHL.playerstats.nhl_player_stats import *
from ReinforcementLearning.NHL.playbyplay.game import *


def get_game_id(home_team_abbr, date_as_str):
    """
    let's convert game date to game code.
    For example Montreal received Ottawa on march 13, 2013 =>
        gameId = get_game_id(home_team_abbr='MTL', date_as_str='2013-03-13')
    """
    try:
        gameInfo    =   pickle.load( open(path.join(db_root, 'gamesInfo.p'), 'rb') )
        gameInfo    =   gameInfo[gameInfo['gameDate']==date_as_str][gameInfo['teamAbbrev']==home_team_abbr]
        gameId      =   gameInfo['gameId']
        gameId      =   int( gameId.values.astype('str')[0][5:] )
        return gameId
    except Exception as e:
        raise IndexError("There was no game for '%s' on '%s'" % (home_team_abbr, date_as_str))


def get_players_classes(repoModel, data_for_game, number_of_games):
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
    # Need to load the data pre-processing variables
    preprocessing = pickle.load(open(path.join(repoModel, 'baseVariables.p'), 'rb'))

    # Need to load the classification model (for players' predicted ranking on trophies voting lists)
    classifier = {'sess': tf.Session(), 'annX': [], 'annY': []}
    saver = tf.train.import_meta_graph(path.join(repoModel, path.basename(repoModel) + '.meta'))
    graph = classifier['sess'].graph
    classifier['annX'] = graph.get_tensor_by_name('Input_to_the_network-player_features:0')
    classifier['annY'] = graph.get_tensor_by_name('prediction:0')
    saver.restore(classifier['sess'], tf.train.latest_checkpoint(path.join(repoModel, './')))

    # Pick players stats - last 'n' games
    data_for_game.pull_line_shifts(team='both', minduration=20)
    data_for_game.pick_regulartime()
    data_for_game.pick_equalstrength()
    data_for_game.pull_players_classes(preprocessing, classifier, nGames=number_of_games)
    return data_for_game.player_classes


def get_model(repoModel):

    # Prepare players model:reload info
    players_model  =   pickle.load(open(path.join(repoModel, 'baseVariables.p'), 'rb'))
    classifier     =   {'sess':tf.Session(), 'annX':[], 'annY':[]}
    saver          =   tf.train.import_meta_graph(path.join(repoModel, path.basename(repoModel) + '.meta'))
    graph          =   classifier['sess'].graph
    classifier['annX'] =   graph.get_tensor_by_name('Input_to_the_network-player_features:0')
    classifier['annY'] =   graph.get_tensor_by_name('prediction:0')
    saver.restore(classifier['sess'], tf.train.latest_checkpoint(path.join(repoModel, './')))
    return players_model, classifier




# ========
# LAUNCHER
# ========
# === Set pointers
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

# === Pipeline
# Select a season
season      =   '20122013'

# List all games
HSS         =   HockeySS(repoPbP, repoPSt)
HSS.list_all_games()

# Keep only selected season
HSS.games_lst  =  HSS.games_lst[HSS.games_lst['season']==int(season)]

# List all teams
allTeams    =   HSS.games_lst[['hometeam', 'awayteam']]
allTeams    =   np.unique( np.concatenate(allTeams.values) )

# Initialize empty containers
seaInfo     =   dict([(x, pd.DataFrame(columns=['gameCode', 'avgQ', 'points', 'goalsFor', 'goalsAg'])) for x in allTeams])

# Line translation table
linedict    =   HockeySS(repoPbP, repoPSt)
linedict.make_line_dictionary()
linedict    =   linedict.line_dictionary

# Qvalue table
# Load the Qvalues table
Qvalues     =   pickle.load( open(path.join(repoCode, 'ReinforcementLearning/NHL/playbyplay/data/RL_action_values.p'), 'rb') )['action_values']

# Loop on games
count       =   0
players_model, classifier   =   get_model(repoModel)
for iG in range(len(HSS.games_lst)):

    # Get game code
    gameCode    =   HSS.games_lst.iloc[iG]['gcode']

    # Get game data
    gameData    =   Game(repoPbP, repoPSt, season, gameCode)
    gameData.pull_line_shifts('both', minduration=20)
    gameData.pick_regulartime()
    gameData.pick_equalstrength()
    gameData.pull_players_classes(players_model, classifier)

    # Get game score
    homeTeam = HSS.games_lst.iloc[iG]['hometeam']
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

