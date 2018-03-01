import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from os import path

from Utils.base import hashable_dict
from ReinforcementLearning.NHL.playerstats.nhl_player_stats import PlayerStatsFetcher, do_normalize_data, do_reduce_data
from Utils.programming.ut_sanitize_matrix import ut_sanitize_matrix

def get_model_and_classifier_from(repoModel: str):
    """

    Args:
        repoModel: the place where the model is saved.

    Returns: A tuple (model, classifier

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
    return (preprocessing, classifier)

class players_classes(object):

    def __init__(self, game_data, model, classifier, stats_fetcher='default'):
        self.game_data = game_data
        self.model = model
        self.classifier = classifier
        self.stats_fetcher=stats_fetcher
        self.players_classes_cache = {}


    @classmethod
    def from_repo(cls, game_data, repoModel: str):
        # Need to load the data pre-processing variables
        preprocessing, classifier = get_model_and_classifier_from(repoModel)
        # preprocessing = pickle.load(open(path.join(repoModel, 'baseVariables.p'), 'rb'))
        #
        # # Need to load the classification model (for players' predicted ranking on trophies voting lists)
        # classifier = {'sess': tf.Session(), 'annX': [], 'annY': []}
        # saver = tf.train.import_meta_graph(path.join(repoModel, path.basename(repoModel) + '.meta'))
        # graph = classifier['sess'].graph
        # classifier['annX'] = graph.get_tensor_by_name('Input_to_the_network-player_features:0')
        # classifier['annY'] = graph.get_tensor_by_name('prediction:0')
        # saver.restore(classifier['sess'], tf.train.latest_checkpoint(path.join(repoModel, './')))

        return cls(game_data, model=preprocessing, classifier=classifier)

    def set_stats_fetcher(self, fetcher):
        self.stats_fetcher = fetcher

    def get(self, equal_strength: bool,
                             regular_time: bool,
                             min_duration: int,
                             nGames=30):
        """
        Gets classes players represented in the shifts for this game.
        Assumes that these shifts are specified (by a previous call to 'pull_line_shifts'
        If the shifts are not specified throws an error.
        Args:
            model: 
            classifier: 
            nGames: 

        Returns:
            Player classes


        """
        cache_key = hashable_dict(
            equal_strength=equal_strength,
            regular_time=regular_time,
            min_duration=min_duration,
            number_of_games=nGames)
        if cache_key not in self.players_classes_cache:

            # List concerned players
            all_line_shifts = self.game_data.lineShifts.as_df('both', equal_strength, regular_time, min_duration)
            # teams_label_for_shift, teams, all_line_shifts = self.calculate_line_shifts(team='both')
            all_pl = all_line_shifts['playersID'].values
            if len(all_pl) == 0:
                self.player_classes = []
                return
            all_plC = np.unique(np.concatenate(all_pl))
            all_plN = self.game_data.rf_wc.set_index('player.id').loc[all_plC[all_plC > 1]]['firstlast'].drop_duplicates(keep='first')
            if len(all_plN) == 0:
                self.player_classes = []
                return
            # Get players' team
            Hp = np.unique(np.concatenate([x[0] for x in all_pl]))
            Ap = np.unique(np.concatenate([x[1] for x in all_pl]))
            # pTeam   =   [ np.where([x in Hp, x in Ap])[0] for x in all_plN.index.values]
            pTeam = [self.game_data.lineShifts.teams[0] if x in Hp else self.game_data.lineShifts.teams[1] for x in all_plN.index.values]
            # Get raw player stats
            # gcode   =   int( str(self.season)[:4]+'0'+str(self.gameId) )
            gcode = int(str(self.game_data.season.year_begin) + '0' + str(self.game_data.gameId))
            if self.stats_fetcher == 'default':
                DT, dtCols = self.game_data.stats_fetcher.pull_stats(uptocode=gcode, nGames=nGames, plNames=all_plN.values)
            else:
                DT, dtCols = self.stats_fetcher.pull_stats(uptocode=gcode, nGames=nGames, plNames=all_plN.values)
            # --- Get player classes
            # pre-process data
            DT[dtCols] = ut_sanitize_matrix(DT[dtCols])
            DT_n, _ = do_normalize_data(DT[dtCols], normalizer=self.model['normalizer'])
            DT_n_p, _ = do_reduce_data(DT_n, pca=self.model['pca'])
            # model players' performance
            DTfeed = {self.classifier['annX']: DT_n_p}
            classif = self.classifier['sess'].run(self.classifier['annY'], feed_dict=DTfeed)
            # Get players class
            ctrLst = np.array(
                (self.model['global_centers']['selke'], self.model['global_centers']['ross'], self.model['global_centers']['poor']))
            pl_class = [np.argmin(np.sum((classif[x, :] - ctrLst) ** 2, axis=1)) for x in range(classif.shape[0])]
            pl_class = pd.DataFrame(pl_class, columns=['class'], index=all_plN.index.values)
            pl_class.loc[:, 'firstlast'] = all_plN
            pl_class.loc[:, 'pred_ross'] = classif[:, 0]
            pl_class.loc[:, 'pred_selke'] = classif[:, 1]
            pl_class.loc[:, 'team'] = pTeam
            self.players_classes_cache[cache_key] = pl_class
        return self.players_classes_cache[cache_key]

