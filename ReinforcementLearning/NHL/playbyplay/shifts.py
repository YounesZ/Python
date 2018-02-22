from copy import deepcopy

import numpy as np
import pandas as pd


class LineShifts(object):
    """Encapsulates queries done to determine line shifts."""

    def __init__(self, game):
        self.shifts   =   None
        self.equal_strength =   True
        self.regular_time   =   True
        self.min_duration   =   0 # minimum number of seconds for which we want to consider shifts.
        self.team   =   'both' # 'home', 'away' or 'both'
        # Pick the right team
        team        =   'both'
        tmP         =   {'home': 'h', 'away': 'a', 'both': 'ha'}[team]

        # Make containers
        LINES = {
            'playersID': [],
            'home_line': [],
            'away_line': [],
            'onice': [0],
            'office': [],
            'iceduration': [],
            'SHOT': [0],
            'GOAL': [0],
            'BLOCK': [0],
            'MISS': [0],
            'PENL': [0],
            'equalstrength': [True],
            'regulartime': [],
            'period': [],
            'differential': []
        }
        # Loop on all table entries
        prevDt          =   []
        prev_home_line  =   prev_away_line = np.ones([1, 3])[0]
        # prevLine = (np.ones([1, 3])[0], np.ones([1, 3])[0]) if team == 'both' else np.array([1, 1, 1])
        evTypes         =   ['GOAL', 'SHOT', 'PENL', 'BLOCK', 'MISS']
        for idL, Line in game.df_wc.iterrows():
            home_line   =   np.sort(game.pull_offensive_players(Line, 'h'))
            away_line   =   np.sort(game.pull_offensive_players(Line, 'a'))
            self.teams  =   [Line['hometeam'], Line['awayteam']]
            # curLine = (home_line, away_line)
            # if team == 'both':
            #     curLine = (home_line, away_line)
            #     teams = [Line['hometeam'], Line['awayteam']]
            # else:
            #     curLine = np.sort(game.pull_offensive_players(Line, tmP))
            #     teams = Line[team + 'team']

            # team of interest has changed?
            if len(prevDt) == 0:
                prevDt  =   Line
                thch    =   False
            else:
                thch    =   not (prev_home_line == home_line).all() or not (prev_away_line == away_line).all()
            # elif team == 'both':
            #     thch = not (prevLine[0] == curLine[0]).all() or not (prevLine[1] == curLine[1]).all()
            # else:
            #     thch = not (prevLine == curLine).all()

            if thch:
                # Terminate this shift
                LINES['playersID'].append((prev_home_line, prev_away_line))
                LINES['home_line'].append(prev_home_line)
                LINES['away_line'].append(prev_away_line)
                LINES['office'].append(prevDt['seconds'])
                LINES['iceduration'].append(LINES['office'][-1] - LINES['onice'][-1])
                LINES['period'].append(prevDt['period'])
                LINES['regulartime'].append(prevDt['period'] < 4)
                LINES['differential'].append(np.sum(LINES['GOAL']))
                # Start new shift
                LINES['onice'].append(prevDt['seconds'])
                LINES['equalstrength'].append(prevDt['away.skaters'] == 6 and prevDt['home.skaters'] == 6)
                LINES['SHOT'].append(0)
                LINES['GOAL'].append(0)
                LINES['PENL'].append(0)
                LINES['BLOCK'].append(0)
                LINES['MISS'].append(0)
            if any([x == Line['etype'] for x in evTypes]):
                sign    =   int(Line['hometeam'] == Line['ev.team']) * 2 - 1
                LINES[Line['etype']][-1] += sign
                if Line['etype'] == 'GOAL':
                    LINES['SHOT'][-1] += sign
            if Line['etype'] == 'PENL':
                LINES['equalstrength'][-1] = False
            prevDt      =   deepcopy(Line)
            prev_home_line = deepcopy(home_line)
            prev_away_line = deepcopy(away_line)
            # prevLine = deepcopy(curLine)

        # Terminate line history
        LINES['office'].append(Line['seconds'])
        LINES['iceduration'].append(LINES['office'][-1] - LINES['onice'][-1])
        LINES['playersID'].append((prev_home_line, prev_away_line))
        LINES['home_line'].append(prev_home_line)
        LINES['away_line'].append(prev_away_line)
        LINES['period'].append(prevDt['period'])
        LINES['regulartime'].append(prevDt['period'] < 4)
        LINES['differential'].append(np.sum(LINES['GOAL']))

        # ok, now let's buid it:
        self.shifts = pd.DataFrame.from_dict(LINES)
        # # all done, then:
        # return (team, teams, lineShifts)

    def as_df(self, team: str, equal_strength: bool, regular_time: bool, min_duration: int) -> pd.DataFrame:
        """Gets line shifts as a data frame."""
        df = self.shifts
        if equal_strength:
            df = df[df['equalstrength']]
        if regular_time:
            df = df[df['regulartime']]
        if not min_duration is None:
            df = df[df['iceduration'] >= min_duration]
        # for which team(s).
        if team == 'both':
            pass
            #print(df.columns.names)
            # df = df.drop(columns=['home_line', 'away_line']) # TODO: see https://github.com/pandas-dev/pandas/issues/19078
        elif team == 'home':
            # df = df.drop(columns=['playersID']) # TODO: see https://github.com/pandas-dev/pandas/issues/19078
            df = df.drop(['playersID'], axis=1)
            df = df.rename(columns={'home_line': 'playersID'})
        elif team == 'away':
            # df = df.drop(columns=['playersID']) # TODO: see https://github.com/pandas-dev/pandas/issues/19078
            df = df.drop(['playersID'], axis=1)
            df = df.rename(columns={'away_line': 'playersID'})
        else:
            raise RuntimeError("Can't choose elements from team '%s'" % (team))
        return df

    def __update__(self):
        pass