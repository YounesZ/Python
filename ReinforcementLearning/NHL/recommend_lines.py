# -*- coding: utf-8 -*-
"""Example of Line Recommendation.

This module demonstrates the usage of a line recommender.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Attributes:

Todo:
    * Nothing for now.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

from ReinforcementLearning.NHL.lines.coach import Coach_Evaluator

if __name__ == '__main__':
    from Utils.base import get_logger
    import pandas as pd
    import matplotlib.pyplot as plt

    dict_abbrevs = {}
    dict_abbrevs['Chicago'] = 'CHI'
    dict_abbrevs['Montreal'] = 'MTL'
    dict_abbrevs['Detroit'] = 'DET'
    dict_abbrevs['Philadelphia'] = 'PHI'
    dict_abbrevs['Toronto'] = 'TOR'
    dict_abbrevs['Ottawa'] = 'OTT'
    dict_abbrevs['Calgary'] = 'CGY'
    dict_abbrevs['Tampa Bay'] = 'T.B'
    dict_abbrevs['Buffalo'] = 'BUF'
    dict_abbrevs['Vancouver'] = 'VAN'
    dict_abbrevs['Minnesota'] = 'MIN'
    dict_abbrevs['Pittsburgh'] = 'PIT'
    dict_abbrevs['Los Angeles'] = 'L.A'
    dict_abbrevs['Washington'] = 'WSH'
    dict_abbrevs['Boston'] = 'BOS'
    dict_abbrevs['San Jose'] = 'S.J'
    dict_abbrevs['Carolina'] = 'CAR'
    dict_abbrevs['St. Louis'] = 'STL'
    dict_abbrevs['NY Rangers'] = 'NYR'
    dict_abbrevs['New Jersey'] = 'N.J'
    dict_abbrevs['Dallas'] = 'DAL'
    dict_abbrevs['Florida'] = 'FLA'
    dict_abbrevs['Nashville'] = 'NSH'
    dict_abbrevs['Edmonton'] = 'EDM'
    dict_abbrevs['Anaheim'] = 'ANA'
    dict_abbrevs['Colorado'] = 'COL'
    dict_abbrevs['Winnipeg'] = 'WPG'
    dict_abbrevs['Columbus'] = 'COL'
    dict_abbrevs['Arizona'] = 'PHX'
    dict_abbrevs['NY Islanders'] = 'NYI'

    common_logger = get_logger(name="common_logger", debug_log_file_name="common_logger.log")
    print("Debug will be written in {}".format(common_logger.handlers[1].baseFilename))

    coaches_evaluator = Coach_Evaluator(common_logger)
    # coaches_evaluator.evaluate_all_coaches(season_year_begin = 2012, teams_opt=None, n_games=2) # 10) # ['PHX'])
    grades = coaches_evaluator.read_and_display_coaches_evals()

    df = pd.DataFrame.from_csv("/Users/luisd/Downloads/NHLAttendance2012-2013.csv")
    # add column with abbreviation
    def name2abbrev(name: str) -> str:
        a = name.split(" ")
        if len(a) > 2:
            key = ' '.join(a[:-1])
        else:
            key = name
        return dict_abbrevs[key]

    df['ABBREV'] = pd.Series(list(map(name2abbrev, df["TEAM"].values)), index=df.index)
    df['GRADE'] = pd.Series(list(map(lambda abbrev: grades[abbrev], df["ABBREV"].values)), index=df.index)
    df.AVG = df.AVG.str.replace(",", "")
    df.AVG = df.AVG.astype(float)
    # plt.figure()
    # plt.interactive(False)
    df.plot(kind='scatter', x="GRADE", y="AVG") # , 'rx')
    plt.show()
    print("hello")
    # import cProfile
    # cProfile.run('evaluate_mtl_coach()', '/tmp/restats')
    # import pstats
    #
    # see_top = 25
    #
    # p = pstats.Stats('/tmp/restats')
    # p.sort_stats('cumulative').print_stats(see_top)
    #
    # p.sort_stats('time').print_stats(see_top)
