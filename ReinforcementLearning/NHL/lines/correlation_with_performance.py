from ReinforcementLearning.NHL.lines.coach import Coach_Evaluator

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
dict_abbrevs['New York Rangers'] = 'NYR'
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
dict_abbrevs['Phoenix Coyotes'] = 'PHX'
dict_abbrevs['NY Islanders'] = 'NYI'
dict_abbrevs['New York Islanders'] = 'NYI'


def name2abbrev(name: str) -> str:
    a = name.split(" ")
    if a[0] in dict_abbrevs:
        return dict_abbrevs[a[0]]
    elif len(a) > 1 and ' '.join(a[0:2]) in dict_abbrevs:
        return dict_abbrevs[' '.join(a[0:2])]
    elif len(a) > 2 and ' '.join(a[0:3]) in dict_abbrevs:
        return dict_abbrevs[' '.join(a[0:3])]
    else:
        raise KeyError("'%s' does not have an abbreviation" % (name))

if __name__ == '__main__':
    from Utils.base import get_logger
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from ReinforcementLearning.NHL.config import Config


    common_logger = get_logger(name="common_logger", debug_log_file_name="common_logger.log")
    print("Debug will be written in {}".format(common_logger.handlers[1].baseFilename))

    coaches_evaluator = Coach_Evaluator(common_logger)
    grades = coaches_evaluator.read_and_display_coaches_evals()

    standings_file_name = os.path.join(Config().data_dir, "NHLStandings2012-2013.csv")
    df = pd.DataFrame.from_csv(standings_file_name)
    # add column with abbreviation
    df['ABBREV'] = pd.Series(list(map(name2abbrev, df["TEAM"].values)), index=df.index)
    df['GRADE'] = pd.Series(list(map(lambda abbrev: grades[abbrev], df["ABBREV"].values)), index=df.index)
    df['WINS'] = pd.Series(list(map(lambda record: int(record.split("-")[0]), df["Overall"].values)), index=df.index)
    df.plot(kind='scatter', x="GRADE", y="WINS") # , 'rx')
    plt.show()
