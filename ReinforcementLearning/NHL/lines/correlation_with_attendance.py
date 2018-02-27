from ReinforcementLearning.NHL.lines.coach import Coach_Evaluator
from ReinforcementLearning.NHL.lines.correlation_with_performance import name2abbrev

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

    attendance_file_name = os.path.join(Config().data_dir, "NHLAttendance2012-2013.csv")
    df = pd.DataFrame.from_csv(attendance_file_name)
    # add column with abbreviation
    df['ABBREV'] = pd.Series(list(map(name2abbrev, df["TEAM"].values)), index=df.index)
    df['GRADE'] = pd.Series(list(map(lambda abbrev: grades[abbrev], df["ABBREV"].values)), index=df.index)
    df.AVG = df.AVG.str.replace(",", "")
    df.AVG = df.AVG.astype(float)
    df.plot(kind='scatter', x="GRADE", y="AVG") # , 'rx')
    plt.show()
