import shutil
import errno
from os import path


def ut_clone_directory(src, dst):

    if path.exists(dst):
        shutil.rmtree(dst)

    try:
        shutil.copytree(src, dst)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else:
            print('Directory not copied. Error: %s' % e)



# ========
# LAUNCHER
"""
src     =   '/home/younesz/Documents/Code/NHL_stats_SL/ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_perceptron_1layer_10units_relu'
dst     =   path.join( src.replace(path.basename(src), ''), 'MODELS', 'MODEL_perceptron_1layer_10units_relu_LOO_SeasonXXX' )
ut_clone_directory(src, dst)
"""