from os import listdir, path
from itertools import compress


def ut_find_folders(repo, nonEmpty=False):
    # Look for folders
    allS    =   listdir( repo )
    isD     =   [path.isdir( path.join(repo,x) ) for x in allS]
    allS    =   list( compress(allS, isD) )

    if nonEmpty:
        # Make sure they're not empty
        isF     =   [len(listdir( path.join(repo,x) ) )>0 for x in allS]
        allS    =   list( compress(allS, isF) )

    return allS