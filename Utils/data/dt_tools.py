def time_lineups(seq, header, gameHome=True, filter=None):

    # Look for line info
    gameloc         =   'h'
    if not(gameHome):
        gameloc     =   'a'
    plCol   =   [header.index(gameloc+str(x)) for x in [1,2,3]]
    tmCol   =   header.index('seconds')

    # Time the lines
    lines   =   {}
    prevTm  =   0
    for ii, jj in zip(seq, filter):

        linii   =   tuple( sorted([ii[x] for x in plCol]) )
        timii   =   float( ii[tmCol] )

        if jj:
            continue

        if not(linii in lines):
            lines[linii]    =   0

        lines[linii]    +=  timii-prevTm
        prevTm          =   timii
    return lines


