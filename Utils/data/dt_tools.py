import csv
import re


class Table:
    def __init__(self, fileLoc, fileName):
        # Data location
        self.fileLoc = fileLoc
        self.fileName = fileName

    def read(self):
        # ======= Read PLAY-BY-PLAY
        # Load file
        PbP = []
        with open(self.fileLoc + '/' + self.fileName, newline='') as csvfile:
            spamreader = csv.reader(csvfile, dialect='excel', delimiter='\t', quotechar='|')
            for row in spamreader:
                PbP.append(re.sub('"', '', row[0]).split(','))
        self.header = PbP[0]
        self.dataRaw = PbP[1:]

    def get_column_idx(self, colName):
        colId = self.header.index(colName)
        colDt = [x[colId] for x in self.dataRaw]
        return colId, colDt

    def filter_column(self, colId_target, colId_filter, filter, table=None):
        if table == None:
            table = self.dataRaw
        colFlt = []
        colIdx = []
        count = 0
        for ii in table:
            if ii[colId_filter] == filter:
                colFlt.append(ii[colId_target])
                colIdx.append(count)
            count += 1
        return colFlt, colIdx

    def get_column_range(self, colId, value):
        vRng = {'start': [], 'end': []}
        prev = False
        count = 0
        for ii in self.dataRaw:
            if ii[colId] == value and not (prev):
                vRng['start'].append(count)
            elif ii[colId] != value and prev:
                vRng['end'].append(count - 1)
            prev = ii[colId] == value
            count += 1
        return vRng

    def slice_table(self, range_row, range_col):
        slice = []
        for ii in range(range_row[0], range_row[1] + 1):
            slice.append(self.dataRaw[ii][range_col[0]:range_col[1] + 1])
        return slice


class PBPtable(Table):
    def __init__(self, fileLoc, fileName, plClasses=['C', 'L', 'R', 'D']):
        # Data location
        Table.__init__(self, fileLoc, fileName)
        self.plClasses = plClasses

    def detect_power_plays(self):
        # Find home player indices
        hpi = [self.get_column_idx('h' + str(x)) for x in [1, 5]]
        hpi = [x[0] for x in hpi]
        # Find away player indices
        api = [self.get_column_idx('a' + str(x)) for x in [1, 5]]
        api = [x[0] for x in api]
        # Loop and detect powerplays: 0=5on5, 1=homePplay, -1=awayPplay
        self.header += ['powerplay']
        for ii in range(len(self.dataRaw)):
            # Home players
            plH = self.dataRaw[ii][hpi[0]:hpi[1] + 1]
            plH = 5 - sum([x == '1' for x in plH])
            # Away players
            plA = self.dataRaw[ii][api[0]:api[1] + 1]
            plA = 5 - sum([x == '1' for x in plA])
            # classify play
            self.dataRaw[ii] += [plH * 10 + plA]


class ROSTERtable(Table):
    def __init__(self, fileLoc, fileName, plClasses=['C', 'L', 'R', 'D']):
        # Data location
        Table.__init__(self, fileLoc, fileName)
        self.plClasses = plClasses


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

        linii   =   tuple( sorted([int(ii[x]) for x in plCol]) )
        timii   =   float( ii[tmCol] )

        if jj:
            prevTm = timii
            continue

        if not(linii in lines):
            lines[linii]    =   0

        lines[linii]    +=  timii-prevTm
        prevTm          =   timii
    return lines



def shift_lineups(seq, header, gameHome=True, filter=None):

    # Look for line info
    gameloc         =   'h'
    if not(gameHome):
        gameloc     =   'a'
    plCol   =   [header.index(gameloc+str(x)) for x in [1,2,3]]
    tmCol   =   header.index('seconds')

    # Time the lines
    lines   =   {}
    prevLn  =   (0,0,0)
    for ii, jj in zip(seq, filter):

        linii   =   tuple( sorted([int(ii[x]) for x in plCol]) )

        if jj:
            prevLn = linii
            continue

        if not(linii in lines):
            lines[linii]    =   0

        if prevLn != linii:
            lines[linii]    +=  1
        prevLn          =   linii
    return lines