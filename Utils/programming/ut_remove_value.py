""" This function removes entries of a list following a given criterion"""
from itertools import compress


def main(seq, filter):

    applyF  =   [eval(str(x)+filter) for x in seq]
    seq     =   list(compress(seq, applyF))
    return  seq

if __name__=='__main__':
    main()