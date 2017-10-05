import copy

def main(dico, value):

    dico2   =   copy.deepcopy(dico)
    valueCp =   copy.deepcopy(value)
    # Prepare holder
    eqKeys  =   ['']*len(valueCp)
    keys    =   dico2.keys()
    for k,v in dico2.items():
        if v in valueCp:
            idx             =   valueCp.index(v)
            eqKeys[idx]     =   k
            dico2[k]        =   -1
            valueCp[idx]    =   -1

    return eqKeys


# Driver
if __name__ == "__main__":
    main()