def main(dico, value):

    # Prepare holder
    eqKeys  =   ['']*len(value)
    keys    =   dico.keys()
    for k,v in dico.items():
        if v in value:
            idx             =   value.index(v)
            eqKeys[idx]     =   k
            dico[k]         =   -1

    return eqKeys


# Driver
if __name__ == "__main__":
    main()