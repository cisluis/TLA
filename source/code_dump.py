def lmeVal(lmearr):
    """
    Generates an LME /single value code from LME array

    Parameters
    ----------
    - lmearr: (numpy) array with index values for a LME instance

    """
    v = 0
    for i in np.arange(lmearr.shape[0]):
        j = lmearr.shape[0] - (i + 1)
        v = v + (10**(2*j + 1))*lmearr[i, 0] + (10**(2*j))*lmearr[i, 1]

    return(int(v))


def lmeArr(lmeval, dim):
    """
    Generates an LME array from LME dim-digit code
    (eg 110120 => [[1,1], [0, 1], [2,0])

    Parameters
    ----------
    - lmeval: (int) LME numerical dim-digit code value
    - dim: (int) number of classes

    """
    def get_digit(number, n):
        return number // 10**n % 10

    lme = np.zeros([dim, 2])
    for i in np.arange(dim):
        j = dim - (i + 1)
        lme[i, 0] = get_digit(lmeval, 2*j + 1)
        lme[i, 1] = get_digit(lmeval, 2*j)

    return(lme)


def lmeGroup(lmearr, dim, thres=0.001):
    """
    Re-name infrequent (<0.1%) lme pixel species to closest frequent (>0.1%)
    species (as long as hamming distance is at most dim)

    Parameters
    ----------
    - lmearr: (numpy) original lmearr
    - dim: (int) number of cell classes
    - thres: (float) minimun frequency of accepted lme classes
    """

    def dist(x, y):
        # Hamming-like distance (with an ordered alphabet of states)
        w = np.array([np.ones(len(x)), np.ones(len(y))])
        return(np.trace(np.dot(w, np.abs(x - y))))

    newarr = lmearr.copy()

    # list unique lme values
    (unique, counts) = np.unique(lmearr, return_counts=True)
    freqs = np.asarray((unique, counts, counts/sum(counts))).T
    freqs = freqs[~np.isnan(freqs[:, 0])]

    # separate low and high frequency lme terms
    low = freqs[freqs[:, 2] < thres]
    hig = freqs[freqs[:, 2] >= thres]

    for i in range(len(low)):
        dlar = np.zeros(len(hig))
        x = lmeArr(low[i, 0], dim)
        for j in range(len(hig)):
            dlar[j] = dist(x, lmeArr(hig[j, 0], dim))

        d = np.nanmin(dlar)
        if d < (dim + 1):
            j = np.nanargmin(dlar)
            newarr[lmearr == low[i, 0]] = hig[j, 0]

    return(newarr)


