import numpy as np
import scipy.stats as st

def conditional_permutation(B, Y, num):
    """
    Permutes Y conditioned on B num different times.
    """
    batchind = np.array([
        np.where(B == b)[0] for b in np.unique(B)
        ])
    ix = np.concatenate([
        bi[np.argsort(np.random.randn(len(bi), num), axis=0)]
        for bi in batchind
        ])
    bix = np.zeros((len(Y), num)).astype(np.int)
    bix[np.concatenate(batchind)] = ix
    result = Y[bix]
    return Y[bix]
