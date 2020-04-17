import numpy as np
import scipy.stats as st
from ._stats import conditional_permutation, type1errors, significant_fwer
import time

def prepare(B, C, s, T, Y): # see analyze(..) for parameter descriptions
    # add dummy batch info if none supplied
    if B is None:
        B = np.ones(len(Y))

    # verify samples are sorted by batch (for null permutation)
    if any(np.diff(B) < 0):
        print('ERROR: samples must be sorted by batch')

    # translate sample-level outcome to cell-level outcome
    y = np.repeat(1000*Y, C).astype(np.float64)

    # add all-ones column to sample-level covariates
    if T is not None:
        T_ = np.hstack([T, np.ones((len(T), 1))])
    else:
        T_ = np.ones((len(Y), 1))

    # combine sample-level and cell-level covariates into one cell-level matrix
    # and create cell-level weight vector
    if s is not None:
        u = np.hstack([
            np.repeat(T_, C, axis=0),
            s]).astype(np.float64)
    else:
        u = np.repeat(T_, C, axis=0)
    w = np.repeat(C, C).astype(np.float64)

    # project covariates out of outcome and weight it
    y = y - u.dot(np.linalg.solve(u.T.dot(u / w[:,None]), u.T.dot(y / w)))
    y /= w

    return B, u, w, y

def get_null(B, C, u, w, Y, num):
    nullY = conditional_permutation(B, Y.astype(np.float64), num)
    nully = np.repeat(1000*nullY, C, axis=0)
    nully -= u.dot(np.linalg.solve(u.T.dot(u / w[:,None]), u.T.dot(nully / w[:,None])))
    nully /= w[:,None]

    return nully

def get_null_mean(B, C, u, w, Y):
    nullMeans = np.array([
        np.mean(Y[B==b])
        for b in B
    ])
    nullmeans = np.repeat(1000*nullMeans, C, axis=0)
    nullmeans -= u.dot(np.linalg.solve(u.T.dot(u / w[:,None]), u.T.dot(nullmeans / w)))
    nullmeans /= w

    return nullmeans

def diffusion(a, Y, C, B=None, T=None, s=None,
        maxsteps=50, loops=1,
        stopthresh=0.05,
        Nnull=100, seed=0,
        outdetail=1, outfreq=5,
        null_info=False):
    """
    Carries out multi-condition analysis using diffusion.

    Parameters:
    a (scipy.sparse.csr.csr_matrix): adjacency matrix of graph, assumed not to contain
        self loops
    Y (1d numpy array): sample-level outcome to associate to graph location
    C (1d numpy array): number of cells in each sample
    B (1d numpy array): batch id of each sample
    T (2d numpy array): sample-level covariates to adjust for
    s (2d numpy array): cell-level covariates to adjust for
    maxsteps (int): maximum number of steps to take in random walk
    loops (float): strength of self loops to add
    stopthresh (float): diffusion stops when percent growth of number of hits goes below this
        number, provided number of hits is above 100. Passing None causes diffusion to
        continue for maxsteps steps.
    Nnull (int): number of null permutations to use to estimate mean and variance of
        null distribution and to perform FWER/FEP/FDR control.
    seed (int): random seed to use
    outdetail (int): level of printed output detail
    outfreq (int): how often to print output (every outfreq steps)

    Returns:
    #TODO update
    1d/2d array: a set of arrays, each of which gives the z-score of each cell at a given
        saved timepoint of the diffusion
    1d/2d array: a set of arrays, each of which gives the estimated FWER of each cell at a
        given saved timepoint of the diffusion. If significance is not None, then this instead
        contains True/False values indicating whether each cell is significant at each FWER
        theshold in sigificance.
    1d/2d array: a set of arrays, each of which gives the estimated probability of FDP <= 5%
        of each cell at a given saved timepoint of the diffusion. If significance is None,
        this array will be empty
    1d/2d array: a set of arrays, each of which gives the estimated FDR (= average FDP) of
        each cell at a given saved timepoint of the diffusion. If significance is None, this
        array will be empty
    0d/1d array: the set of timepoints corresponding to the returned results
    """
    def process_step():
        # compute z-scores
        std_c = np.sqrt((Nd_c**2).mean(axis=1))
        z_c = d_c / std_c
        Nz_c = Nd_c / std_c[:,None]
        h_c = significant_fwer(z_c, Nz_c, 0.05).sum()
        Nh_c = significant_fwer(Nz_c, Nz_c, 0.05).sum(axis=0)
        return h_c, Nh_c, z_c, Nz_c
    def stop_condition():
        # the +1 in the code below avoids a numpy warning, does not change the algorithm 
        # since h_c has to be avove 100 anyway
        return h_p >= 100 and (h_c - h_p) / (h_p + 1) < stopthresh, \
            (Nh_p >= 100) & ((Nh_c - Nh_p) / (Nh_p + 1) < stopthresh)

    # initialize random seed and result variables
    np.random.seed(seed)

    # prepare data
    B, u, w, y = prepare(B, C, s, T, Y)
    colsums = np.array(a.sum(axis=0)).flatten() + loops
    nullmean = get_null_mean(B, C, u, w, Y) #TODO probably should just residualize out batch

    # initialize time 0 of random walk. Naming conventions:
    #   d = raw diffusion scores, z = z-scores, h = number of hits
    #   *_c = current, *_p = previous, _f = final, N* = null
    d_c = y - nullmean
    Nd_c = get_null(B, C, u, w, Y, Nnull) - nullmean[:,None]
    z_c = np.zeros(d_c.shape)
    Nz_c = np.zeros(Nd_c.shape)
    z_f = np.zeros(d_c.shape)
    Nz_f = np.zeros(Nd_c.shape)
    t_f = None # the timepoint at which the real data stopped diffusing
    Nt_f = np.zeros(Nnull); Nt_f[:] = np.nan
    h_c, h_p = 0, 0
    Nh_c, Nh_p = np.zeros(Nnull), np.zeros(Nnull)

    # do diffusion
    t = 0
    start = time.time()
    _, _, z_c, Nz_c = process_step() # TODO: justify why we don't record h_c at time 0
    if outdetail > 0: print('t=0')

    for t in range(1, maxsteps+1):
        # take step
        h_p, Nh_p = h_c, Nh_c
        d_c = a.dot(d_c / colsums) + loops * d_c/colsums
        Nd_c = a.dot(Nd_c / colsums[:,None]) + loops * Nd_c / colsums[:,None]

        # compute z-scores and number of hits at 5% FWER
        h_c, Nh_c, z_c, Nz_c = process_step()

        # decide which random walks are finished
        stop, Nstop = stop_condition()
        if stop and t_f is None:
            t_f = t
            z_f = z_c
        nstopped = (Nstop & np.isnan(Nt_f)).sum()
        Nz_f[:,Nstop & np.isnan(Nt_f)] = Nz_c[:,Nstop & np.isnan(Nt_f)]
        Nt_f[Nstop & np.isnan(Nt_f)] = t

        # print progress
        if outdetail > 0 and t % outfreq == 0:
            print('t={:d} ({:.1f}s)'.format(t, time.time()-start))
        if outdetail > 1:
            print('\t', h_c, 'hits,', (h_c-h_p)/(h_p+1), 'relative growth')
            print('\t', nstopped, 'nulls stopped here,',
                (~np.isnan(Nt_f)).sum(), 'stopped total. t_f =', t_f)
    if outdetail > 0: print('t={:d}: finished'.format(t))

    # call all unfinished random walks as final
    if t_f is None:
        t_f = t
        z_f = z_c
    Nz_f[:,np.isnan(Nt_f)] = Nz_c[:,np.isnan(Nt_f)]
    Nt_f[np.isnan(Nt_f)] = t

    fwer, fep95, fdr = type1errors(z_f, Nz_f)

    if null_info:
        return z_f, fwer, fep95, fdr, t_f, Nt_f, Nz_f
    else:
        return z_f, fwer, fep95, fdr, t_f

def diffusion_uncorrected(a, Y, C, B=None, T=None, s=None,
        maxsteps=50, loops=1,
        keepevery=None,
        stopthresh=0.05,
        significance=None,
        Nnull=100, seed=0,
        outdetail=1, outfreq=5):
    """
    Carries out multi-condition analysis using diffusion.
    Parameters:
    a (scipy.sparse.csr.csr_matrix): adjacency matrix of graph, assumed not to contain
        self loops
    Y (1d numpy array): sample-level outcome to associate to graph location
    C (1d numpy array): number of cells in each sample
    B (1d numpy array): batch id of each sample
    T (2d numpy array): sample-level covariates to adjust for
    s (2d numpy array): cell-level covariates to adjust for
    maxsteps (int): maximum number of steps to take in random walk
    loops (float): strength of self loops to add
    keepevery (int): default (None) causes diffusion to return only the last timestep. Any
        other int will cause diffusion to return every keepevery-th step starting with t=0.
    stopthresh (float): diffusion stops when percent growth of number of hits goes below this
        number, provided number of hits is above 100. Passing None causes diffusion to
        continue for maxsteps steps.
    significance (float or iterable of floats): set of significance levels for which
        diffusion will perform significance testing with FWER control. Default is None,
        for which diffusion returns estimated FWERs for all cells as well as FDRs and FEPs,
        but this is slower.
    Nnull (int): number of null permutations to use to estimate mean and variance of
        null distribution and to perform FWER/FEP/FDR control.
    seed (int): random seed to use
    outdetail (int): level of printed output detail
    outfreq (int): how often to print output (every outfreq steps)
    Returns:
    1d/2d array: a set of arrays, each of which gives the z-score of each cell at a given
        saved timepoint of the diffusion
    1d/2d array: a set of arrays, each of which gives the estimated FWER of each cell at a
        given saved timepoint of the diffusion. If significance is not None, then this instead
        contains True/False values indicating whether each cell is significant at each FWER
        theshold in sigificance.
    1d/2d array: a set of arrays, each of which gives the estimated probability of FDP <= 5%
        of each cell at a given saved timepoint of the diffusion. If significance is None,
        this array will be empty
    1d/2d array: a set of arrays, each of which gives the estimated FDR (= average FDP) of
        each cell at a given saved timepoint of the diffusion. If significance is None, this
        array will be empty
    0d/1d array: the set of timepoints corresponding to the returned results
    """
    def process_step():
        # compute z-scores
        std_c = np.sqrt((Nd_c**2).mean(axis=1))
        z_c = d_c / std_c
        Nz_c = Nd_c / std_c[:,None]
        h_c = significant_fwer(z_c, Nz_c, 0.05).sum()
        return h_c, Nz_c, std_c, z_c
    def save_snapshot():
        ts.append(t)
        ds.append(d_c)
        zs.append(z_c)

        if significance is None:
            fwer, fep95, fdr = type1errors(z_c, Nz_c)
            fwers.append(fwer)
            feps95.append(fep95)
            fdrs.append(fdr)
        else:
            hits = significant_fwer(z_c, Nz_c, significance)
            fwers.append(hits)
    def stop_condition():
        if stopthresh is None:
            return False
        else:
            return h_c >= 100 and \
                (h_c - h_p) / (h_p + 1) < stopthresh # the +1 avoids a numpy warning, does
                                                        # not change the algorithm since h_c
                                                        # has to be avove 100 anyway

    # initialize random seed and result variables
    np.random.seed(seed)
    ds, zs, fwers, feps95, fdrs, ts = list(), list(), list(), list(), list(), list()

    # prepare data
    B, u, w, y = prepare(B, C, s, T, Y)
    colsums = np.array(a.sum(axis=0)).flatten() + loops
    nullmean = get_null_mean(B, C, u, w, Y)
    significance = np.array(significance) if significance is not None else None

    # initialize time 0 of random walk. Naming conventions:
    #   d = raw diffusion scores, z = z-scores, h = number of hits
    #   *_c = current, *_p = previous, N* = null
    d_c = y - nullmean
    Nd_c = get_null(B, C, u, w, Y, Nnull) - nullmean[:,None]
    std_c = np.zeros(d_c.shape)
    z_c = np.zeros(d_c.shape)
    Nz_c = np.zeros(Nd_c.shape)
    h_c, h_p = 0, 0

    # do diffusion
    start = time.time()

    t = 0
    h_c, Nz_c, std_c, z_c = process_step()
    if outdetail > 0: print('t=0')
    if keepevery is not None:
        save_snapshot()

    for t in range(1, maxsteps+1):
        # take step
        h_p = h_c
        d_c = a.dot(d_c / colsums) + loops * d_c/colsums
        Nd_c = a.dot(Nd_c / colsums[:,None]) + loops * Nd_c / colsums[:,None]

        # compute z-scores and number of hits at 5% FWER
        h_c, Nz_c, std_c, z_c = process_step()

        # print progress
        if outdetail > 0 and t % outfreq == 0:
            print('t={:d} ({:.1f}s)'.format(t, time.time()-start))
        if outdetail > 1:
            print(h_c, 'hits,', (h_c-h_p)/(h_p+1), 'percent growth')

        # decide whether to stop and whether to save current timestep
        if stop_condition():
            save_snapshot()
            break
        if keepevery is not None and t % keepevery == 0:
            save_snapshot()

    # save last timepoint if it isn't already saved
    if t not in ts:
        save_snapshot()

    if outdetail > 0: print('t={:d}: finished'.format(t))

    return np.array(zs).squeeze(), \
        np.array(fwers).squeeze(), \
        np.array(fdrs).squeeze(), \
        np.array(feps95).squeeze(), \
        np.array(ts).squeeze()
