import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st
import time, gc
from argparse import Namespace

#TODO rename sampleXmeta to samplem or samples, consider making it an attribute of data

# creates a neighborhood frequency matrix
#   requires data.uns[sampleXmeta][ncellsid] to contain the number of cells in each sample.
#   this can be obtained using mcsc.pp.sample_size
def nfm(data, nsteps=3, sampleXmeta='sampleXmeta', ncellsid='C', key_added='sampleXnh'):
    a = data.uns['neighbors']['connectivities']
    C = data.uns[sampleXmeta][ncellsid].values
    colsums = np.array(a.sum(axis=0)).flatten() + 1
    s = np.repeat(np.eye(len(C)), C, axis=0)

    for i in range(nsteps):
        print(i)
        s = a.dot(s/colsums[:,None]) + s/colsums[:,None]
    snorm = s / C

    data.uns[key_added] = snorm.T

# creates a cluster frequency matrix
#   data.obs[clusters] must contain the cluster assignment for each cell
def cfm(data, clusters, sampleXmeta='sampleXmeta', sampleid='id', key_added=None):
    if key_added is None:
        key_added = 'sampleX'+clusters

    sm = data.uns[sampleXmeta]
    nclusters = len(data.obs[clusters].unique())
    cols = []
    for i in range(nclusters):
        cols.append(clusters+'_'+str(i))
        sm[cols[-1]] = data.obs.groupby(sampleid)[clusters].aggregate(
            lambda x: (x.astype(np.int)==i).mean())
    #TODO this should use pd.get_dummies and then aggregate that instead

    data.uns[key_added] = sm[cols].values
    sm.drop(columns=cols, inplace=True)

def pca(data, repname='sampleXnh', npcs=None):
    if npcs is None:
        npcs = min(*data.uns[repname].shape)
    s = data.uns[repname].copy()
    s = s - s.mean(axis=0)
    s = s / s.std(axis=0)
    ssT = s.dot(s.T)
    V, d, VT = np.linalg.svd(ssT)
    U = s.T.dot(V) / np.sqrt(d)
    del s; gc.collect()

    data.uns[repname+'_sqevals'] = d[:npcs]
    data.uns[repname+'_featureXpc'] = U[:,:npcs]
    data.uns[repname+'_sampleXpc'] = V[:,:npcs]

def mixedmodel(data, phenoname, covnames=[],
        batchname='batch', npcs=20, repname='sampleXnh', usepca=True,
        pval='lrt', badbatch_r2=0.05, outputlevel=1):
    #extract relevant covariates from data
    Y = data.uns['sampleXmeta'][phenoname]
    T = data.uns['sampleXmeta'][covnames]
    if batchname is not None:
        B = data.uns['sampleXmeta'][batchname]
    else:
        B = pd.DataFrame(np.ones(len(Y)), columns=['batch'])

    #compute nfm if not found
    if repname not in data.uns and repname=='sampleXnh':
        print('computing neighborhood frequency matrix')
        nfm(data)
    elif repname not in data.uns:
        raise ValueError(repname + ' not found')

    # compute PCA if needed and not found
    if npcs is None or npcs > min(*data.uns[repname].shape):
        npcs = min(*data.uns[repname].shape) - 1
    if usepca and repname+'_sampleXpc' not in data.uns.keys() \
        or usepca and len(data.uns[repname+'_sqevals']) < npcs:
        print('computing PCA')
        pca(data, repname=repname, npcs=npcs)

    # define X
    if usepca:
        X = pd.DataFrame(data.uns[repname+'_sampleXpc'][:,:npcs],
                index=data.uns['sampleXmeta'].index,
                columns=['PC'+str(i) for i in range(npcs)])
    else:
        X = pd.DataFrame(data.uns[repname],
                columns=['X'+str(i) for i in range(len(X.T))])

    # add any problematic batches as fixed effects
    corrs = np.array([
        np.corrcoef((B==b).astype(np.float), X.T)[0,1:]
        for b in np.unique(B)
    ])
    badbatches = np.unique(B)[(corrs**2).max(axis=1) > badbatch_r2]
    badbatchnames = ['B'+str(b) for b in badbatches]
    if len(badbatches) > 0:
        batch_fe = np.array([
            (B == b).astype(np.float)
            for b in badbatches]).T
        B = B.copy()
        B[np.isin(B, badbatches)] = -1
    else:
        batch_fe = np.zeros((len(X), 0))
    batch_fe = pd.DataFrame(batch_fe, columns=badbatchnames,
                                    index=data.uns['sampleXmeta'].index)
    #TODO: account for the case where all batche are bad batches, in which case
    #   we have a singular matrix

    # construct the dataframe for the analysis
    df = pd.concat([X, T, batch_fe, B, Y], axis=1)

    # build the null model
    fixedeffects = list(covnames) + list(batch_fe.columns)
    md0 = smf.mixedlm(
        phenoname + ' ~ ' + ('1' if fixedeffects == [] else '+'.join(fixedeffects)),
        df, groups=batchname)
    mdf0 = md0.fit(reml=False)
    if outputlevel > 0: print(mdf0.summary())

    # build necessary alternative models and compute p-values
    res = {}
    if pval == 'lrt':
        md1 = smf.mixedlm(
            phenoname + ' ~ ' + '+'.join(fixedeffects+list(X.columns)),
            df, groups='batch')
        mdf1 = md1.fit(reml=False)
        if outputlevel > 0: print(mdf1.summary())
        llr = mdf1.llf - mdf0.llf
        res['p'] = st.chi2.sf(2*llr, len(X.columns))
        res['gamma'] = mdf1.params[X.columns] # coefficients in linear regression
        res['gamma_p'] = mdf1.pvalues[X.columns] # p-values for individual coefficients
        if usepca:
            res['gamma_scale'] = np.sqrt(data.uns[repname + '_sqevals'][:npcs])
            V = data.uns[repname + '_featureXpc'][:,:npcs]
            res['beta'] = V.dot(res['gamma'] / res['gamma_scale'])
        return Namespace(**res)
    else:
        print('ERROR: the only pval method currently supported is lrt')
        return None, None, None
