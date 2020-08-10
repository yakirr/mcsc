import scanpy as sc
from ._multisample import issorted, sortedcopy, sample_size

# assumes that data.obs contains a field for sample id and a field for batch id
def init(data, sampleid='id', batchid='batch', inplace=True):
    print('creating and populating sampleXmeta')
    data.uns['sampleXmeta'] = \
        data.obs[[sampleid, batchid]].drop_duplicates().set_index(sampleid)

    if not issorted(data):
        if inplace:
            raise ValueError('data.obs is not sorted by sample; this is incompatible with '+\
                'inplace=True')
        print('sorting cells by sample')
        data = sortedcopy(data)
    else:
        if not inplace:
            data = data.copy()

    print('computing sample sizes')
    data.uns['sampleXmeta']['C'] = sample_size(data)

    print('computing nn graph')
    sc.pp.neighbors(data)

    print('computing umap for visualization')
    #sc.tl.umap(data)

    return data
