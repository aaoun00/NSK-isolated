import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.batch_space import SpikeClusterBatch
from core.spikes import SpikeCluster, Spike
from library.ensemble_space import CellEnsemble

import numpy as np

def feature_energy(data):
    """
    Normlized energy calculation discussed from:
    Quantitative measures of cluster quality for use in extraCellEnsembleular recordings by Redish et al.

    Args:
        data: ndarray representing spike data (num_channels X num_spikes X samples_per_spike)
        iPC (optional): the number
        norm (optional): normalize the waveform (True), or not (False).

    Returns:
        E:

    """

    # if isinstance(spike_cluster, CellEnsemble):
    #     data = spike_cluster.signal 
    # else:
    #     data = spike_cluster.waveforms

    # if isinstance(spike_cluster, Spike):
    #     data = np.asarray(data)
    #     data = data.reshape((data.shape[0], 1, data.shape[1]))
    # else:
    #     data = np.array(list(data.values()))


    # energy sqrt of the sum of the squares of each point of the waveform, divided by number of samples in waveform
    # energy and first principal component coefficient
    nSamp = data.shape[-1]

    sum_squared = np.sum(data ** 2, axis=-1)

    if len(np.where(sum_squared < 0)[0]) > 0:
        raise ValueError('summing squared values produced negative number!')

    E = np.divide(np.sqrt(sum_squared), np.sqrt(nSamp))  # shape: channel numbers x spike number
    # E[np.where(E == 0)] = 1  # remove any 0's so we avoid dividing by zero

    return E.T

def feature_wave_PCX(data, iPC=1, norm=True):
    """Creates the principal components for the waveforms

    Args:
        data: ndarray representing spike data (num_channels X num_spikes X samples_per_spike)
        iPC (optional): the number
        norm (optional): normalize the waveform (True), or not (False).

    Returns:
        FD:

    """
    
    # if isinstance(spike_cluster, CellEnsemble):
    #     data = spike_cluster.signal 
    # else:
    #     data = spike_cluster.waveforms

    # if isinstance(spike_cluster, Spike):
    #     data = np.asarray(data)
    #     data = data.reshape((data.shape[0], 1, data.shape[1]))
    # else:
    #     data = np.array(list(data.values()))

    nCh, nSpikes, nSamp = data.shape

    wavePCData = np.zeros((nSpikes, nCh))

    if norm:
        l2norms = np.sqrt(np.sum(data ** 2, axis=-1))
        l2norms = l2norms.reshape((nCh, -1, 1))

        data = np.divide(data, l2norms)

        # removing NaNs from potential division by zero
        data[np.where(np.isnan(data) == True)] = 0

    for i, w in enumerate(data):
        av = np.mean(w, axis=0)  # row wise average
        # covariance matrix
        cv = np.cov(w.T)
        sd = np.sqrt(np.diag(cv)).T  # row wise standard deviation

        pc, _, _, _ = _wave_PCA(cv)

        # standardize data to zero mean and unit variance
        wstd = (w - av) / (sd)

        # project the data onto principal component axes
        wpc = np.dot(wstd, pc)

        wavePCData[:, i] = wpc[:, iPC - 1]

    return wavePCData

# def create_features(spike_cluster: Spike | SpikeCluster | SpikeClusterBatch | CellEnsemble, featuresToCalculate=['energy', 'wave_PCX!1']):
def create_features(data, featuresToCalculate=['energy', 'wave_PCX!1']):

    """Creates the features to be analyzed

    Args:
        featuresToCalculate:

    Returns:
        FD:

    """

    FD = np.array([])
    for feature_name in featuresToCalculate:
        if 'wave_PCX' in feature_name:
            # instead of creating a function for every PCX you want to use, I just
            # created one. Use WavePCX!1 for PC1, WavePCX!2 for PC2, etc.

            # all the feature functions are named 'feature_featureName'
            # WavePCX is special though since you need a PC number so separate that
            feature_name, pc_number = feature_name.split('!')
            variables = 'data, %s' % pc_number
        else:
            variables = 'data'
        
        fnc_name = 'feature_%s' % feature_name

        current_FD = eval("%s(%s)" % (fnc_name, variables))

        if len(FD) == 0:
            FD = current_FD
        else:
            FD = np.hstack((FD, current_FD))

        current_FD = None

    # spike_cluster.stats_dict['cluster']['FD'] = FD

    return FD

def _wave_PCA(cv):
    """
    Principal Component Analysis of standardized waveforms from a
    given (unstandardized) waveform covariance matrix cv(nSamp,nSamp).

    Args:
        cv: nSamp x nSamp wavefrom covariance matrix (unnormalized)

    Returns:
        pc: column oriented principal components (Eigenvectors)
        rpc: column oriented Eigenvectors weighted with their relative amplitudes
        ev: eigenvalues of SVD (= std deviation of data projected onto pc)
        rev: relative eigenvalues so that their sum = 1

    """
    sd = np.sqrt(np.diag(cv)).reshape((-1, 1))  # row wise standard deviation

    # standardized covariance matrix
    cvn = np.divide(cv, np.multiply(sd, sd.T))

    if len(np.where(cvn != cvn)[0]) > 0:
        print('Nan in covariance matrix, replacing with 0 and proceeding')
        cvn[cvn != cvn] = 0
        print(cvn)
    u, ev, pc = np.linalg.svd(cvn)

    # the pc is transposed in the matlab version
    pc = pc.T

    ev = ev.reshape((-1, 1))

    rev = ev / np.sum(ev)  # relative eigne values so that their sum = 1

    rpc = np.multiply(pc, rev.T)

    return pc, rpc, ev, rev
