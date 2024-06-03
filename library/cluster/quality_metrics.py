import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import numpy as np
from scipy.stats import chi2
from library.cluster import create_features
from core.spikes import SpikeCluster
from library.ensemble_space import CellEnsemble
from library.batch_space import SpikeClusterBatch

def L_ratio(FD, ClusterSpikes):
    """Measures the L-Ratio, a cluster quality metric.

    Args:
        FD (ndarray): N by D array of feature vectors (N spikes, D dimensional feature space)
        ClusterSpikes (ndarray): Index into FD which lists spikes from the cell whose quality is to be evaluated.

    Returns:
        L :
        Lratio :
        df: degrees of freedom (number of features)

    """

    # if 'FD' not in spike_cluster.stats_dict['cluster']:
    #     FD = create_features(spike_cluster)
    # else:
    #     FD = spike_cluster.stats_dict['cluster']['FD']

    # if isinstance(spike_cluster, CellEnsemble):
    #     valid_ids = np.array(list(map(lambda x: x.cluster.cluster_label, spike_cluster.cells)), dtype=int)
    #     cluster_labels = np.array(spike_cluster.cells[0].cluster.cluster_labels, dtype=int)
    #     mask = list(map(lambda x: True if x in valid_ids else False, cluster_labels))
    #     ClusterSpikes = cluster_labels[mask]
    # else:
    #     ClusterSpikes = spike_cluster.cluster_labels

    try:

        nSpikes = FD.shape[0]

        nClusterSpikes = len(ClusterSpikes)

        # mark spikes which are not a part of the cluster as Noise
        NoiseSpikes = np.setdiff1d(np.arange(nSpikes), ClusterSpikes)

        # compute _mahalanobis distances
        m = _mahal(FD, FD[ClusterSpikes, :])

        df = FD.shape[1]

        L = np.sum(1 - chi2.cdf(m[NoiseSpikes], df))
        Lratio = L / nClusterSpikes

    except:
        L = np.nan
        Lratio = np.nan
        df = np.nan

    # spike_cluster.stats_dict['cluster']['L'] = L
    # spike_cluster.stats_dict['cluster']['L_ratio'] = Lratio
    # spike_cluster.stats_dict['cluster']['df'] = df

    return L, Lratio, df

def _mahal(Y, X):
    """_mahal _mahalanobis distance
    D2 = _mahal(Y,X) returns the _mahalanobis distance (in squared units) of
    each observation (point) in Y from the sample data in X, i.e.,

       D2(I) = (Y(I,:)-MU) * SIGMA^(-1) * (Y(I,:)-MU)',

    where MU and SIGMA are the sample mean and covariance of the data in X.
    Rows of Y and X correspond to observations, and columns to variables.  X
    and Y must have the same number of columns, but can have different numbers
    of rows.  X must have more rows than columns.

    Example:  Generate some highly correlated bivariate data in X.  The
    observations in Y with equal coordinate values are much closer to X as
    defined by _mahalanobis distance, compared to the observations with opposite
    coordinate values, even though they are all approximately equidistant from
    the mean using Euclidean distance.

       x = mvnrnd([0;0], [1 .9;.9 1], 100);
       y = [1 1;1 -1;-1 1;-1 -1];
       _mahalDist = _mahal(y,x)
       sqEuclidDist = sum((y - repmat(mean(x),4,1)).^2, 2)
       plot(x(:,1),x(:,2),'b.',y(:,1),y(:,2),'ro')


    Args:
        FD (ndarray): N by D array of feature vectors (N spikes, D dimensional feature space)
        ClusterSpikes (ndarray): Index into FD which lists spikes from the cell whose quality is to be evaluated.

    Returns:
        IsoDist: the isolation distance

    """
    rx, cx = X.shape
    ry, cy = Y.shape

    if cx != cy:
        raise ValueError('_mahal: Input Size Mismatch!')

    if rx < cx:
        raise ValueError('_mahal: Too few rows!')

    if len(np.where(np.iscomplex(X) == True)[0]) > 0:
        raise ValueError('_mahal: no complex values (X)!')
    elif len(np.where(np.iscomplex(Y) == True)[0]) > 0:
        raise ValueError('_mahal: no complex values (Y)!')

    m = np.mean(X, axis=0)

    M = np.tile(m, (ry, 1))
    C = X - np.tile(m, (rx, 1))

    Q, R = np.linalg.qr(C)

    ri = np.linalg.lstsq(R.T, (Y - M).T, rcond=None)[0]

    d = np.sum(np.multiply(ri, ri), axis=0).T * (rx - 1)

    return d

def isolation_distance(FD, ClusterSpikes):
    """Measures the Isolation Distance, a cluster quality metric.

    Args:
        FD (ndarray): N by D array of feature vectors (N spikes, D dimensional feature space)
        ClusterSpikes (ndarray): Index into FD which lists spikes from the cell whose quality is to be evaluated.

    Returns:
        IsoDist: the isolation distance

    """
    # if 'FD' not in spike_cluster.stats_dict['cluster']:
    #     FD = create_features(spike_cluster)
    # else:
    #     FD = spike_cluster.stats_dict['cluster']['FD']
    
    # if isinstance(spike_cluster, CellEnsemble):
    #     valid_ids = np.array(list(map(lambda x: x.cluster.cluster_label, spike_cluster.cells)), dtype=int)
    #     cluster_labels = np.array(spike_cluster.cells[0].cluster.cluster_labels, dtype=int)
    #     mask = list(map(lambda x: True if x in valid_ids else False, cluster_labels))
    #     ClusterSpikes = cluster_labels[mask]
    # else:
    #     ClusterSpikes = spike_cluster.cluster_labels

    nSpikes = FD.shape[0]

    nClusterSpikes = len(ClusterSpikes)

    print(nClusterSpikes, nSpikes)

    if nClusterSpikes > nSpikes / 2:
        IsoDist = np.NaN  # not enough out-of-cluster-spikes - IsoD undefined

    else:

        try:

            # InClu = ClusterSpikes
            OutClu = np.setdiff1d(np.arange(nSpikes), ClusterSpikes)

            # compute _mahalanobis distances
            m = _mahal(FD, FD[ClusterSpikes, :])

            mNoise = m[OutClu]  # _mahal dist of all other spikes

            # calculate point where mD of other spikes = n of this cell
            sorted_values = np.sort(mNoise)
            IsoDist = sorted_values[nClusterSpikes - 1]

        except:

            IsoDist = np.nan

    # spike_cluster.stats_dict['cluster']['iso_dist'] = IsoDist

    return IsoDist
