

"""
This module contains functions for extracting waveform features for each unit.

- Functions for extracting a long feature vector for each unit.
- Functions for summarizing aggregated unit features.
"""

import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

from _prototypes.unit_matcher.spike import (
    spike_features
)
from core.spikes import SpikeCluster

# Utility functions for comparing distributions

#TODO test
def jensen_shannon_distance(X:np.array, Y:np.array):
    """Compute the Jensen-Shannon distance between two probability distributions.

    Input
    -----
    X, Y : 2D arrays (sample_size, dimensions)
        Probability distributions of equal length that sum to 1

    Output
    ------
    jensen_shannen_distance : float
    """
    # print(P.shape, Q.shape)
    X_sample_size, X_dimensions = X.shape
    Y_sample_size, Y_dimensions = Y.shape
    assert X_dimensions == Y_dimensions, f"Dimensionality of X ({X_dimensions}) and Y ({Y_dimensions}) must be equal"
    dimensions = X_dimensions

    if dimensions == 1:
        # samples = [P_sample_size, Q_sample_size]
        # arg_largest = np.argmax(samples)
        # max_sample_size = np.min(samples)
        # to_shuffle = np.copy(samples[arg_largest])
        # np.shuffle(to_shuffle)

        # bin_count
        n = 100

        # kernel density
        # kde_P = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
        # pdf_P = kde_P.score_samples(X)
        # kde_Q = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(Y)
        # pdf_Q = kde_P.score_samples(Y)

        # histo
        cts_P, bins_P = np.histogram(X, bins=n) # will these be the same bins?
        cts_Q, bins_Q = np.histogram(Y, bins=n) # will these be the same bins?

        P = cts_P / sum(cts_P)
        Q = cts_Q / sum(cts_Q)

        M = (cts_P + cts_Q) / sum(cts_P + cts_Q)

        # print(M[:10])

        # print(M[:10])

        _kldiv = lambda A, B: np.sum([v for v in A * np.log(A/B) if not np.isnan(v).any()])
    elif dimensions > 1:
        M = compute_mixture(X, Y)
        P = X
        Q = Y
        _kldiv = lambda A, B: multivariate_kullback_leibler_divergence(A, B)
    else:
        raise ValueError(f"Dimensionality of X ({X_dimensions}) and Y ({Y_dimensions}) must be greater than 0")

    # print(M.shape, P.shape, Q.shape)
    kl_pm = _kldiv(P, M)
    # print(kl_pm)
    kl_qm = _kldiv(Q, M)
    # print(kl_qm)

    jensen_shannen_divergence = (kl_pm + kl_qm)/2

    # print('JSD: ' + str(np.sqrt(jensen_shannen_divergence)) + ' ' + str(P.shape) + ' ' + str(Q.shape))

    return np.sqrt(jensen_shannen_divergence)


def compute_mixture(P:np.array, Q:np.array):
    """Sample a mixture distribution between two probability distributions.

    Input
    -----
    P, Q : 2D arrays (sample_size, dimensions)
        Probability distributions of equal length that sum to 1
    """
    P_sample_size, P_dimensions = P.shape
    Q_sample_size, Q_dimensions = Q.shape

    half_sample_size = min(P_sample_size, Q_sample_size)
    i = np.argmin([P_sample_size, Q_sample_size])
    if i == 0:
        P_sample = P
        Q_sample_index = np.random.choice(list(range(Q_sample_size)), size=half_sample_size, replace=False)
        Q_sample = Q[Q_sample_index]
    elif i == 1:
        P_sample_index = np.random.choice(list(range(P_sample_size)), size=half_sample_size, replace=False)
        P_sample = P[P_sample_index]
        Q_sample = Q
    # print(P_sample.shape, Q_sample.shape)
    M_sample = np.concatenate((P_sample, Q_sample), axis=0)
    # print(M_sample.shape)
    # return list(M_sample.flatten())
    return M_sample


def kullback_leibler_divergence(P, Q):
    return np.sum(list(filter(lambda x: not np.isnan(x), P * np.log(P/Q))))

def multivariate_kullback_leibler_divergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    Parameters
    ----------
    x : 2D array (sample_size, dimensionality)
        Samples from distribution P, which typically represents the true
        distribution.
    y : 2D array (sample_size, dimensionality)
        Samples from distribution Q, which typically represents the approximate
        distribution.
    Returns
    -------
    out : float
        The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    Adapted from https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
    """
    #from scipy.spatial import KDTree
    #from sklearn.neighbors import KDTree
    from sklearn.neighbors import BallTree
    from scipy.spatial.distance import cosine, correlation
    from sklearn.metrics.pairwise import cosine_distances

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    x_sample_size, x_dimensions = x.shape
    y_sample_size, y_dimensions = y.shape

    assert(x_dimensions == y_dimensions), f"Both samples must have the same number of dimensions. x has {x_dimensions} dimensions, while y has {y_dimensions} dimensions."

    d = x_dimensions
    n = x_sample_size
    m = y_sample_size

    # Build a KD tree representation of the samples and find the nearest
    # neighbour of each point in x.
    xtree = BallTree(x, metric='euclidean')
    ytree = BallTree(y, metric='euclidean')

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    R = xtree.query(x, k=3)[0]
    r = np.array([min(list(filter(lambda x: x > 0, r))) for r in R])
    S = ytree.query(x, k=3)[0]
    s = np.array([min(list(filter(lambda x: x > 0, s))) for s in S])

    # FORMULA ESTIMATES KL DIV #
    kl_div = sum(np.log2(s/r)) * d / n + np.log2(m / (n - 1.))

    return max(kl_div, 0)

def spike_level_feature_array(unit: SpikeCluster, time_step):
    """Compute features for each spike in a unit.

    This function mutates a unit (SpikeCluster) by adding a spike-level feature dictionary to the unit.features attribute.

    Input
    -----
    unit : 2D array (spike_size, dimensions)
        Waveforms for each spike in a unit.
    time_step : float
        Time between samples in the waveform.

    Output
    ------
    features : dict
        Dictionary of features for each spike in the unit.
    """
    spikes = unit.get_spike_object_instances()
    #features = {}
    feature_array = []
    for spike in spikes:
        features = spike_features(spike, time_step)
        if features is not None:
            #spike.features = spike_features(spike, time_step)
            feature_vector = list(features.values())
            feature_array.append(feature_vector)
    # def get_spike_feature(i):
    #     spike = spikes[i]
    #     features = spike_features(spike, time_step)
    #     if features is not None:
    #         #spike.features = spike_features(spike, time_step)
    #         feature_vector = list(features.values())
    #         # feature_array.append(feature_vector)
    #         return feature_vector


    # feature_array = list(map(get_spike_feature, range(len(spikes))))
    # print(feature_array)
    feature_array = np.array(feature_array)
    # print('done casting')
    return feature_array


"""
['euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity', 'seuclidean', 'mahalanobis', 'wminkowski', 'hamming', 'canberra', 'braycurtis', 'matching', 'jaccard', 'dice', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'haversine', 'pyfunc']
"""