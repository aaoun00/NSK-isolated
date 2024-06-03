import numpy as np
import os, sys
import itertools
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.cell_remapping.src.backend import get_backend, NumpyBackend
from _prototypes.cell_remapping.src.utils import list_to_array, _get_ratemap_bucket_midpoints
from _prototypes.cell_remapping.src.masks import flat_disk_mask




# move to stats or stats_utils
def compute_null_spike_density(prev_pts, curr_pts, num_iterations, n_projections):
    np.random.seed(0)
    combined_spike_train = np.concatenate([prev_pts, curr_pts])
    num_spikes = len(prev_pts)
    num_total_spikes = len(combined_spike_train)

    # Repeat the combined spike train for the desired number of iterations
    repeated_spike_train = np.tile(combined_spike_train, (num_iterations, 1, 1))

    for i in range(num_iterations):
        # Shuffle the spike trains along the second axis (columns)
        np.random.shuffle(repeated_spike_train[i])

    # Extract the shuffled spike trains for A and B
    shuffled_prev = repeated_spike_train[:, :num_spikes]
    shuffled_curr = repeated_spike_train[:, num_spikes:num_total_spikes]

    emd_values = np.empty(num_iterations)
    for i in range(num_iterations):
        emd_values[i] = pot_sliced_wasserstein(shuffled_prev[i], shuffled_curr[i], n_projections=n_projections)

    return emd_values

# same as above
def compute_null_emd(spike_train_a, spike_train_b, num_iterations, bin_size):
    np.random.seed(0)
    combined_spike_train = np.concatenate([spike_train_a, spike_train_b])
    num_spikes = len(spike_train_a)
    num_total_spikes = len(combined_spike_train)

    # Repeat the combined spike train for the desired number of iterations
    repeated_spike_train = np.tile(combined_spike_train, (num_iterations, 1))

    for i in range(100):
        # Shuffle the spike trains along the second axis (columns)
        np.apply_along_axis(np.random.shuffle, axis=1, arr=repeated_spike_train)

    # Extract the shuffled spike trains for A and B
    shuffled_spike_train_a = repeated_spike_train[:, :num_spikes]
    shuffled_spike_train_b = repeated_spike_train[:, num_spikes:num_total_spikes]

    emd_values = np.empty(num_iterations)
    for i in range(num_iterations):
        emd_values[i] = compute_temporal_emd(shuffled_spike_train_a[i], shuffled_spike_train_b[i], bin_size)

    return emd_values


# change to temporal emd and move to wasserstein_distance
def compute_temporal_emd(spike_train_a, spike_train_b, bin_size, end_time=None):
    # Determine the start and end times for aligning the spike trains
    if end_time is not None:
        start_time = 0
        mx = np.max([np.max(spike_train_a), np.max(spike_train_b)])
        assert mx <= end_time, 'Max spike time {} greater than recording session length {}'.format(str(mx), str(end_time))
    else:
        start_time = np.min([np.min(spike_train_a), np.min(spike_train_b)])
        end_time = np.max([np.max(spike_train_a), np.max(spike_train_b)])
    # bins = np.arange(start_time, end_time + 1, bin_size)
    bins = np.arange(start_time, end_time + bin_size, bin_size)
    # Create aligned spike trains
    aligned_a, _ = np.histogram(spike_train_a, bins=bins)
    aligned_b, _ = np.histogram(spike_train_b, bins=bins)
    # Compute the cumulative distribution functions (CDFs)
    cdf_a = np.cumsum(aligned_a) / len(spike_train_a)
    cdf_b = np.cumsum(aligned_b) / len(spike_train_b)
    # Compute the EMD by integrating the absolute difference between CDFs
    emd = np.sum(np.abs(cdf_a - cdf_b))
    return emd


def compute_centroid_remapping(label_t, label_s, spatial_spike_train_t, spatial_spike_train_s, centroids_t, centroids_s, settings, cylinder=False):
    """
    _s/_t are source and target (i.e. prev/curr, ses1/ses2)

    label is map of same dims as ratemap with diff label for each blob/activation field

    spatial_spike_train is object to get ratemap/arena size from

    centroid is a pt of centroid centre on blob map, this is (64,64) ratemap bins, need to convert to height bucket midpoints, these are also (x,y) so need to flip to (y,x) for (row,col), (height,width)
    """
    # centroid_wass = np.zeros((len(np.unique(label_s)-1), len(centroid_t)))
    
    centroid_pairs = []

    target_rate_map_obj = spatial_spike_train_t.get_map('rate')

    if settings['normalizeRate']:
        target_map, _ = target_rate_map_obj.get_rate_map()
    else:
        _, target_map = target_rate_map_obj.get_rate_map()
                    
    # target_map, raw_target_map = target_rate_map_obj.get_rate_map()

    y, x = target_map.shape

    source_rate_map_obj = spatial_spike_train_s.get_map('rate')
    if settings['normalizeRate']:
        source_map, _ = source_rate_map_obj.get_rate_map()
    else:
        _, source_map = source_rate_map_obj.get_rate_map()

    # source_map, raw_source_map = source_rate_map_obj.get_rate_map()

    # # normalize rate map
    # total_mass = np.sum(source_map)
    # if total_mass != 1:
    #     source_map = source_map / total_mass

    # # normalize rate map
    # total_mass = np.sum(target_map)
    # if total_mass != 1:
    #     target_map = target_map / total_mass

    if cylinder:
        source_map = flat_disk_mask(source_map)
        target_map = flat_disk_mask(target_map)

    height_bucket_midpoints, width_bucket_midpoints = _get_ratemap_bucket_midpoints(spatial_spike_train_t.arena_size, y, x)

    field_wass = []
    test_field_wass = []
    centroid_wass = []
    bin_field_wass = []
    centroid_coords = []

    permute_dict = {}
    cumulative_dict = {}

    unq_s = np.unique(label_s)
    unq_t = np.unique(label_t)
    # drop nan from disk mask

    # for every unique blob label in source and target maps, compute wass
    # 1: to reemove 0 label
    for i in unq_s[1:]:
        rows, cols = np.where(label_s == i)
        source_ids = np.array([rows, cols]).T
        source_weights = np.array(list(map(lambda x: source_map[x[0], x[1]], source_ids)))
        source_weights = source_weights / np.sum(source_weights)
        # bin_source_map = np.zeros(source_map.shape)
        # bin_source_map[rows, cols] = 1
        
        height_source_pts = height_bucket_midpoints[rows]
        width_source_pts = width_bucket_midpoints[cols]
        source_pts = np.array([height_source_pts, width_source_pts]).T
        

        # for j in range(1,len(unq_t)):
        for j in unq_t[1:]:
            rows, cols = np.where(label_t == j)
            target_ids = np.array([rows, cols]).T
            target_weights = np.array(list(map(lambda x: target_map[x[0], x[1]], target_ids)))
            target_weights = target_weights / np.sum(target_weights)

            height_target_pts = height_bucket_midpoints[rows]
            width_target_pts = width_bucket_midpoints[cols]
            target_pts = np.array([height_target_pts, width_target_pts]).T


            # sliced wass on source pts and target pts (y,x) coordinates
            wass = pot_sliced_wasserstein(source_pts, target_pts, source_weights, target_weights, n_projections=settings['n_projections'])
            # sliced wass on source pts and target pts (y,x) coordinates
            bin_wass = pot_sliced_wasserstein(source_pts, target_pts, n_projections=settings['n_projections'])

            # bin_target_map = np.zeros(target_map.shape)
            # bin_target_map[rows, cols] = 1

            centroid_pairs.append([i,j])
            centroid_coords.append([centroids_s[i-1], centroids_t[j-1]])

            # euclidean distance between points
            c_wass = np.linalg.norm(np.array((centroids_t[j-1][0], centroids_t[j-1][1])) - np.array((centroids_s[i-1][0],centroids_s[i-1][1])))

            # testing cdist / linear sum approach
            # bin_source_map = bin_source_map/np.sum(bin_source_map)
            # bin_target_map = bin_target_map/np.sum(bin_target_map)

            # d = cdist(bin_source_map, bin_target_map)
            # assignment = linear_sum_assignment(d)
            # test_wass = d[assignment].sum() / spatial_spike_train_s.arena_size[0] 

            # test sliced wass on binary maps
            # bin_wass = pot_sliced_wasserstein(bin_source_map, bin_target_map)
        
            field_wass.append(wass)
            # test_field_wass.append(test_wass)
            centroid_wass.append(c_wass)
            bin_field_wass.append(bin_wass)

    # cumulative wasses
    rows, cols = np.where(label_s != 0)
    source_ids = np.array([rows, cols]).T
    height_source_pts = height_bucket_midpoints[rows]
    width_source_pts = width_bucket_midpoints[cols]
    source_pts = np.array([height_source_pts, width_source_pts]).T
    # bin_source_map = np.zeros(source_map.shape)
    # bin_source_map[rows, cols] = 1
    
    rows, cols = np.where(label_t != 0)
    target_ids = np.array([rows, cols]).T
    height_source_pts = height_bucket_midpoints[rows]
    width_source_pts = width_bucket_midpoints[cols]
    target_pts = np.array([height_source_pts, width_source_pts]).T
    # bin_target_map = np.zeros(target_map.shape)
    # bin_target_map[rows, cols] = 1

    source_weights = np.array(list(map(lambda x: source_map[x[0], x[1]], source_ids)))
    target_weights = np.array(list(map(lambda x: target_map[x[0], x[1]], target_ids)))

    source_weights = source_weights / np.sum(source_weights)
    target_weights = target_weights / np.sum(target_weights)
    # print(source_pts.shape, target_pts.shape, source_weights.shape, target_weights.shape, source_ids.shape, target_ids.shape)
    wass = pot_sliced_wasserstein(source_pts, target_pts, source_weights, target_weights, n_projections=settings['n_projections'])
    bin_wass = pot_sliced_wasserstein(source_pts, target_pts, n_projections=settings['n_projections'])
    c_wass = np.linalg.norm(np.array((np.mean(centroids_t, axis=0)[0], np.mean(centroids_t, axis=0)[1])) - np.array((np.mean(centroids_s, axis=0)[0],np.mean(centroids_s, axis=0)[1])))
    
    cumulative_dict['field_wass'] = wass
    cumulative_dict['binary_wass'] = bin_wass
    cumulative_dict['centroid_wass'] = c_wass

    permute_dict['field_wass'] = np.array(field_wass)
    permute_dict['pairs'] = np.array(centroid_pairs)
    permute_dict['coords'] = np.array(centroid_coords)
    # permute_dict['test_field_wass'] = np.array(test_field_wass)
    permute_dict['centroid_wass'] = np.array(centroid_wass)
    permute_dict['binary_wass'] = np.array(bin_field_wass)

    # sum(field_wass) is returned as cumualtive wass
    # return np.array(field_wass), np.array(centroid_pairs), np.sum(field_wass), test_field_wass, np.array(centroid_wass), np.array(bin_field_wass)
    return permute_dict, cumulative_dict


def single_point_wasserstein(object_coords, rate_map, arena_size, ids=None, density=False, density_map=None, use_pos_directly=False):
    """
    Computes wass distancees for map relative to single point coordinate

    Can pass in masked map using ids to denote bins to include
    """

    # gets rate map dimensions (64, 64)
    y, x = rate_map.shape
    # print(x, y, arena_size)

    height_bucket_midpoints, width_bucket_midpoints = _get_ratemap_bucket_midpoints(arena_size, y, x)
    # rate_map, _ = rate_map_obj.get_rate_map()
    # print(height_bucket_midpoints.shape, width_bucket_midpoints.shape)
    # print(height_bucket_midpoints, width_bucket_midpoints)

    # normalize rate map
    total_mass = np.sum(rate_map[rate_map == rate_map])
    if total_mass != 1:
        rate_map = rate_map / total_mass

    # these are the coordinates of the object on a 64,64 array so e.g. (0,32)
    if not use_pos_directly:
        if isinstance(object_coords, dict):
            obj_x = width_bucket_midpoints[object_coords['x']]
            obj_y = height_bucket_midpoints[object_coords['y']]
        else:
            obj_y = height_bucket_midpoints[object_coords[0]]
            obj_x = width_bucket_midpoints[object_coords[1]]
    else:
        obj_y = object_coords[0]
        obj_x = object_coords[1]

    if density:
        assert ids is None, "Cannot pass in ids with density=True as spike density is all raw spike positions not ratemap ids"

    if not density:
        # Batch apply euclidean distance metric, use itertools for permutations
        if ids is None:
            weighted_dists = list(map(lambda x: np.linalg.norm(np.array((obj_y, obj_x)) - np.array((height_bucket_midpoints[x[0]], width_bucket_midpoints[x[1]]))) * rate_map[x[0],x[1]], list(itertools.product(np.arange(0,y,1),np.arange(0,x,1)))))
        else:
            # pdct = itertools.product(np.arange(0,y,1),np.arange(0,x,1))
            # new_ids = set(list(pdct)).intersection(tuple(map(tuple, ids)))
            # print(len(ids), len(list(new_ids)))
            # normalize only ids mask to 1
            rate_map[ids[:,0], ids[:,1]] = rate_map[ids[:,0], ids[:,1]] / np.sum(rate_map[ids[:,0], ids[:,1]])
            # old_weighted_dists = list(map(lambda x, y: np.linalg.norm(np.array((obj_y, obj_x)) - np.array((height_bucket_midpoints[x], width_bucket_midpoints[y]))) * rate_map[x,y], np.array(list(new_ids)).T[0], np.array(list(new_ids)).T[1]))
            weighted_dists = list(map(lambda x, y: np.linalg.norm(np.array((obj_y, obj_x)) - np.array((height_bucket_midpoints[x], width_bucket_midpoints[y]))) * rate_map[x,y], ids[:,0], ids[:,1]))
            # assert np.array(old_weighted_dists).all() == np.array(weighted_dists).all(), "Something went wrong with the ids mask"
    elif density:
        weighted_dists = list(map(lambda x: np.linalg.norm(np.array((obj_y, obj_x)) - np.array((x[0], x[1]))) * 1/len(density_map), density_map))


    return np.sum(weighted_dists)


# https://stats.stackexchange.com/questions/404775/calculate-earth-movers-distance-for-two-grayscale-images
def sliced_wasserstein(X, Y, num_proj):
    dim = X.shape[1]
    estimates = []
    for _ in range(num_proj):
        # sample uniformly from the unit sphere
        dir = np.random.rand(dim)
        dir /= np.linalg.norm(dir)

        # project the data
        X_proj = X @ dir
        Y_proj = Y @ dir

        # compute 1d wasserstein
        estimates.append(wasserstein_distance(X_proj, Y_proj))
    return np.mean(estimates)

########################################################################################################################################################
""" Code below from POT """
########################################################################################################################################################

def wasserstein_1d(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    r"""
    Computes the 1 dimensional OT loss [15] between two (batched) empirical
    distributions

    .. math:
        OT_{loss} = \int_0^1 |cdf_u^{-1}(q)  cdf_v^{-1}(q)|^p dq

    It is formally the p-Wasserstein distance raised to the power p.
    We do so in a vectorized way by first building the individual quantile functions then integrating them.

    This function should be preferred to `emd_1d` whenever the backend is
    different to numpy, and when gradients over
    either sample positions or weights are required.

    Parameters
    ----------
    u_values: array-like, shape (n, ...)
        locations of the first empirical distribution
    v_values: array-like, shape (m, ...)
        locations of the second empirical distribution
    u_weights: array-like, shape (n, ...), optional
        weights of the first empirical distribution, if None then uniform weights are used
    v_weights: array-like, shape (m, ...), optional
        weights of the second empirical distribution, if None then uniform weights are used
    p: int, optional
        order of the ground metric used, should be at least 1 (see [2, Chap. 2], default is 1
    require_sort: bool, optional
        sort the distributions atoms locations, if False we will consider they have been sorted prior to being passed to
        the function, default is True

    Returns
    -------
    cost: float/array-like, shape (...)
        the batched EMD

    References
    ----------
    .. [15] Peyré, G., & Cuturi, M. (2018). Computational Optimal Transport.

    """

    assert p >= 1, "The OT loss is only valid for p>=1, {p} was given".format(p=p)

    if u_weights is not None and v_weights is not None:
        nx = get_backend(u_values, v_values, u_weights, v_weights)
    else:
        nx = get_backend(u_values, v_values)

    n = u_values.shape[0]
    m = v_values.shape[0]

    if u_weights is None:
        u_weights = nx.full(u_values.shape, 1. / n, type_as=u_values)
    elif u_weights.ndim != u_values.ndim:
        u_weights = nx.repeat(u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = nx.full(v_values.shape, 1. / m, type_as=v_values)
    elif v_weights.ndim != v_values.ndim:
        v_weights = nx.repeat(v_weights[..., None], v_values.shape[-1], -1)

    if require_sort:
        u_sorter = nx.argsort(u_values, 0)
        u_values = nx.take_along_axis(u_values, u_sorter, 0)

        v_sorter = nx.argsort(v_values, 0)
        v_values = nx.take_along_axis(v_values, v_sorter, 0)

        u_weights = nx.take_along_axis(u_weights, u_sorter, 0)
        v_weights = nx.take_along_axis(v_weights, v_sorter, 0)

    u_cumweights = nx.cumsum(u_weights, 0)
    v_cumweights = nx.cumsum(v_weights, 0)

    qs = nx.sort(nx.concatenate((u_cumweights, v_cumweights), 0), 0)
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)
    qs = nx.zero_pad(qs, pad_width=[(1, 0)] + (qs.ndim - 1) * [(0, 0)])
    delta = qs[1:, ...] - qs[:-1, ...]
    diff_quantiles = nx.abs(u_quantiles - v_quantiles)

    if p == 1:
        return nx.sum(delta * nx.abs(diff_quantiles), axis=0)
    return nx.sum(delta * nx.power(diff_quantiles, p), axis=0)


def pot_sliced_wasserstein(X_s, X_t, a=None, b=None, n_projections=50, p=2,
                                projections=None, seed=None, log=False):
    r"""
    Computes a Monte-Carlo approximation of the p-Sliced Wasserstein distance

    .. math::
        \mathcal{SWD}_p(\mu, \nu) = \underset{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}{\mathbb{E}}\left(\mathcal{W}_p^p(\theta_\# \mu, \theta_\# \nu)\right)^{\frac{1}{p}}


    where :

    - :math:`\theta_\# \mu` stands for the pushforwards of the projection :math:`X \in \mathbb{R}^d \mapsto \langle \theta, X \rangle`


    Parameters
    ----------
    X_s : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    X_t : ndarray, shape (n_samples_b, dim)
        samples in the target domain
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,), optional
        samples weights in the target domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    p: float, optional =
        Power p used for computing the sliced Wasserstein
    projections: shape (dim, n_projections), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    log: bool, optional
        if True, sliced_wasserstein_distance returns the projections used and their associated EMD.

    Returns
    -------
    cost: float
        Sliced Wasserstein Cost
    log : dict, optional
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_samples_a = 20
    >>> reg = 0.1
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> sliced_wasserstein_distance(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0

    References
    ----------

    .. [31] Bonneel, Nicolas, et al. "Sliced and radon wasserstein barycenters of measures." Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45
    """

    X_s, X_t = list_to_array(X_s, X_t)

    if a is not None and b is not None and projections is None:
        nx = get_backend(X_s, X_t, a, b)
    elif a is not None and b is not None and projections is not None:
        nx = get_backend(X_s, X_t, a, b, projections)
    elif a is None and b is None and projections is not None:
        nx = get_backend(X_s, X_t, projections)
    else:
        nx = get_backend(X_s, X_t)

    n = X_s.shape[0]
    m = X_t.shape[0]

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(X_s.shape[1],
                                                                                                      X_t.shape[1]))

    if a is None:
        a = nx.full(n, 1 / n, type_as=X_s)
    if b is None:
        b = nx.full(m, 1 / m, type_as=X_s)

    d = X_s.shape[1]

    if projections is None:
        projections = get_random_projections(d, n_projections, seed, backend=nx, type_as=X_s)

    X_s_projections = nx.dot(X_s, projections)
    X_t_projections = nx.dot(X_t, projections)

    projected_emd = wasserstein_1d(X_s_projections, X_t_projections, a, b, p=p)

    res = (nx.sum(projected_emd) / n_projections) ** (1.0 / p)
    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res

def get_random_projections(d, n_projections, seed=None, backend=None, type_as=None):
    r"""
    Generates n_projections samples from the uniform on the unit sphere of dimension :math:`d-1`: :math:`\mathcal{U}(\mathcal{S}^{d-1})`

    Parameters
    ----------
    d : int
        dimension of the space
    n_projections : int
        number of samples requested
    seed: int or RandomState, optional
        Seed used for numpy random number generator
    backend:
        Backend to ue for random generation

    Returns
    -------
    out: ndarray, shape (d, n_projections)
        The uniform unit vectors on the sphere

    Examples
    --------
    >>> n_projections = 100
    >>> d = 5
    >>> projs = get_random_projections(d, n_projections)
    >>> np.allclose(np.sum(np.square(projs), 0), 1.)  # doctest: +NORMALIZE_WHITESPACE
    True

    """

    if backend is None:
        nx = NumpyBackend()
    else:
        nx = backend

    if isinstance(seed, np.random.RandomState) and str(nx) == 'numpy':
        projections = seed.randn(d, n_projections)
    else:
        if seed is not None:
            nx.seed(seed)
        projections = nx.randn(d, n_projections, type_as=type_as)

    projections = projections / nx.sqrt(nx.sum(projections**2, 0, keepdims=True))
    return projections

def quantile_function(qs, cws, xs):
    r""" Computes the quantile function of an empirical distribution

    Parameters
    ----------
    qs: array-like, shape (n,)
        Quantiles at which the quantile function is evaluated
    cws: array-like, shape (m, ...)
        cumulative weights of the 1D empirical distribution, if batched, must be similar to xs
    xs: array-like, shape (n, ...)
        locations of the 1D empirical distribution, batched against the `xs.ndim - 1` first dimensions

    Returns
    -------
    q: array-like, shape (..., n)
        The quantiles of the distribution
    """
    nx = get_backend(qs, cws)
    n = xs.shape[0]
    if nx.__name__ == 'torch':
        # this is to ensure the best performance for torch searchsorted
        # and avoid a warninng related to non-contiguous arrays
        cws = cws.T.contiguous()
        qs = qs.T.contiguous()
    else:
        cws = cws.T
        qs = qs.T
    idx = nx.searchsorted(cws, qs).T
    return nx.take_along_axis(xs, nx.clip(idx, 0, n - 1), axis=0)