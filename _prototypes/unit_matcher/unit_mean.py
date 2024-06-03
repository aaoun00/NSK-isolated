import numpy as np

def mean_squared_difference_of_unit_means(unit1, unit2):
    """Return the mean squared difference between the mean waveforms of two units.

    Parameters
    ----------
    unit1 : int
        The first unit ID.
    unit2 : int
        The second unit ID.

    Returns
    -------
    float
        The mean squared difference between the mean waveforms of the two units.
    """
    return mean_squared_difference(unit_mean(unit1), unit_mean(unit2))


def mean_squared_difference(unit_mean_1, unit_mean_2):
    """Return the mean squared difference between two waveforms.

    Parameters
    ----------
    a : np.ndarray
        The first waveform.
    b : np.ndarray
        The second waveform.

    Returns
    -------
    float
        The mean squared difference between the two waveforms.
    """
    return np.mean((unit_mean_1 - unit_mean_2) ** 2)


def unit_mean(waveforms_dict, **kwargs):
    """Return the mean waveform of a single unit.

    Parameters
    ----------
    unit: ClusterInstance

    Returns
    -------
    np.ndarray
        The mean waveform of the unit.
    """
    mean_dict = dict()
    mean = list()
    for key, value in waveforms_dict.items():
        mean_dict[key] = np.mean(value, axis=0)
        mean.extend(mean_dict[key])
    return np.array(mean)