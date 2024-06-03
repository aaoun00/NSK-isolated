import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import pytest
import numpy as np
import _prototypes.unit_matcher.tests.read as read
from _prototypes.unit_matcher.unit import (
    multivariate_kullback_leibler_divergence
    ,jensen_shannon_distance
)

P = np.random.rand(5500, 24)
Q = np.random.rand(10000, 24) + 100

def test_multivariate_kullback_leibler_divergence():
    print("D(P||Q): ", multivariate_kullback_leibler_divergence(P, Q))
    print("D(Q||P): ", multivariate_kullback_leibler_divergence(Q, P))
    print("D(P||P): ", multivariate_kullback_leibler_divergence(P, P))


def test_jensen_shannon_divergence():
    print("JS(P,Q): ", jensen_shannon_distance(P, Q))
    print("JS(Q,P): ", jensen_shannon_distance(Q, P))
    print("JS(P,P): ", jensen_shannon_distance(P, P))
    #assert jensen_shannon_distance(P, Q) == jensen_shannon_distance(Q, P)

@pytest.mark.skip(reason="Not implemented yet")
def test_reduce_dimensionality():
    pass

@pytest.mark.skip(reason="Not implemented yet")
def extract_unit_features():
    pass